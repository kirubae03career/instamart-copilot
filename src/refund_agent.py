
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm
from policy_rag import load_policy_chunks, build_index, retrieve


DB_PATH = PROJECT_ROOT / "data" / "instamart.db"
SCENARIOS_PATH = PROJECT_ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STATE_DIR = PROJECT_ROOT / "state"

RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
CHECKPOINT_PATH = STATE_DIR / "agent_checkpoint.json"

MAX_ITERATIONS = 6
TOKEN_LIMIT = 6000


def read_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl_once(path, record, unique_key):
    """Append a JSON record only when its unique key is not present."""
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if path.exists():
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    existing.append(json.loads(line))

    if any(item.get(unique_key) == record.get(unique_key)
           for item in existing):
        return

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


def get_order(order_id):
    """Read order information from SQLite."""
    import sqlite3

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT order_id, customer_id, status,
                   order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            LIMIT 1
            """,
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "Order not found", "order_id": order_id}

    return dict(row)


def search_policy(query):
    """Retrieve relevant chunks from the local policy documents."""
    chunks = load_policy_chunks()
    vectorizer, matrix = build_index(chunks)
    matches = retrieve(query, chunks, vectorizer, matrix, top_k=3)

    return {
        "results": [
            {"source": item["source"], "text": item["text"]}
            for item in matches
        ]
    }


def calculate_refund(order_amount, delivery_minutes, status):
    """Apply completed-order and late-delivery eligibility rules."""
    if str(status).lower() != "completed":
        amount = 0.0
        reason = "Only completed orders are eligible."
    elif delivery_minutes is None or pd.isna(delivery_minutes):
        amount = 0.0
        reason = "Delivery time is unavailable."
    elif float(delivery_minutes) <= 45:
        amount = 0.0
        reason = "Delivery was not more than 45 minutes."
    else:
        amount = round(float(order_amount), 2)
        reason = "Completed order delivered more than 45 minutes."

    return {
        "refund_inr": amount,
        "reason": reason,
    }


def issue_refund(order_id, refund_inr, scenario_id):
    """Record an automatic refund with an idempotency key."""
    record = {
        "scenario_id": scenario_id,
        "order_id": order_id,
        "amount_inr": round(float(refund_inr), 2),
        "status": "issued",
    }
    append_jsonl_once(LEDGER_PATH, record, "scenario_id")
    return {"refund_inr": record["amount_inr"], "status": "issued"}


def escalate_to_human(order_id, refund_inr, scenario_id):
    """Record a refund requiring human approval."""
    return {
        "scenario_id": scenario_id,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "status": "escalated",
        "reason": "Refund above INR 1000 requires human approval.",
    }


def execute_tool(name, arguments, scenario_id, order):
    """Dispatch a mock LLM tool call to a local Python function."""
    if name == "get_order":
        return get_order(arguments["order_id"])

    if name == "search_policy":
        return search_policy(arguments["query"])

    if name == "calculate_refund":
        if order is None or "error" in order:
            return {"refund_inr": 0.0, "reason": "Order unavailable."}

        return calculate_refund(
            order_amount=arguments["order_amount"],
            delivery_minutes=arguments["delivery_minutes"],
            status=order["status"],
        )

    if name == "issue_refund":
        return issue_refund(
            arguments["order_id"],
            arguments["refund_inr"],
            scenario_id,
        )

    if name == "escalate_to_human":
        return escalate_to_human(
            arguments["order_id"],
            arguments["refund_inr"],
            scenario_id,
        )

    return {"error": f"Unknown tool: {name}"}


def get_order_from_messages(messages):
    for message in messages:
        if message.get("role") == "tool" and message.get("name") == "get_order":
            return json.loads(message["content"])
    return None


def get_terminal_status(answer):
    text = (answer or "").upper()

    if text.startswith("REFUND:"):
        return "REFUND"
    if text.startswith("ESCALATE:"):
        return "ESCALATE"
    if text.startswith("DENIED:"):
        return "DENIED"
    if text.startswith("NOT_FOUND:"):
        return "NOT_FOUND"
    if text.startswith("NEEDS_INFO:"):
        return "NEEDS_INFO"

    return "FALLBACK_INVALID_RESPONSE"


def save_checkpoint(checkpoint):
    write_json(CHECKPOINT_PATH, checkpoint)


def run_scenario(
    scenario,
    checkpoint,
    *,
    token_limit=TOKEN_LIMIT,
    simulate_crash=False,
    simulate_timeout=False,
):
    """Run or resume one scenario, saving progress after every tool."""
    scenario_id = str(scenario["scenario_id"])
    state = checkpoint["scenarios"].get(scenario_id)

    if state and state.get("completed"):
        return state["result"]

    if state is None:
        order_match = re.search(r"ORD-\d{6}", str(scenario["message"]))
        user_message = str(scenario["message"])

        # S7 deliberately tests the maximum-iteration fallback.
        if scenario_id == "S7" and "[LOOP]" not in user_message:
            user_message += " [LOOP]"

        state = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "AGENT\n"
                        "Use tools to check the order, consult the policy, "
                        "calculate eligibility, then refund or escalate."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            "tokens_used": 0,
            "tool_names": [],
            "iterations": 0,
            "completed": False,
            "order_id": order_match.group(0) if order_match else None,
        }
        checkpoint["scenarios"][scenario_id] = state
        save_checkpoint(checkpoint)

    if simulate_timeout:
        result = {
            "scenario_id": scenario_id,
            "status": "FALLBACK_TIMEOUT",
            "answer": "Human review required after simulated timeout.",
            "tokens_used": state["tokens_used"],
            "tool_names": state["tool_names"],
        }
        state["completed"] = True
        state["result"] = result
        save_checkpoint(checkpoint)
        return result

    while state["iterations"] < MAX_ITERATIONS:
        if state["tokens_used"] >= token_limit:
            result = {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "answer": "Human review required: token limit reached.",
                "tokens_used": state["tokens_used"],
                "tool_names": state["tool_names"],
            }
            state["completed"] = True
            state["result"] = result
            save_checkpoint(checkpoint)
            return result

        response = mock_llm.chat(state["messages"])
        state["tokens_used"] += response.usage.total_tokens
        state["iterations"] += 1

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            answer = message.content or ""
            result = {
                "scenario_id": scenario_id,
                "status": get_terminal_status(answer),
                "answer": answer,
                "tokens_used": state["tokens_used"],
                "tool_names": state["tool_names"],
            }
            state["completed"] = True
            state["result"] = result
            save_checkpoint(checkpoint)
            return result

        # Save the assistant's tool request before executing the tool.
        state["messages"].append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in tool_calls
            ],
        })
        save_checkpoint(checkpoint)

        for call in tool_calls:
            # Resume safety: a completed tool call ID is already represented
            # by its tool message. Never execute it a second time.
            completed_ids = {
                item.get("tool_call_id")
                for item in state["messages"]
                if item.get("role") == "tool"
            }
            if call.id in completed_ids:
                continue

            order = get_order_from_messages(state["messages"])
            arguments = call.arguments

            tool_result = execute_tool(
                call.name,
                arguments,
                scenario_id,
                order,
            )

            state["tool_names"].append(call.name)
            state["messages"].append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(tool_result, default=str),
            })
            save_checkpoint(checkpoint)

            # S1 simulates a crash after its first two tool executions.
            # The checkpoint is left intact for the next run to resume.
            if (
                simulate_crash
                and scenario_id == "S1"
                and len(state["tool_names"]) == 2
            ):
                print("Simulated crash after two tools for S1.")
                return {
                    "scenario_id": scenario_id,
                    "status": "INTERRUPTED_SIMULATED_CRASH",
                    "answer": "Checkpoint saved; rerun to resume.",
                    "tokens_used": state["tokens_used"],
                    "tool_names": state["tool_names"],
                }

    result = {
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "answer": "Human review required: maximum iterations reached.",
        "tokens_used": state["tokens_used"],
        "tool_names": state["tool_names"],
    }
    state["completed"] = True
    state["result"] = result
    save_checkpoint(checkpoint)
    return result


def write_run_log(checkpoint):
    """Rebuild the run log from completed checkpoint entries."""
    records = [
        state["result"]
        for state in checkpoint["scenarios"].values()
        if state.get("completed") and state.get("result")
    ]

    with RUNS_PATH.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, default=str) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    scenarios_df = pd.read_csv(SCENARIOS_PATH)
    checkpoint = read_json(CHECKPOINT_PATH, {"scenarios": {}})

    # First, demonstrate a crash after two tools on S1.
    s1_rows = scenarios_df[scenarios_df["scenario_id"] == "S1"]
    if not s1_rows.empty:
        print("\n--- S1 crash/resume demonstration ---")
        interrupted = run_scenario(
            s1_rows.iloc[0].to_dict(),
            checkpoint,
            simulate_crash=True,
        )
        print("First attempt:", interrupted["status"])

        resumed = run_scenario(
            s1_rows.iloc[0].to_dict(),
            checkpoint,
        )
        print("Resumed result:", resumed["status"])
        print("S1 tools:", resumed["tool_names"])
        print("S1 tool execution count:", len(resumed["tool_names"]))

    # Run the remaining scenarios, resuming saved work if needed.
    print("\n--- Running agent scenarios ---")
    for _, row in scenarios_df.iterrows():
        scenario = row.to_dict()
        scenario_id = str(scenario["scenario_id"])

        if scenario_id == "S1":
            continue

        result = run_scenario(scenario, checkpoint)
        print(scenario_id, "->", result["status"])

    # Separate deterministic demonstrations for the two other fallbacks.
    if not scenarios_df.empty:
        demo = scenarios_df.iloc[0].to_dict()

        token_result = run_scenario(
            {**demo, "scenario_id": "DEMO_TOKEN_LIMIT"},
            checkpoint,
            token_limit=1,
        )
        print("Token demo:", token_result["status"])

        timeout_result = run_scenario(
            {**demo, "scenario_id": "DEMO_TIMEOUT"},
            checkpoint,
            simulate_timeout=True,
        )
        print("Timeout demo:", timeout_result["status"])

    write_run_log(checkpoint)

    print("\nRun log:", RUNS_PATH)
    print("Refund ledger:", LEDGER_PATH)
    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
