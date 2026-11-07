
"""Part C: Support ticket privacy checks and guardrails."""

import re
import math
import pandas as pd

INPUT_PATH = "data/raw/tickets.csv"
OUTPUT_PATH = "outputs/ticket_guardrail_check.csv"

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_ID_PATTERN = re.compile(r"ORD-\d{6}")

def redact_pii(text):
    """Redact email, phone, and order ID in the required order."""
    text = "" if pd.isna(text) else str(text)

    text, emails = EMAIL_PATTERN.subn("[EMAIL]", text)
    text, phones = PHONE_PATTERN.subn("[PHONE]", text)
    text, order_ids = ORDER_ID_PATTERN.subn("[ORDER_ID]", text)

    counts = {
        "email_redactions": emails,
        "phone_redactions": phones,
        "order_id_redactions": order_ids,
    }
    return text, counts


def est_tokens(text):
    """Use the supplied mock LLM's token estimator."""
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import mock_llm
    return mock_llm.est_tokens(text)


def inspect_tickets():
    df = pd.read_csv(INPUT_PATH)
    required = {
        "ticket_id", "customer_id", "order_id", "text", "true_category"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    total_counts = {
        "email_redactions": 0,
        "phone_redactions": 0,
        "order_id_redactions": 0,
    }

    for _, ticket in df.iterrows():
        original_text = "" if pd.isna(ticket["text"]) else str(ticket["text"])
        redacted_text, counts = redact_pii(original_text)

        for key in total_counts:
            total_counts[key] += counts[key]

        if not original_text.strip():
            status = "skipped_empty"
        elif est_tokens(original_text) > 400:
            status = "flagged_long"
        else:
            status = "ready"

        # Guardrail statuses are decided before any model call.
        rows.append({
            "ticket_id": ticket["ticket_id"],
            "status": status,
            "estimated_tokens": est_tokens(original_text),
            "redacted_text": redacted_text,
            **counts,
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("Tickets checked:", len(result))
    print("Guardrail statuses:")
    print(result["status"].value_counts().to_string())
    print("PII redaction totals:")
    for key, value in total_counts.items():
        print(f"{key}: {value}")
    print("Saved:", OUTPUT_PATH)

    # Confirm no specified PII patterns remain in the redacted text.
    for pattern in (EMAIL_PATTERN, PHONE_PATTERN, ORDER_ID_PATTERN):
        assert not result["redacted_text"].fillna("").apply(
            lambda value: bool(pattern.search(value))
        ).any(), f"Residual PII found for {pattern.pattern}"

    print("PII residual checks: PASSED")

if __name__ == "__main__":
    inspect_tickets()


# ---------- Classification prompts and response validation ----------

import json
import random
import time

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

ZERO_SHOT_SYSTEM_PROMPT = """
CLASSIFY each customer support ticket into exactly one category:
billing, delivery, technical, or fraud.

Return JSON only, with exactly these keys:
{"category": "...", "reason": "..."}

Do not include markdown or any text outside the JSON.
""".strip()

FEW_SHOT_SYSTEM_PROMPT = """
CLASSIFY each customer support ticket into exactly one category:
billing, delivery, technical, or fraud.

Return JSON only, with exactly these keys:
{"category": "...", "reason": "..."}

Do not include markdown or any text outside the JSON.

EXAMPLES:
Ticket: I was charged twice for the same purchase.
Output: {"category": "billing", "reason": "duplicate charge"}

Ticket: My package has not arrived yet.
Output: {"category": "delivery", "reason": "package not received"}

Ticket: The app freezes whenever I try to log in.
Output: {"category": "technical", "reason": "app malfunction"}

Ticket: There is an unauthorized transaction on my card.
Output: {"category": "fraud", "reason": "unauthorized transaction"}
""".strip()


def validate_model_response(content):
    """
    Return (status, category, reason).

    Valid JSON with an allowed category -> classified_json.
    Otherwise, accept exactly one allowed category word -> classified_keyword.
    If neither rule succeeds -> flagged_invalid_output.
    """
    content = "" if content is None else str(content)

    try:
        parsed = json.loads(content)
        if (
            isinstance(parsed, dict)
            and isinstance(parsed.get("category"), str)
            and parsed["category"].lower() in ALLOWED_CATEGORIES
        ):
            return (
                "classified_json",
                parsed["category"].lower(),
                str(parsed.get("reason", "")),
            )
    except (json.JSONDecodeError, TypeError):
        pass

    words = re.findall(r"[a-z]+", content.lower())
    category_words = [word for word in words if word in ALLOWED_CATEGORIES]
    unique_categories = set(category_words)

    if len(unique_categories) == 1:
        return ("classified_keyword", next(iter(unique_categories)), "")

    return ("flagged_invalid_output", "", "")


# ---------- Retry handling ----------

BASE_DELAY = 0.05
MAX_RETRIES = 3


def call_with_retry(messages, ticket_id, retry_log):
    """
    Retry transient mock LLM failures up to 3 times.
    The initial call is followed by at most 3 retries.
    """
    import mock_llm

    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise

            wait_seconds = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            retry_log.append({
                "ticket_id": ticket_id,
                "retry_number": retry_number + 1,
                "wait_seconds": wait_seconds,
            })

            print(
                f"Ticket {ticket_id}: retry {retry_number + 1}, "
                f"waiting {wait_seconds:.4f}s"
            )
            time.sleep(wait_seconds)


# ---------- Classification runner ----------

# ---------- Classification runner ----------

def run_classification(mode, token_budget=None, output_path=None):
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import mock_llm

    if mode not in {"zero_shot", "few_shot"}:
        raise ValueError("mode must be 'zero_shot' or 'few_shot'")

    if token_budget is not None and token_budget < 0:
        raise ValueError("token_budget must be non-negative")

    system_prompt = (
        ZERO_SHOT_SYSTEM_PROMPT
        if mode == "zero_shot"
        else FEW_SHOT_SYSTEM_PROMPT
    )

    input_path = Path(INPUT_PATH)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    df = pd.read_csv(input_path)
    mock_llm.reset()

    results = []
    retry_log = []
    total_tokens = 0

    for _, ticket in df.iterrows():
        original_text = (
            "" if pd.isna(ticket["text"]) else str(ticket["text"])
        )
        redacted_text, _ = redact_pii(original_text)

        # Guardrail 1: empty tickets never reach the model.
        if not original_text.strip():
            results.append({
                "ticket_id": ticket["ticket_id"],
                "status": "skipped_empty",
                "category": "",
                "reason": "",
                "tokens_used": 0,
            })
            continue

        # Guardrail 2: long tickets never reach the model.
        if est_tokens(original_text) > 400:
            results.append({
                "ticket_id": ticket["ticket_id"],
                "status": "flagged_long",
                "category": "",
                "reason": "",
                "tokens_used": 0,
            })
            continue

        # Budget check happens before every model call.
        if (
            token_budget is not None
            and total_tokens >= token_budget
        ):
            results.append({
                "ticket_id": ticket["ticket_id"],
                "status": "budget_exhausted",
                "category": "",
                "reason": "",
                "tokens_used": 0,
            })
            continue

        # The model receives only the redacted ticket text.
        assert redacted_text == redact_pii(original_text)[0]
        assert not EMAIL_PATTERN.search(redacted_text)
        assert not PHONE_PATTERN.search(redacted_text)
        assert not ORDER_ID_PATTERN.search(redacted_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text},
        ]

        response = call_with_retry(
            messages, ticket["ticket_id"], retry_log
        )

        content = response.choices[0].message.content
        status, category, reason = validate_model_response(content)
        used_tokens = int(response.usage.total_tokens)
        total_tokens += used_tokens

        results.append({
            "ticket_id": ticket["ticket_id"],
            "status": status,
            "category": category,
            "reason": reason,
            "tokens_used": used_tokens,
        })

    results_df = pd.DataFrame(results)

    retry_df = pd.DataFrame(
        retry_log,
        columns=["ticket_id", "retry_number", "wait_seconds"],
    )

    if output_path is None:
        output_file = project_root / "outputs" / f"triage_{mode}.csv"
    else:
        output_file = Path(output_path)
        if not output_file.is_absolute():
            output_file = project_root / output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    retry_path = project_root / "outputs" / f"retry_log_{mode}.csv"
    retry_path.parent.mkdir(parents=True, exist_ok=True)
    retry_df.to_csv(retry_path, index=False)

    eligible = results_df[
        results_df["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    print(f"\nMode: {mode}")
    print("Status counts:")
    print(results_df["status"].value_counts().to_string())
    print("Classified tickets:", len(eligible))
    print("Total tokens:", int(results_df["tokens_used"].sum()))
    print(
        "Tokens per classified ticket:",
        round(eligible["tokens_used"].sum() / len(eligible), 2)
        if len(eligible) else 0,
    )
    print("Retry attempts:", len(retry_log))
    print("Saved:", output_file)
    print("Saved:", retry_path)

    return results_df