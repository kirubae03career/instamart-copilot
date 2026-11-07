
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.ticket_triage import run_classification

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1: Demonstrate the zero-shot token budget.
    budget_results = run_classification(
        "zero_shot",
        token_budget=6000,
        output_path=OUTPUT_DIR / "triage_budget_demo.csv",
    )

    exhausted = int(
        (budget_results["status"] == "budget_exhausted").sum()
    )
    print(f"\nBudget demo: 6000 tokens")
    print(f"Tickets stopped by budget: {exhausted}")

    if exhausted == 0:
        raise RuntimeError(
            "No tickets exhausted the budget. Try a lower demo budget."
        )

    # Part 2: Run few-shot classification with the required budget.
    run_classification(
        "few_shot",
        token_budget=100000,
        output_path=OUTPUT_DIR / "triage_results.csv",
    )

    # Enrich results with customer IDs from the original tickets.
    results = pd.read_csv(OUTPUT_DIR / "triage_results.csv")
    tickets = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "tickets.csv")
    segments = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")

    results = results.merge(
        tickets[["ticket_id", "customer_id"]],
        on="ticket_id",
        how="left",
        validate="one_to_one",
    )

    results = results.merge(
        segments[["customer_id", "segment_name", "is_premium"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    results["segment_name"] = results["segment_name"].fillna("")

    # Missing customers are treated as not premium.
    premium = (
        results["is_premium"]
        .map({
            True: True,
            False: False,
            "True": True,
            "False": False,
            1: True,
            0: False,
        })
        .fillna(False)
    )

    results["priority"] = "NORMAL"
    results["route"] = "support_queue"

    needs_human = results["status"].isin([
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    ])

    is_fraud = results["category"].eq("fraud")

    results.loc[needs_human, "route"] = "human_review"

    results.loc[
        ~needs_human & is_fraud, "route"
    ] = "security_team"

    results.loc[
        ~needs_human & is_fraud, "priority"
    ] = "HIGH"

    premium_route = ~needs_human & ~is_fraud & premium

    results.loc[
        premium_route, "route"
    ] = "premium_desk"

    results.loc[
        premium_route, "priority"
    ] = "HIGH"

    # Keep exactly the columns required by the project brief.
    columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    results = results[columns]

    output_path = OUTPUT_DIR / "triage_results.csv"
    results.to_csv(output_path, index=False)

    # Final validation.
    assert len(results) == 164, (
        f"Expected 164 tickets, found {len(results)}"
    )
    assert results["ticket_id"].is_unique
    assert results["route"].notna().all()
    assert set(results["route"]).issubset({
        "human_review",
        "security_team",
        "premium_desk",
        "support_queue",
    })

    print("\nFinal triage routing:")
    print(results["route"].value_counts().to_string())
    print("\nFinal priority counts:")
    print(results["priority"].value_counts().to_string())
    print(f"\nValidated {len(results)} unique tickets.")
    print(f"Saved: {output_path}")
    print(f"Saved: {OUTPUT_DIR / 'triage_budget_demo.csv'}")


if __name__ == "__main__":
    main()
