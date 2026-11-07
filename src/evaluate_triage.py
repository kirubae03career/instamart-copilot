
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]

tickets = pd.read_csv(ROOT / "data" / "raw" / "tickets.csv")

for mode in ["zero_shot", "few_shot"]:
    results = pd.read_csv(ROOT / "outputs" / f"triage_{mode}.csv")

    # Compare only tickets with a valid classification.
    comparison = results.merge(
        tickets[["ticket_id", "true_category"]],
        on="ticket_id",
        how="left",
    )

    classified = comparison[
        comparison["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ].copy()

    accuracy = accuracy_score(
        classified["true_category"],
        classified["category"],
    )

    print(f"\n{'=' * 40}")
    print(f"Mode: {mode}")
    print(f"Classified tickets: {len(classified)}")
    print(f"Correct predictions: "
          f"{(classified['category'] == classified['true_category']).sum()}")
    print(f"Accuracy: {accuracy:.2%}")

    labels = ["billing", "delivery", "technical", "fraud"]
    matrix = confusion_matrix(
        classified["true_category"],
        classified["category"],
        labels=labels,
    )

    print("\nConfusion matrix")
    print("Rows = actual category; columns = predicted category")
    print(pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ))
