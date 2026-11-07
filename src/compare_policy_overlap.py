from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from policy_rag import load_policy_chunks, build_index
from policy_assistant import answer_question

EVALUATION_FILE = PROJECT_ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "policy_eval_zero_overlap_results.csv"


def main():
    evaluation = pd.read_csv(EVALUATION_FILE, keep_default_na=False)

    # Build the index without overlapping words.
    chunks = load_policy_chunks(overlap=0)
    vectorizer, matrix = build_index(chunks)

    print("Chunk overlap: 0 words")
    print("Total chunks:", len(chunks))

    results = []

    for _, row in evaluation.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        answer, retrieved, tokens_used = answer_question(
            question,
            chunks,
            vectorizer,
            matrix,
        )

        retrieved_docs = sorted({
            item["source"] for item in retrieved
        })

        is_answerable = expected_doc != "NONE"

        document_hit = (
            expected_doc in retrieved_docs
            if is_answerable else None
        )

        # Check the expected fact case-sensitively.
        fact_found = (
            expected_fact in answer
            if is_answerable and expected_fact else None
        )

        fallback_correct = (
            answer.startswith("I could not find")
            if not is_answerable else None
        )

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "retrieved_docs": ";".join(retrieved_docs),
            "document_hit": document_hit,
            "expected_fact": expected_fact,
            "fact_found_in_answer": fact_found,
            "fallback_correct": fallback_correct,
            "answer": answer,
            "tokens_used": tokens_used,
        })

    results_df = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    answerable = results_df[
        results_df["expected_doc"] != "NONE"
    ]
    out_of_scope = results_df[
        results_df["expected_doc"] == "NONE"
    ]

    print("\n=== ZERO-OVERLAP EVALUATION ===")
    print("Total questions:", len(results_df))
    print(
        "Expected document retrieved:",
        f"{int(answerable['document_hit'].sum())}/{len(answerable)}",
    )
    print(
        "Expected fact found (case-sensitive):",
        f"{int(answerable['fact_found_in_answer'].sum())}/{len(answerable)}",
    )
    print(
        "Correct out-of-scope fallbacks:",
        f"{int(out_of_scope['fallback_correct'].sum())}/{len(out_of_scope)}",
    )

    failed = answerable[
        (answerable["document_hit"] != True)
        | (answerable["fact_found_in_answer"] != True)
    ]

    print("\nAnswerable questions that failed:")
    if failed.empty:
        print("None. Inspect the results and compare with overlapping chunks.")
    else:
        for _, row in failed.iterrows():
            print("\nQuestion:", row["question"])
            print("Expected fact:", row["expected_fact"])
            print("Retrieved document:", row["retrieved_docs"])
            print("Answer:", row["answer"])

    print("\nSaved results:", OUTPUT_FILE)


if __name__ == "__main__":
    main()