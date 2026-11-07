
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

import mock_llm

from policy_rag import load_policy_chunks, build_index, retrieve


EVALUATION_FILE = (
    PROJECT_ROOT / "data" / "raw" / "policy_eval.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "outputs" / "policy_eval_results.csv"
)



def clean_answer(answer):
    """Remove repeated phrases while preserving important policy facts."""
    import re

    answer = re.sub(r"\s+", " ", str(answer)).strip()
    words = answer.split()

    # Remove a repeated phrase only when the full phrase appears twice.
    # Start with longer phrases to avoid removing individual key words.
    for size in range(min(18, len(words) // 2), 3, -1):
        changed = True

        while changed:
            changed = False
            n = len(words)

            for start in range(n - size + 1):
                phrase = [
                    word.casefold()
                    for word in words[start:start + size]
                ]

                for later in range(start + size, n - size + 1):
                    candidate = [
                        word.casefold()
                        for word in words[later:later + size]
                    ]

                    if phrase == candidate:
                        del words[later:later + size]
                        changed = True
                        break

                if changed:
                    break

    return " ".join(words).strip()

def answer_question(question, chunks, vectorizer, matrix):
    """Retrieve relevant policy chunks and ask the offline mock LLM."""

    results = retrieve(
        query=question,
        chunks=chunks,
        vectorizer=vectorizer,
        matrix=matrix,
        top_k=3,
    )

    context_parts = []

    for result in results:
        context_parts.append(
            f"[{result['source']}]\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    # Keep exactly two messages: system and user.
    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT\n"
                "Answer the user's question using only the supplied "
                "policy context. Do not invent policy details."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n"
                f"QUESTION:\n{question}"
            ),
        },
    ]

    response = mock_llm.chat(messages)

    original_answer = response.choices[0].message.content
    answer = clean_answer(original_answer)

    return answer, results, response.usage.total_tokens


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    evaluation = pd.read_csv(
        EVALUATION_FILE,
        keep_default_na=False,
    )

    chunks = load_policy_chunks()
    vectorizer, matrix = build_index(chunks)

    evaluation_results = []

    # Reset mock LLM state before the evaluation run.
    mock_llm.reset()

    for _, row in evaluation.iterrows():
        question = str(row["question"])
        expected_doc = str(row["expected_doc"])
        expected_fact = str(row["expected_fact"])

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
            if is_answerable
            else None
        )

        # Keep the required case-sensitive expected-fact check.
        fact_found = (
            expected_fact in answer
            if is_answerable and expected_fact
            else None
        )

        fallback_correct = (
            answer.startswith("I could not find")
            if not is_answerable
            else None
        )

        evaluation_results.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "retrieved_docs": ";".join(retrieved_docs),
            "document_hit": document_hit,
            "answer": answer,
            "fact_found_in_answer": fact_found,
            "fallback_correct": fallback_correct,
            "tokens_used": tokens_used,
        })

        print("\nQuestion:", question)
        print("Expected document:", expected_doc)
        print("Retrieved documents:", ", ".join(retrieved_docs))
        print("Answer:", answer)
        print("Expected fact found in answer:", fact_found)

        if not is_answerable:
            print("Correct fallback:", fallback_correct)

    results_df = pd.DataFrame(evaluation_results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    answerable = results_df[
        results_df["expected_doc"] != "NONE"
    ]

    out_of_scope = results_df[
        results_df["expected_doc"] == "NONE"
    ]

    print("\n" + "=" * 60)
    print("Policy evaluation complete")
    print("Questions evaluated:", len(results_df))

    print(
        "Expected document retrieved:",
        f"{int(answerable['document_hit'].sum())}/{len(answerable)}",
    )

    print(
        "Expected fact found (case-sensitive):",
        f"{int(answerable['fact_found_in_answer'].sum())}"
        f"/{len(answerable)}",
    )

    print(
        "Correct out-of-scope fallbacks:",
        f"{int(out_of_scope['fallback_correct'].sum())}"
        f"/{len(out_of_scope)}",
    )

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
