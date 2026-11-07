from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = PROJECT_ROOT / "data" / "policies"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10


def load_policy_chunks(overlap=CHUNK_OVERLAP):
    """Load policy files and split them into configurable word chunks.

    Args:
        overlap: Number of words shared between consecutive chunks.
                 Defaults to 10; use 0 for the zero-overlap evaluation.
    """
    if not isinstance(overlap, int) or not 0 <= overlap < CHUNK_SIZE:
        raise ValueError(
            f"overlap must be an integer from 0 to {CHUNK_SIZE - 1}"
        )

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy directory not found: {POLICY_DIR}"
        )

    chunks = []
    step = CHUNK_SIZE - overlap

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        words = text.split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + CHUNK_SIZE]

            if not chunk_words:
                break

            chunks.append({
                "source": path.name,
                "chunk_id": len(chunks) + 1,
                "text": " ".join(chunk_words),
            })

            # Stop after the first chunk that reaches the document end.
            if start + CHUNK_SIZE >= len(words):
                break

    return chunks


def build_index(chunks):
    """Create a TF-IDF index from the policy chunks."""
    if not chunks:
        raise ValueError("No policy chunks found.")

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    return vectorizer, matrix


def retrieve(query, chunks, vectorizer, matrix, top_k=3):
    """Return the most relevant policy chunks using cosine similarity."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            **chunks[index],
            "score": round(float(scores[index]), 4),
        })

    return results


def main():
    # The normal project configuration remains 40 words / 10 overlap.
    chunks = load_policy_chunks()
    vectorizer, matrix = build_index(chunks)

    print("Policy files:", len(list(POLICY_DIR.glob("*.txt"))))
    print("Total chunks:", len(chunks))
    print("Chunk size:", CHUNK_SIZE, "words")
    print("Chunk overlap:", CHUNK_OVERLAP, "words")

    query = "refund policy late delivery"
    results = retrieve(query, chunks, vectorizer, matrix, top_k=3)

    print(f"\nQuery: {query}")
    print("\nTop 3 retrieved chunks:")

    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. {result['source']}")
        print("Chunk ID:", result["chunk_id"])
        print("Similarity:", result["score"])
        print("Text:", result["text"])

    print("\nSource file counts:")
    for source in sorted({chunk["source"] for chunk in chunks}):
        count = sum(chunk["source"] == source for chunk in chunks)
        print(f"{source}: {count} chunks")

    refund_results = [
        result for result in results
        if result["source"] == "refund_policy.txt"
    ]

    if CHUNK_OVERLAP == 10:
        if len(chunks) != 15:
            raise AssertionError(
                f"Expected 15 chunks, got {len(chunks)}"
            )

        if len(refund_results) != 3:
            raise AssertionError(
                "Expected all top 3 results to come from refund_policy.txt"
            )

        print("\nPASS: 15 chunks created.")
        print("PASS: All top 3 results came from refund_policy.txt.")


if __name__ == "__main__":
    main()