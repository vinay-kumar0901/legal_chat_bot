
import json
import csv
from pathlib import Path
from collections import defaultdict

from search_rag import search_legal_documents


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

EVALUATION_FILE = Path("evaluation_dataset.jsonl")
OUTPUT_FILE = Path("rag_evaluation_results.csv")

K = 5


# -----------------------------------------
# EXTRACT LEGAL REFERENCES
# -----------------------------------------

def extract_reference(item):
    """
    Extract the legal reference from a retrieved result.

    Adjust the keys below if your search_rag.py
    returns a different metadata structure.
    """

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):

        possible_keys = [
            "Jurisdictional Reference",
            "jurisdictional_reference",
            "reference",
            "legal_reference",
            "source",
            "id"
        ]

        for key in possible_keys:
            if item.get(key) is not None:
                return str(item[key]).strip()

        # Check nested metadata
        metadata = item.get("metadata", {})

        if isinstance(metadata, dict):
            for key in possible_keys:
                if metadata.get(key) is not None:
                    return str(metadata[key]).strip()

    raise ValueError(
        "Could not extract legal reference from result: "
        + repr(item)
    )


# -----------------------------------------
# RETRIEVE DOCUMENTS
# -----------------------------------------

def retrieve_documents(question):

    results = search_legal_documents(question)

    # If your search function returns a dictionary,
    # extract its result list.
    if isinstance(results, dict):

        for key in ["sources", "results", "documents"]:
            if key in results:
                results = results[key]
                break

    if results is None:
        return []

    return [
        extract_reference(item)
        for item in results
    ]


# -----------------------------------------
# EVALUATION METRICS
# -----------------------------------------

def precision_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    if k == 0:
        return 0.0

    relevant_count = sum(
        1 for item in retrieved_k
        if item in relevant
    )

    return relevant_count / k


def recall_at_k(retrieved, relevant, k):

    if not relevant:
        return 0.0

    retrieved_k = retrieved[:k]

    relevant_count = sum(
        1 for item in retrieved_k
        if item in relevant
    )

    return relevant_count / len(relevant)


def hit_rate_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    return float(
        any(item in relevant for item in retrieved_k)
    )


def reciprocal_rank(retrieved, relevant):

    for rank, item in enumerate(retrieved, start=1):

        if item in relevant:
            return 1 / rank

    return 0.0


def ndcg_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    dcg = 0.0

    for rank, item in enumerate(retrieved_k, start=1):

        if item in relevant:
            dcg += 1 / __import__("math").log2(rank + 1)

    ideal_relevant = min(len(relevant), k)

    idcg = sum(
        1 / __import__("math").log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )

    return dcg / idcg if idcg else 0.0


# -----------------------------------------
# RUN EVALUATION
# -----------------------------------------

def evaluate_rag():

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Missing evaluation file: {EVALUATION_FILE}"
        )

    results = []

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            test_case = json.loads(line)

            question = test_case["question"]

            relevant = set(
                str(ref).strip()
                for ref in test_case["relevant_refs"]
            )

            print(f"\nEvaluating question {line_number}:")
            print(question)

            retrieved = retrieve_documents(question)

            p = precision_at_k(retrieved, relevant, K)
            r = recall_at_k(retrieved, relevant, K)
            h = hit_rate_at_k(retrieved, relevant, K)
            mrr = reciprocal_rank(retrieved, relevant)
            ndcg = ndcg_at_k(retrieved, relevant, K)

            row = {
                "question": question,
                "relevant_refs": ", ".join(sorted(relevant)),
                "retrieved_refs": ", ".join(retrieved),
                f"precision@{K}": p,
                f"recall@{K}": r,
                f"hit_rate@{K}": h,
                "MRR": mrr,
                f"nDCG@{K}": ndcg
            }

            results.append(row)

            print("Retrieved:", retrieved)
            print(f"Precision@{K}: {p:.3f}")
            print(f"Recall@{K}: {r:.3f}")
            print(f"MRR: {mrr:.3f}")
            print(f"nDCG@{K}: {ndcg:.3f}")

    if not results:
        raise ValueError("No evaluation questions found.")

    # Save individual question results
    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    # Calculate average metrics
    metric_names = [
        f"precision@{K}",
        f"recall@{K}",
        f"hit_rate@{K}",
        "MRR",
        f"nDCG@{K}"
    ]

    averages = {}

    for metric in metric_names:

        averages[metric] = sum(
            row[metric] for row in results
        ) / len(results)

    print("\n========== FINAL RAG EVALUATION ==========")

    for metric, value in averages.items():
        print(f"{metric}: {value:.4f}")

    print("\nTotal questions:", len(results))
    print("Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    evaluate_rag()
