import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"
INDEX_FILE = INDEX_DIR / "legal.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"
CONFIG_FILE = INDEX_DIR / "index_config.json"

required_files = [INDEX_FILE, METADATA_FILE, CONFIG_FILE]
missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        "The RAG index is missing. Run chatbot.py to build it first.\n"
        + "\n".join(missing_files)
    )

index = faiss.read_index(str(INDEX_FILE))

with METADATA_FILE.open("r", encoding="utf-8") as file:
    documents = json.load(file)

with CONFIG_FILE.open("r", encoding="utf-8") as file:
    config = json.load(file)

if index.ntotal != len(documents):
    raise RuntimeError(
        f"Index/metadata mismatch: index has {index.ntotal} vectors, "
        f"but metadata has {len(documents)} records. Rebuild the index."
    )

model = SentenceTransformer(config["embedding_model"])


def search_legal_documents(question, top_k=5):
    """Return the top matching legal records for a question."""
    if not isinstance(question, str) or not question.strip():
        return []

    if index.ntotal == 0:
        return []

    try:
        top_k = max(1, int(top_k))
    except (TypeError, ValueError):
        top_k = 5
    top_k = min(top_k, index.ntotal)

    query_embedding = model.encode(
        [question.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    scores, indices = index.search(query_embedding, top_k)
    results = []

    for score, document_index in zip(scores[0], indices[0]):
        if document_index < 0:
            continue
        document = documents[int(document_index)]
        results.append({
            "reference": document.get("reference", ""),
            "category": document.get("category", ""),
            "description": document.get("description", ""),
            "similarity": float(score),
        })

    return results


if __name__ == "__main__":
    print("Legal RAG Search — type 'exit' to stop.")
    while True:
        question = input("\nEnter your legal question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        for number, result in enumerate(search_legal_documents(question), start=1):
            print(f"\n--- Result {number} ---")
            print("Reference:", result["reference"])
            print("Category:", result["category"])
            print("Description:", result["description"])
            print("Cosine similarity:", round(result["similarity"], 4))
