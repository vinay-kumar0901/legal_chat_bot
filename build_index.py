import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_cleaned.csv"
INDEX_DIR = BASE_DIR / "index"
INDEX_FILE = INDEX_DIR / "legal.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"
CONFIG_FILE = INDEX_DIR / "index_config.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_COLUMNS = ["Jurisdictional Reference", "Type", "Description"]


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}\n"
            "Run Cleaned_data.py first, or run chatbot.py to execute the full pipeline."
        )

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df.columns = [str(column).strip() for column in df.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert required fields to strings and trim whitespace.
    for column in REQUIRED_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()

    # A usable record needs a reference and a description.
    df = df[
        df["Jurisdictional Reference"].ne("")
        & df["Description"].ne("")
    ].copy()

    # Remove duplicate legal records based on the fields used for retrieval.
    df = df.drop_duplicates(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable legal records remain after cleaning.")

    documents = []
    for row_number, row in df.iterrows():
        reference = row["Jurisdictional Reference"]
        category = row["Type"]
        description = row["Description"]
        searchable_text = (
            f"Legal reference: {reference}\n"
            f"Category: {category}\n"
            f"Description: {description}"
        )
        documents.append({
            "id": int(row_number),
            "reference": reference,
            "category": category,
            "description": description,
            "text": searchable_text,
        })

    texts = [document["text"] for document in documents]

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Inner product on normalized embeddings equals cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Write temporary files first, then replace the final files.
    temp_index = INDEX_DIR / "legal.faiss.tmp"
    temp_metadata = INDEX_DIR / "metadata.json.tmp"
    temp_config = INDEX_DIR / "index_config.json.tmp"

    faiss.write_index(index, str(temp_index))
    with temp_metadata.open("w", encoding="utf-8") as file:
        json.dump(documents, file, ensure_ascii=False, indent=2)

    config = {
        "embedding_model": MODEL_NAME,
        "record_count": len(documents),
        "embedding_dimension": int(embeddings.shape[1]),
        "source_file": DATA_FILE.name,
    }
    with temp_config.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    temp_index.replace(INDEX_FILE)
    temp_metadata.replace(METADATA_FILE)
    temp_config.replace(CONFIG_FILE)

    # Verify the written index.
    saved_index = faiss.read_index(str(INDEX_FILE))
    if saved_index.ntotal != len(documents):
        raise RuntimeError(
            f"Index verification failed: expected {len(documents)} records, "
            f"found {saved_index.ntotal}."
        )

    print("\nRAG index built successfully.")
    print(f"Indexed records: {saved_index.ntotal}")
    print(f"Index file: {INDEX_FILE}")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"Configuration file: {CONFIG_FILE}")


if __name__ == "__main__":
    main()
