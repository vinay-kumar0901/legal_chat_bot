from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "dataset.csv"
OUTPUT_FILE = BASE_DIR / "dataset_cleaned.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {INPUT_FILE}\n"
            "Place dataset.csv in the same folder as Cleaned_data.py."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print(f"Original dataset shape: {df.shape}")
    print("\nMissing values before cleaning:")
    print(df.isna().sum().to_string())

    # Normalize column names and whitespace before removing duplicates.
    df.columns = [str(column).strip() for column in df.columns]

    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    duplicates_before = int(df.duplicated().sum())
    df = df.drop_duplicates(keep="first").reset_index(drop=True)

    print(f"\nDuplicate rows removed: {duplicates_before}")
    print(f"Cleaned dataset shape: {df.shape}")
    print("\nMissing values after cleaning:")
    print(df.isna().sum().to_string())

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nCleaned dataset saved to: {OUTPUT_FILE}")
    print("Cleaning completed successfully.")


if __name__ == "__main__":
    main()
