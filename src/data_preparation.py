"""Phase 1 data acquisition and initial data understanding.

This module loads the Arabic AI-generated abstracts dataset from
Hugging Face and validates its basic structure for Phase 1 of the
project.
"""

import pandas as pd
from datasets import DatasetDict, load_dataset


DATASET_NAME = "KFUPM-JRCAI/arabic-generated-abstracts"

EXPECTED_COLUMNS = [
    "original_abstract",
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract",
]

AI_COLUMNS = [
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract",
    
]
HUMAN_COLUMN = "original_abstract"

EXPECTED_ROW_COUNTS = {
    "by_polishing": 2851,
    "from_title": 2963,
    "from_title_and_content": 2574,
}


def load_raw_dataset() -> DatasetDict:
    """Load the raw dataset from Hugging Face.

    Returns:
        DatasetDict: Dataset organized by generation method.
    """
    return load_dataset(DATASET_NAME)


def validate_structure(dataset: DatasetDict) -> None:
    """Validate dataset columns, data types, and row counts."""

    print("\n=== Dataset Structure Validation ===")

    for split_name, split in dataset.items():
        columns_valid = split.column_names == EXPECTED_COLUMNS

        data_types_valid = all(
            str(split.features[column]) == "Value('string')"
            for column in EXPECTED_COLUMNS
        )

        expected_rows = EXPECTED_ROW_COUNTS[split_name]
        row_count_valid = len(split) == expected_rows

        print(f"\n{split_name}")
        print(f"Columns valid: {columns_valid}")
        print(f"Data types valid: {data_types_valid}")
        print(
            f"Row count valid: {row_count_valid} "
            f"({len(split)} rows)"
        )
def summarize_target_distribution(dataset: DatasetDict) -> pd.DataFrame:
    """Summarize the raw Human vs AI text-instance distribution."""

    total_records = sum(len(split) for split in dataset.values())

    human_count = total_records
    ai_count = total_records * len(AI_COLUMNS)
    total_texts = human_count + ai_count

    distribution = pd.DataFrame(
        {
            "class": ["Human", "AI-generated"],
            "count": [human_count, ai_count],
        }
    )

    distribution["percentage"] = (
        distribution["count"] / total_texts * 100
    )

    return distribution
def has_no_alphanumeric(text: str) -> bool:
    """Return True when a text contains no letters or digits."""

    return not any(character.isalnum() for character in text)
def assess_data_quality(
    dataset: DatasetDict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assess missing, empty, duplicate, and malformed text values."""

    column_results = []
    row_results = []

    for split_name, split in dataset.items():
        df = split.to_pandas()

        row_results.append(
            {
                "split": split_name,
                "duplicate_rows": int(df.duplicated().sum()),
            }
        )

        for column in df.columns:
            texts = df[column]

            missing_count = int(texts.isna().sum())

            empty_count = int(
                texts.fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )

            duplicate_value_count = int(
                texts.duplicated().sum()
            )

            no_alphanumeric_count = int(
                texts.fillna("")
                .astype(str)
                .map(has_no_alphanumeric)
                .sum()
            )

            column_results.append(
                {
                    "split": split_name,
                    "column": column,
                    "missing": missing_count,
                    "empty_or_whitespace": empty_count,
                    "duplicate_values": duplicate_value_count,
                    "no_alphanumeric": no_alphanumeric_count,
                }
            )

    column_quality = pd.DataFrame(column_results)
    duplicate_rows = pd.DataFrame(row_results)

    return column_quality, duplicate_rows


def main() -> None:
    """Run the Phase 1 data understanding workflow."""

    print("Loading dataset...")
    dataset = load_raw_dataset()

    print("\nDataset loaded successfully.")
    print(dataset)

    validate_structure(dataset)

    print("\n=== Target Distribution ===")
    target_distribution = summarize_target_distribution(dataset)
    print(target_distribution.to_string(index=False))

    print("\n=== Column-Level Data Quality ===")
    column_quality, duplicate_rows = assess_data_quality(dataset)
    print(column_quality.to_string(index=False))

    print("\n=== Duplicate Rows ===")
    print(duplicate_rows.to_string(index=False))


if __name__ == "__main__":
    main()