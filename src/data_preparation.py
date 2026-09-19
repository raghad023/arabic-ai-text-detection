"""Phase 1 data acquisition and initial data quality assessment.

This module loads the Arabic AI-generated abstracts dataset from
Hugging Face, validates its structure, summarizes the target
distribution, and performs initial data quality checks.
"""

from collections import Counter
from itertools import combinations

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

HUMAN_COLUMN = "original_abstract"

AI_COLUMNS = [
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract",
]

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
        columns_valid = (
            set(split.column_names) == set(EXPECTED_COLUMNS)
        )

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


def summarize_target_distribution(
    dataset: DatasetDict,
) -> pd.DataFrame:
    """Summarize the raw Human vs AI text-instance distribution."""

    human_count = sum(
        len(split[HUMAN_COLUMN])
        for split in dataset.values()
    )

    ai_count = sum(
        len(split[column])
        for split in dataset.values()
        for column in AI_COLUMNS
    )

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
    """Return True for non-empty text containing no letters or digits."""

    stripped_text = text.strip()

    return (
        bool(stripped_text)
        and not any(
            character.isalnum()
            for character in stripped_text
        )
    )


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
                "duplicate_rows": int(
                    df.duplicated().sum()
                ),
            }
        )

        for column in df.columns:
            texts = df[column]

            missing_count = int(
                texts.isna().sum()
            )

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


def analyze_original_overlap(
    dataset: DatasetDict,
) -> pd.DataFrame:
    """Analyze exact original-abstract overlap across dataset subsets."""

    subset_names = list(dataset.keys())

    original_sets = {
        split_name: set(
            dataset[split_name][HUMAN_COLUMN]
        )
        for split_name in subset_names
    }

    results = []

    for subset_a, subset_b in combinations(subset_names, 2):
        overlap = (
            original_sets[subset_a]
            & original_sets[subset_b]
        )

        results.append(
            {
                "metric": (
                    f"{subset_a} vs {subset_b}"
                ),
                "count": len(overlap),
            }
        )

    common_to_all = set.intersection(
        *original_sets.values()
    )

    all_originals = []

    for split_name in subset_names:
        all_originals.extend(
            dataset[split_name][HUMAN_COLUMN]
        )

    original_counts = Counter(all_originals)
    occurrence_distribution = Counter(
        original_counts.values()
    )

    results.append(
        {
            "metric": "Common to all subsets",
            "count": len(common_to_all),
        }
    )

    results.append(
        {
            "metric": "Total original instances",
            "count": len(all_originals),
        }
    )

    results.append(
        {
            "metric": "Unique original abstracts",
            "count": len(original_counts),
        }
    )

    for occurrences in sorted(occurrence_distribution):
        results.append(
            {
                "metric": (
                    f"Originals appearing in "
                    f"{occurrences} subset(s)"
                ),
                "count": occurrence_distribution[occurrences],
            }
        )

    return pd.DataFrame(results)


def analyze_human_ai_overlap(
    dataset: DatasetDict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify exact text overlap between Human and AI outputs."""

    human_texts = set()
    ai_texts = set()

    for split_name in dataset.keys():
        human_texts.update(
            dataset[split_name][HUMAN_COLUMN]
        )

        for column in AI_COLUMNS:
            ai_texts.update(
                dataset[split_name][column]
            )

    human_ai_overlap = human_texts & ai_texts

    summary = pd.DataFrame(
        {
            "metric": [
                "Unique Human texts",
                "Unique AI-generated texts",
                "Exact Human-AI overlaps",
            ],
            "count": [
                len(human_texts),
                len(ai_texts),
                len(human_ai_overlap),
            ],
        }
    )

    overlap_records = []

    for split_name, split in dataset.items():
        df = split.to_pandas()

        for column in AI_COLUMNS:
            matching_rows = df[
                df[column].isin(human_ai_overlap)
            ]

            for row_index, row in matching_rows.iterrows():
                overlap_records.append(
                    {
                        "split": split_name,
                        "row_index": row_index,
                        "generator": column,
                        "same_as_own_original": (
                            row[column]
                            == row[HUMAN_COLUMN]
                        ),
                    }
                )

    overlap_details = pd.DataFrame(overlap_records)

    return summary, overlap_details


def main() -> None:
    """Run the Phase 1 data understanding workflow."""

    print("Loading dataset...")
    dataset = load_raw_dataset()

    print("\nDataset loaded successfully.")
    print(dataset)

    validate_structure(dataset)

    print("\n=== Target Distribution ===")
    target_distribution = summarize_target_distribution(dataset)
    print(
        target_distribution.to_string(index=False)
    )

    print("\n=== Column-Level Data Quality ===")
    column_quality, duplicate_rows = assess_data_quality(dataset)
    print(
        column_quality.to_string(index=False)
    )

    print("\n=== Duplicate Rows ===")
    print(
        duplicate_rows.to_string(index=False)
    )

    print("\n=== Cross-Subset Original Overlap ===")
    original_overlap = analyze_original_overlap(dataset)
    print(
        original_overlap.to_string(index=False)
    )

    print("\n=== Human-AI Exact Text Overlap ===")
    overlap_summary, overlap_details = analyze_human_ai_overlap(
        dataset
    )

    print(
        overlap_summary.to_string(index=False)
    )

    if not overlap_details.empty:
        print("\nOverlap details:")
        print(
            overlap_details.to_string(index=False)
        )


if __name__ == "__main__":
    main()