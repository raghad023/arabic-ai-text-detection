# Detection of AI-Generated Arabic Text: A Data Mining Approach

A data mining project for detecting AI-generated Arabic academic abstracts using
stylometric features, traditional machine learning methods, and BERT-based
representations.

## Dataset

The project uses the **KFUPM-JRCAI/arabic-generated-abstracts** dataset available
on Hugging Face.

The dataset contains three subsets based on the text generation method:

- `by_polishing`
- `from_title`
- `from_title_and_content`

Each record contains one original human-written Arabic abstract and four
AI-generated versions produced by:

- ALLAM
- JAIS
- LLaMA
- OpenAI

### Dataset Size

| Generation Method | Records |
|---|---:|
| by_polishing | 2,851 |
| from_title | 2,963 |
| from_title_and_content | 2,574 |
| **Total** | **8,388** |

## Phase 1: Project Setup and Data Acquisition

Phase 1 focuses on:

- Setting up the project repository and development environment.
- Loading the dataset directly from Hugging Face using the `datasets` library.
- Inspecting dataset structure, columns, and data types.
- Examining the Human vs. AI-generated target distribution.
- Assessing initial data quality, including missing values, duplicate values,
  duplicate rows, and basic inconsistencies.

### Initial Findings

- There are **8,388 raw records** in the dataset.
- Each raw record contains one human abstract and four AI-generated abstracts.
- The resulting raw text-instance distribution is:
  - **8,388 Human texts (20%)**
  - **33,552 AI-generated texts (80%)**
- No missing values were identified.
- No empty or whitespace-only texts were identified.
- No exact duplicate rows were identified.
- Three duplicate text occurrences were found in
  `jais_generated_abstract` within the `from_title_and_content` subset.
- Four JAIS outputs, in the subset had no alphanumeric characters. This shows that the generated text is malformed

## Project Structure

```text
arabic-ai-text-detection/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   └── data_preparation.py
├── docs/
├── reports/
│   ├── figures/
│   └── presentations/
├── models/
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
