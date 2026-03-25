"""
Step 1: Load CFPB Consumer Complaint data, filter for entries
with narrative text, and draw a random sample of 50,000.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "complaints.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "complaints_sample.csv"

SAMPLE_SIZE = 50_000
RANDOM_SEED = 42

COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Complaint ID",
]


def load_and_filter(path, columns):
    df = pd.read_csv(path, usecols=columns, dtype=str)
    df = df.dropna(subset=["Consumer complaint narrative"])
    df = df[df["Consumer complaint narrative"].str.strip() != ""]
    return df.reset_index(drop=True)


def sample_data(df, n, seed):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_filter(RAW_DATA_PATH, COLUMNS)
    df_sample = sample_data(df, SAMPLE_SIZE, RANDOM_SEED)

    print(f"Filtered: {len(df)} entries with narrative")
    print(f"Sample: {len(df_sample)} entries")
    print(f"Products:\n{df_sample['Product'].value_counts().head(5)}")

    df_sample.to_csv(SAMPLE_PATH, index=False)
