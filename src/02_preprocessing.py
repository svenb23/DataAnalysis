"""
Step 2: Preprocess complaint narratives – lowercasing, removing
special characters/URLs/numbers, tokenization, stopword removal,
and lemmatization using spaCy.
"""

import re
import pandas as pd
import spacy
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = PROJECT_ROOT / "data" / "complaints_sample.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "complaints_clean.csv"

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"xx+", "", text)  # CFPB redacts info with XX/XXX
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize(texts, batch_size=1000):
    results = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total=len(texts), desc="Lemmatizing"):
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and len(token.lemma_) > 2
        ]
        results.append(" ".join(tokens))
    return results


if __name__ == "__main__":
    df = pd.read_csv(SAMPLE_PATH, dtype=str)

    df["clean_text"] = df["Consumer complaint narrative"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    df["processed_text"] = lemmatize(df["clean_text"].tolist())
    df = df[df["processed_text"].str.strip() != ""].reset_index(drop=True)

    print(f"Processed: {len(df)} entries")
    print(f"Example:\n{df['processed_text'].iloc[0][:200]}")

    df.to_csv(CLEAN_PATH, index=False)
