"""
Step 3: Vectorize preprocessed texts using TF-IDF and Word2Vec.
Saves both representations for downstream topic modeling.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "complaints_clean.csv"
TFIDF_PATH = PROJECT_ROOT / "data" / "tfidf_matrix.pkl"
W2V_PATH = PROJECT_ROOT / "data" / "word2vec.model"
VOCAB_PATH = PROJECT_ROOT / "data" / "tfidf_vectorizer.pkl"


def build_tfidf(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def build_word2vec(texts, vector_size=100, window=5, min_count=5, workers=4):
    tokenized = [text.split() for text in texts]
    model = Word2Vec(tokenized, vector_size=vector_size, window=window,
                     min_count=min_count, workers=workers, seed=42)
    return model


def document_vectors(model, texts):
    vectors = []
    for text in texts:
        words = [w for w in text.split() if w in model.wv]
        if words:
            vectors.append(np.mean(model.wv[words], axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH, dtype=str)
    texts = df["processed_text"].fillna("").tolist()

    # TF-IDF
    vectorizer, tfidf_matrix = build_tfidf(texts)
    print(f"TF-IDF: {tfidf_matrix.shape[0]} docs x {tfidf_matrix.shape[1]} features")

    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    # Word2Vec
    w2v_model = build_word2vec(texts)
    w2v_model.save(str(W2V_PATH))
    doc_vecs = document_vectors(w2v_model, texts)
    print(f"Word2Vec: {doc_vecs.shape[0]} docs x {doc_vecs.shape[1]} dims")
    print(f"Vocabulary size: {len(w2v_model.wv)}")

    np.save(PROJECT_ROOT / "data" / "w2v_doc_vectors.npy", doc_vecs)
