"""
Step 6 (extension): Coherence-based selection of the number of topics k.

Runs LDA, NMF and K-Means on Word2Vec for k in {5,7,10,12,15,20},
computes c_v coherence and saves both a CSV and a comparison plot.

This script provides the empirical justification for the choice
NUM_TOPICS = 10 used in 04_topic_modeling.py.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from gensim.models import LdaMulticore, CoherenceModel
from gensim.corpora import Dictionary
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "complaints_clean.csv"
TFIDF_PATH = PROJECT_ROOT / "data" / "tfidf_matrix.pkl"
VOCAB_PATH = PROJECT_ROOT / "data" / "tfidf_vectorizer.pkl"
W2V_VECS_PATH = PROJECT_ROOT / "data" / "w2v_doc_vectors.npy"
RESULTS_PATH = PROJECT_ROOT / "results"

K_RANGE = [5, 7, 10, 12, 15, 20]
RANDOM_SEED = 42
NUM_WORDS = 10


def lda_topics_as_words(model, num_words=NUM_WORDS):
    return [[w for w, _ in model.show_topic(i, topn=num_words)]
            for i in range(model.num_topics)]


def nmf_topics_as_words(model, feature_names, num_words=NUM_WORDS):
    out = []
    for comp in model.components_:
        idx = comp.argsort()[-num_words:][::-1]
        out.append([feature_names[i] for i in idx])
    return out


def kmeans_topics_as_words(labels, texts, num_words=NUM_WORDS):
    out = []
    for cid in range(int(labels.max()) + 1):
        cluster = [texts[i] for i, l in enumerate(labels) if l == cid]
        words = " ".join(cluster).split()
        top = [w for w, _ in Counter(words).most_common(num_words)]
        out.append(top)
    return out


def coherence(topics_words, tokenized, dictionary):
    cm = CoherenceModel(topics=topics_words, texts=tokenized,
                        dictionary=dictionary, coherence="c_v")
    return cm.get_coherence()


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH, dtype=str)
    texts = df["processed_text"].fillna("").tolist()
    tokenized = [t.split() for t in texts]

    with open(TFIDF_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(VOCAB_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    feature_names = vectorizer.get_feature_names_out()

    w2v_vectors = np.load(W2V_VECS_PATH)

    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    rows = []
    for k in K_RANGE:
        print(f"\n=== k = {k} ===")

        lda = LdaMulticore(corpus, num_topics=k, id2word=dictionary,
                           passes=3, random_state=RANDOM_SEED, workers=3)
        c_lda = coherence(lda_topics_as_words(lda), tokenized, dictionary)
        print(f"  LDA     c_v = {c_lda:.4f}")

        nmf = NMF(n_components=k, random_state=RANDOM_SEED, max_iter=300)
        nmf.fit(tfidf_matrix)
        c_nmf = coherence(nmf_topics_as_words(nmf, feature_names),
                          tokenized, dictionary)
        print(f"  NMF     c_v = {c_nmf:.4f}")

        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(w2v_vectors)
        c_km = coherence(kmeans_topics_as_words(labels, texts),
                         tokenized, dictionary)
        print(f"  K-Means c_v = {c_km:.4f}")

        rows.append({"k": k, "LDA": round(c_lda, 4),
                     "NMF": round(c_nmf, 4),
                     "KMeans_W2V": round(c_km, 4)})

    sweep_df = pd.DataFrame(rows)
    RESULTS_PATH.mkdir(exist_ok=True)
    sweep_df.to_csv(RESULTS_PATH / "coherence_sweep.csv", index=False)
    print("\nResults:\n", sweep_df)

    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["k"], sweep_df["LDA"], "o-", label="LDA (TF-IDF)")
    plt.plot(sweep_df["k"], sweep_df["NMF"], "s-", label="NMF (TF-IDF)")
    plt.plot(sweep_df["k"], sweep_df["KMeans_W2V"], "^--",
             label="K-Means (Word2Vec)")
    plt.xlabel("Anzahl Topics k")
    plt.ylabel(r"Coherence Score $c_v$")
    plt.title("Coherence Score in Abhängigkeit der Topic-Anzahl")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "coherence_by_k.png", dpi=150)
    print(f"\nSaved: {RESULTS_PATH / 'coherence_by_k.png'}")
