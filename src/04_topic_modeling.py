"""
Step 4: Extract topics using LDA (gensim), NMF (scikit-learn),
and K-Means on Word2Vec document vectors. Computes coherence
scores for quantitative comparison.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import LdaMulticore, CoherenceModel
from gensim.corpora import Dictionary
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "complaints_clean.csv"
TFIDF_PATH = PROJECT_ROOT / "data" / "tfidf_matrix.pkl"
VOCAB_PATH = PROJECT_ROOT / "data" / "tfidf_vectorizer.pkl"
W2V_VECS_PATH = PROJECT_ROOT / "data" / "w2v_doc_vectors.npy"
RESULTS_PATH = PROJECT_ROOT / "results"

NUM_TOPICS = 10
RANDOM_SEED = 42


def run_lda(texts, num_topics, passes=10):
    tokenized = [text.split() for text in texts]
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    model = LdaMulticore(
        corpus, num_topics=num_topics, id2word=dictionary,
        passes=passes, random_state=RANDOM_SEED, workers=3
    )
    return model, corpus, dictionary, tokenized


def run_nmf(tfidf_matrix, num_topics):
    model = NMF(n_components=num_topics, random_state=RANDOM_SEED, max_iter=300)
    W = model.fit_transform(tfidf_matrix)
    return model, W


def run_kmeans_w2v(doc_vectors, num_clusters):
    km = KMeans(n_clusters=num_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(doc_vectors)
    return km, labels


def get_kmeans_topics(labels, texts, num_words=10):
    topics = []
    for cluster_id in range(max(labels) + 1):
        cluster_texts = [texts[i] for i, l in enumerate(labels) if l == cluster_id]
        all_words = " ".join(cluster_texts).split()
        word_counts = Counter(all_words).most_common(num_words)
        topics.append(", ".join([w for w, _ in word_counts]))
    return topics


def compute_coherence(model, tokenized, dictionary, corpus):
    cm = CoherenceModel(model=model, texts=tokenized,
                        dictionary=dictionary, coherence="c_v")
    return cm.get_coherence()


def compute_nmf_coherence(nmf_model, feature_names, tokenized, dictionary, num_words=10):
    topics_as_words = []
    for component in nmf_model.components_:
        top_idx = component.argsort()[-num_words:][::-1]
        topics_as_words.append([feature_names[i] for i in top_idx])

    cm = CoherenceModel(topics=topics_as_words, texts=tokenized,
                        dictionary=dictionary, coherence="c_v")
    return cm.get_coherence()


def print_topics(label, topics):
    print(f"\n{'=' * 50}")
    print(f"{label} – Top 10 words per topic")
    print("=" * 50)
    for i, words in enumerate(topics):
        print(f"Topic {i+1}: {words}")


def get_lda_topics(model, num_words=10):
    topics = []
    for idx in range(model.num_topics):
        words = [w for w, _ in model.show_topic(idx, topn=num_words)]
        topics.append(", ".join(words))
    return topics


def get_nmf_topics(model, feature_names, num_words=10):
    topics = []
    for component in model.components_:
        top_indices = component.argsort()[-num_words:][::-1]
        words = [feature_names[i] for i in top_indices]
        topics.append(", ".join(words))
    return topics


def save_topics(lda_topics, nmf_topics, w2v_topics, path):
    rows = []
    for i in range(len(lda_topics)):
        rows.append({
            "topic": i + 1,
            "lda": lda_topics[i],
            "nmf": nmf_topics[i],
            "w2v_kmeans": w2v_topics[i],
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH, dtype=str)
    texts = df["processed_text"].fillna("").tolist()

    with open(TFIDF_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(VOCAB_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    w2v_vectors = np.load(W2V_VECS_PATH)

    # LDA
    lda_model, corpus, dictionary, tokenized = run_lda(texts, NUM_TOPICS)
    lda_topics = get_lda_topics(lda_model)
    print_topics("LDA", lda_topics)

    # NMF
    nmf_model, W = run_nmf(tfidf_matrix, NUM_TOPICS)
    feature_names = vectorizer.get_feature_names_out()
    nmf_topics = get_nmf_topics(nmf_model, feature_names)
    print_topics("NMF", nmf_topics)

    # K-Means on Word2Vec
    km_model, km_labels = run_kmeans_w2v(w2v_vectors, NUM_TOPICS)
    w2v_topics = get_kmeans_topics(km_labels, texts)
    print_topics("K-Means (Word2Vec)", w2v_topics)

    # Coherence scores
    print(f"\n{'=' * 50}")
    print("Coherence Scores (c_v)")
    print("=" * 50)
    lda_coherence = compute_coherence(lda_model, tokenized, dictionary, corpus)
    print(f"LDA:              {lda_coherence:.4f}")

    nmf_coherence = compute_nmf_coherence(nmf_model, feature_names, tokenized, dictionary)
    print(f"NMF:              {nmf_coherence:.4f}")

    # Save
    RESULTS_PATH.mkdir(exist_ok=True)
    save_topics(lda_topics, nmf_topics, w2v_topics, RESULTS_PATH / "topics.csv")
    lda_model.save(str(PROJECT_ROOT / "data" / "lda_model"))

    coherence_df = pd.DataFrame([
        {"method": "LDA", "coherence_cv": round(lda_coherence, 4)},
        {"method": "NMF", "coherence_cv": round(nmf_coherence, 4)},
    ])
    coherence_df.to_csv(RESULTS_PATH / "coherence_scores.csv", index=False)
