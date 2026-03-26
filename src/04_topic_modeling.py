"""
Step 4: Extract topics using LDA (gensim) and NMF (scikit-learn).
Prints top words per topic for comparison.
"""

import pickle
import pandas as pd
from pathlib import Path
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
from sklearn.decomposition import NMF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "complaints_clean.csv"
TFIDF_PATH = PROJECT_ROOT / "data" / "tfidf_matrix.pkl"
VOCAB_PATH = PROJECT_ROOT / "data" / "tfidf_vectorizer.pkl"
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
    return model, corpus, dictionary


def run_nmf(tfidf_matrix, num_topics):
    model = NMF(n_components=num_topics, random_state=RANDOM_SEED, max_iter=300)
    W = model.fit_transform(tfidf_matrix)
    return model, W


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
    for idx, component in enumerate(model.components_):
        top_indices = component.argsort()[-num_words:][::-1]
        words = [feature_names[i] for i in top_indices]
        topics.append(", ".join(words))
    return topics


def save_topics(lda_topics, nmf_topics, path):
    rows = []
    for i, (lda_t, nmf_t) in enumerate(zip(lda_topics, nmf_topics)):
        rows.append({"topic": i + 1, "lda": lda_t, "nmf": nmf_t})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH, dtype=str)
    texts = df["processed_text"].fillna("").tolist()

    with open(TFIDF_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(VOCAB_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    # LDA
    lda_model, corpus, dictionary = run_lda(texts, NUM_TOPICS)
    lda_topics = get_lda_topics(lda_model)
    print_topics("LDA", lda_topics)

    # NMF
    nmf_model, W = run_nmf(tfidf_matrix, NUM_TOPICS)
    feature_names = vectorizer.get_feature_names_out()
    nmf_topics = get_nmf_topics(nmf_model, feature_names)
    print_topics("NMF", nmf_topics)

    # Save
    RESULTS_PATH.mkdir(exist_ok=True)
    save_topics(lda_topics, nmf_topics, RESULTS_PATH / "topics.csv")
    lda_model.save(str(PROJECT_ROOT / "data" / "lda_model"))
