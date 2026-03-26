"""
Step 5: Visualize topic modeling results – word clouds per topic,
top words bar charts, and product distribution.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
from sklearn.decomposition import NMF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "complaints_clean.csv"
TFIDF_PATH = PROJECT_ROOT / "data" / "tfidf_matrix.pkl"
VOCAB_PATH = PROJECT_ROOT / "data" / "tfidf_vectorizer.pkl"
RESULTS_PATH = PROJECT_ROOT / "results"

NUM_TOPICS = 10


def plot_top_words(model, feature_names, n_words, title, filename):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
    axes = axes.flatten()

    for i, (component, ax) in enumerate(zip(model.components_, axes)):
        top_idx = component.argsort()[-n_words:][::-1]
        words = [feature_names[j] for j in top_idx]
        weights = [component[j] for j in top_idx]
        ax.barh(words[::-1], weights[::-1])
        ax.set_title(f"Topic {i+1}", fontsize=11)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_wordclouds(model, feature_names, title, filename):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, (component, ax) in enumerate(zip(model.components_, axes)):
        top_idx = component.argsort()[-30:][::-1]
        word_weights = {feature_names[j]: component[j] for j in top_idx}
        wc = WordCloud(width=400, height=300, background_color="white",
                       colormap="viridis")
        wc.generate_from_frequencies(word_weights)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Topic {i+1}", fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_product_distribution(df, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df["Product"].value_counts().head(10)
    counts.plot(kind="barh", ax=ax)
    ax.set_xlabel("Number of complaints")
    ax.set_title("Top 10 Product Categories in Sample")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_topic_distribution(W, filename):
    topic_assignments = np.argmax(W, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = pd.Series(topic_assignments).value_counts().sort_index()
    counts.index = [f"Topic {i+1}" for i in counts.index]
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of documents")
    ax.set_title("NMF Topic Distribution across Documents")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / filename, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    RESULTS_PATH.mkdir(exist_ok=True)

    df = pd.read_csv(CLEAN_PATH, dtype=str)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(VOCAB_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    feature_names = vectorizer.get_feature_names_out()

    # Refit NMF for visualization
    nmf_model = NMF(n_components=NUM_TOPICS, random_state=42, max_iter=300)
    W = nmf_model.fit_transform(tfidf_matrix)

    plot_top_words(nmf_model, feature_names, 10,
                   "NMF – Top 10 Words per Topic", "nmf_top_words.png")

    plot_wordclouds(nmf_model, feature_names,
                    "NMF – Word Clouds per Topic", "nmf_wordclouds.png")

    plot_product_distribution(df, "product_distribution.png")

    plot_topic_distribution(W, "nmf_topic_distribution.png")

    print(f"Saved plots to {RESULTS_PATH}")
