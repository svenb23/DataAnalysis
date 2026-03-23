# Data Analysis – NLP Text Analysis

IU Portfolio Project: Applying NLP techniques to analyze a collection of complaint texts and extract the most frequently discussed topics.

## Project Overview

This project uses Natural Language Processing (NLP) to analyze unstructured text data containing complaints. The goal is to extract the most common topics and provide actionable insights for decision-makers.

### Pipeline
1. **Data Loading** – Load and explore the dataset
2. **Preprocessing** – Clean texts (tokenization, stopword removal, lemmatization)
3. **Vectorization** – Convert texts to numerical vectors (TF-IDF, Word2Vec)
4. **Topic Extraction** – Extract topics using LDA and NMF
5. **Visualization** – Present results with word clouds and charts

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Project Structure

```
DataAnalysis/
├── data/              # Datasets
├── notebooks/         # Jupyter Notebooks
├── src/               # Python modules
├── results/           # Output plots and results
├── requirements.txt   # Python dependencies
└── README.md
```

## Tech Stack

- **Python 3.13**
- **NLP:** spaCy, NLTK
- **ML:** scikit-learn, gensim
- **Visualization:** matplotlib, seaborn, wordcloud, plotly
