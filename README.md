# Data Analysis – NLP Text Analysis

IU Portfolio Project (DLBDSEDA02_D): Applying NLP techniques to analyze a collection of complaint texts and extract the most frequently discussed topics.

## Project Overview

This project uses Natural Language Processing (NLP) to analyze unstructured text data from the CFPB Consumer Complaint Database. The goal is to extract the most common topics and provide actionable insights for decision-makers.

## Pipeline

1. **Data Loading** (`src/01_load_data.py`) – Load CSV, filter for narrative text, draw 50k sample
2. **Preprocessing** (`src/02_preprocessing.py`) – Lowercasing, special char/URL removal, tokenization, stopword removal, lemmatization
3. **Vectorization** (`src/03_vectorization.py`) – TF-IDF (5,000 features) and Word2Vec (100-dim)
4. **Topic Extraction** (`src/04_topic_modeling.py`) – LDA, NMF, K-Means on Word2Vec + coherence scores
5. **Visualization** (`src/05_visualization.py`) – Word clouds, top words charts, coherence comparison

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv/Scripts/activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

Download the CFPB dataset from https://www.consumerfinance.gov/data-research/consumer-complaints/ and place `complaints.csv` in the `data/` folder. Then run the scripts sequentially:

```bash
python src/01_load_data.py
python src/02_preprocessing.py
python src/03_vectorization.py
python src/04_topic_modeling.py
python src/05_visualization.py
```

Results are saved in the `results/` folder.

## Project Structure

```
DataAnalysis/
├── data/              # Raw and processed datasets
├── src/               # Python scripts (pipeline)
├── results/           # Output plots and CSV files
├── reports/           # LaTeX reports (Phase 1-3)
├── Quellen/           # Scientific sources (PDFs)
├── requirements.txt   # Python dependencies
└── README.md
```

## Tech Stack

- **Python 3.13**
- **NLP:** spaCy, NLTK
- **ML:** scikit-learn, gensim
- **Visualization:** matplotlib, wordcloud
