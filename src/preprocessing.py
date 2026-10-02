"""
preprocessing.py
----------------
Text cleaning and feature extraction (TF-IDF) for the email classifier.
Corresponds to Module 5 (feature engineering for ML) and, indirectly,
Module 4 (the cleaned tokens are also reused by the rule-based layer).
"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

STOPWORDS = set("""
a an the is are was were be been being to of in on for and or with at by
from as this that it its your you i we our us they he she his her them
will would can could should may might do does did not no nor so if then
than up down out over under again further once here there when where why
how all any both each few more most other some such only own same
""".split())


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/emails/punctuation/digits noise, remove stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                     # emails
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def build_corpus(df: pd.DataFrame) -> pd.Series:
    """Combine subject + body (subject weighted x2 since it's more predictive) and clean."""
    combined = (df["subject"].fillna("") + " ") * 2 + df["body"].fillna("")
    return combined.apply(clean_text)


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    df["clean_text"] = build_corpus(df)
    return df


def make_vectorizer(max_features: int = 3000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)
