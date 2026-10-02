"""
classifiers.py
--------------
Trains and compares three ML algorithms from Module 5 (Machine Learning for
AI) on the TF-IDF email features:

  1. Multinomial Naive Bayes  (Probabilistic Learning & Naive Bayes)
  2. Linear SVM                (Supervised Learning - SVM)
  3. Decision Tree              (Supervised Learning - Decision Trees)

This "Compare ML algorithms" step is the strong academic component called
out in the project brief: it trains all three on an identical train/test
split and reports Accuracy, Precision/Recall/F1 and a confusion matrix for
each, so the report can justify which algorithm is actually deployed.
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import load_dataset, make_vectorizer
import paths

MODELS_DIR = paths.models_dir()


def get_model_zoo():
    return {
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=42, max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=30),
    }


def train_and_compare(csv_path: str, test_size: float = 0.25, random_state: int = 42):
    df = load_dataset(csv_path)
    vectorizer = make_vectorizer()
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = {}
    trained_models = {}
    for name, model in get_model_zoo().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))
        results[name] = {
            "accuracy": acc,
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "report": report,
            "confusion_matrix": cm.tolist(),
            "labels": sorted(y.unique()),
        }
        trained_models[name] = model

    # pick the best model by macro F1 (fairer than accuracy on balanced-ish
    # but multi-class data) to actually deploy in the pipeline
    best_name = max(results, key=lambda k: results[k]["macro_f1"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))
    for name, model in trained_models.items():
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(MODELS_DIR, fname))
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.joblib"))

    return results, best_name, trained_models, vectorizer


def print_comparison_report(results: dict, best_name: str):
    print("\n" + "=" * 72)
    print("ALGORITHM COMPARISON  (Module 5: Machine Learning for AI)")
    print("=" * 72)
    print(f"{'Algorithm':<16}{'Accuracy':<12}{'Macro F1':<12}{'Weighted F1':<12}")
    print("-" * 72)
    for name, r in results.items():
        marker = "  <-- selected" if name == best_name else ""
        print(f"{name:<16}{r['accuracy']*100:>7.2f}%   {r['macro_f1']*100:>7.2f}%    {r['weighted_f1']*100:>7.2f}%{marker}")
    print("=" * 72)
    print(f"Best performing algorithm on this dataset: {best_name}\n")


if __name__ == "__main__":
    csv_path = paths.emails_csv()
    results, best_name, models, vec = train_and_compare(csv_path)
    print_comparison_report(results, best_name)
