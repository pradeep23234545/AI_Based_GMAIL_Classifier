"""
explainability.py
------------------
"Explainable classification" (Module 6: AI Ethics and Explainable AI).

Gmail tells you an email is Promotions/Spam/etc. but never *why*. This
module opens that black box: for any prediction, it reports the exact words
in THIS email that pushed the model toward its decision, working across all
three algorithm types we compare:

  - Naive Bayes    -> uses each word's log-probability under the predicted
                        class vs. its average log-probability under all
                        other classes (a log-likelihood-ratio contribution).
  - Linear SVM      -> uses the learned per-class weight (coef_) multiplied
                        by the word's TF-IDF value in this email.
  - Decision Tree   -> uses global feature_importances_, restricted to the
                        words actually present in this email.

The output is used both to build user trust and to satisfy CO6 (AI Ethics &
Explainable AI) academically.
"""

import numpy as np


def _words_present(vectorizer, text: str):
    """Return (word, tfidf_value, feature_index) for every vocabulary term
    present in this email's cleaned text, sorted by tfidf weight."""
    vec = vectorizer.transform([text])
    row = vec.tocoo()
    feature_names = np.array(vectorizer.get_feature_names_out())
    pairs = [(feature_names[j], v, j) for j, v in zip(row.col, row.data)]
    pairs.sort(key=lambda p: -p[1])
    return pairs


def explain_prediction(model, vectorizer, text: str, predicted_label: str, top_n: int = 6):
    """
    Returns a list of dicts: [{"word": str, "contribution": float}, ...]
    sorted by how strongly each word supported the predicted label.
    Falls back gracefully if the model type isn't recognised.
    """
    present = _words_present(vectorizer, text)
    if not present:
        return []

    classes = list(model.classes_)
    if predicted_label not in classes:
        return []
    class_idx = classes.index(predicted_label)

    contributions = []

    if hasattr(model, "feature_log_prob_"):
        # Naive Bayes: log P(word | predicted class) minus the average
        # log P(word | other classes) = how much more "at home" this word
        # is in the predicted class than elsewhere.
        log_probs = model.feature_log_prob_  # shape (n_classes, n_features)
        other_idx = [i for i in range(len(classes)) if i != class_idx]
        for word, tfidf_val, j in present:
            this_class_lp = log_probs[class_idx, j]
            other_avg_lp = np.mean(log_probs[other_idx, j]) if other_idx else 0.0
            score = (this_class_lp - other_avg_lp) * tfidf_val
            contributions.append((word, score))

    elif hasattr(model, "coef_"):
        # Linear SVM: sign/magnitude of the learned weight for this class,
        # scaled by how strongly the word appears in this email.
        coef = model.coef_
        # LinearSVC uses one-vs-rest; for binary case coef_ has 1 row
        row_idx = class_idx if coef.shape[0] > 1 else 0
        for word, tfidf_val, j in present:
            weight = coef[row_idx, j]
            contributions.append((word, weight * tfidf_val))

    elif hasattr(model, "feature_importances_"):
        # Decision Tree: global importance, restricted to words in this email.
        importances = model.feature_importances_
        for word, tfidf_val, j in present:
            contributions.append((word, importances[j] * tfidf_val))

    else:
        return []

    contributions.sort(key=lambda x: -x[1])
    top = [c for c in contributions if c[1] > 0][:top_n]
    return [{"word": w, "contribution": round(float(s), 4)} for w, s in top]


def format_explanation(predicted_label: str, explanation: list) -> str:
    if not explanation:
        return f"Predicted '{predicted_label}' (no strong individual word signals found)."
    words = ", ".join(f"'{e['word']}'" for e in explanation)
    return f"Predicted '{predicted_label}' mainly because of these words: {words}."
