"""
priority_model.py
------------------
"Custom priority model" - a feature the brief calls out as a strong,
non-trivial addition beyond what Gmail already ships (Gmail's own
"Importance markers" are a black box; ours is a transparent, tunable hybrid
model combining symbolic rules (Module 4) with the ML classifier's own
confidence (Module 5)).

Priority is scored independently of the category label: a Promotions email
can still be "Low" while a Primary email about a deadline can be "Critical".

Score = category_weight + urgency_keyword_score + model_confidence_bonus
        (each component capped, summed, then clipped to [0, 100])
"""

import re
import numpy as np

CATEGORY_BASE_WEIGHT = {
    "Primary": 40,
    "Finance_Bills": 45,
    "Job_Academic": 45,
    "Social": 15,
    "Promotions": 5,
    "Spam": 0,
}

URGENCY_KEYWORDS = {
    "urgent": 15, "asap": 15, "immediately": 15, "deadline": 12, "due": 10,
    "today": 10, "tomorrow": 8, "final notice": 20, "overdue": 15,
    "action required": 12, "interview": 15, "expires": 10, "reminder": 6,
    "last chance": 8, "important": 10,
}

BAND_THRESHOLDS = [
    (75, "Critical"),
    (50, "High"),
    (25, "Medium"),
    (0, "Low"),
]


def _get_model_confidence(model, vectorizer, text: str, predicted_label: str) -> float:
    """Return a 0-1 confidence estimate for the predicted label, using
    predict_proba when available (Naive Bayes) and a normalised
    decision_function otherwise (LinearSVC has no predict_proba by default;
    DecisionTreeClassifier does support predict_proba)."""
    X = vectorizer.transform([text])
    classes = list(model.classes_)
    idx = classes.index(predicted_label) if predicted_label in classes else 0

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        return float(proba[idx])

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)[0]
        scores = np.atleast_1d(scores)
        # softmax-normalise the decision scores into a pseudo-probability
        exp_scores = np.exp(scores - np.max(scores))
        proba = exp_scores / exp_scores.sum()
        return float(proba[idx]) if len(proba) > idx else float(proba[0])

    return 0.5  # neutral fallback


def score_priority(sender: str, subject: str, body: str, predicted_label: str,
                    model=None, vectorizer=None, clean_text: str = ""):
    """
    Returns a dict: {"score": int 0-100, "band": str, "reasons": [str, ...]}
    """
    reasons = []
    text_lower = f"{subject} {body}".lower()

    base = CATEGORY_BASE_WEIGHT.get(predicted_label, 10)
    reasons.append(f"Base weight for category '{predicted_label}': +{base}")
    score = base

    urgency_score = 0
    for kw, weight in URGENCY_KEYWORDS.items():
        if kw in text_lower:
            urgency_score += weight
            reasons.append(f"Urgency keyword '{kw}' found: +{weight}")
    urgency_score = min(urgency_score, 35)  # cap contribution
    score += urgency_score

    if model is not None and vectorizer is not None:
        confidence = _get_model_confidence(model, vectorizer, clean_text or text_lower, predicted_label)
        conf_bonus = round(confidence * 15)  # up to +15 for a very confident, "clear-cut" mail
        score += conf_bonus
        reasons.append(f"Model confidence {confidence*100:.1f}% in this category: +{conf_bonus}")

    score = int(max(0, min(100, score)))
    band = next(label for threshold, label in BAND_THRESHOLDS if score >= threshold)

    return {"score": score, "band": band, "reasons": reasons}
