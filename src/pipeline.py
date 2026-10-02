"""
pipeline.py
-----------
Ties the whole system together into one function, `classify_email()`:

  1. Rule-based first pass (Module 4)      -> fast, transparent, high-confidence cases
  2. ML classifier fallback (Module 5)      -> Naive Bayes / SVM / Decision Tree
  3. Explainability (Module 6)              -> top contributing words
  4. Custom priority scoring                -> urgency band, independent of category

This is the single entry point used by both the offline demo (demo.py) and
the live Gmail integration (main_live.py).
"""

import os
import joblib

from preprocessing import clean_text
from rule_filter import apply_rules
from explainability import explain_prediction, format_explanation
from priority_model import score_priority
import paths

ALL_CATEGORIES = ["Primary", "Social", "Promotions", "Spam", "Finance_Bills", "Job_Academic"]


class EmailClassifierPipeline:
    def __init__(self, model_name: str = None, progress_callback=None):
        import bootstrap
        bootstrap.ensure_ready(progress_callback=progress_callback)  # auto-generate dataset + train models on first run

        self.vectorizer = joblib.load(paths.vectorizer_path())
        best_name = model_name or joblib.load(paths.best_model_name_path())
        self.model = joblib.load(paths.model_path(best_name))
        self.model_name = best_name

    def classify_email(self, sender: str, subject: str, body: str):
        # 1. Rule-based pass
        rule_label, rule_conf, rule_reason = apply_rules(sender, subject, body)

        text_clean = clean_text((subject + " ") * 2 + body)

        if rule_label is not None:
            predicted_label = rule_label
            source = "rule-based"
            confidence_note = f"{rule_conf*100:.0f}% (rule-based, deterministic)"
            explanation = [{"word": "(rule engine)", "contribution": 1.0}]
            explanation_text = f"Predicted '{predicted_label}' by rule: {rule_reason}."
        else:
            # 2. ML fallback
            predicted_label = self.model.predict(self.vectorizer.transform([text_clean]))[0]
            source = f"ml:{self.model_name}"
            explanation = explain_prediction(self.model, self.vectorizer, text_clean, predicted_label)
            explanation_text = format_explanation(predicted_label, explanation)
            confidence_note = f"predicted by {self.model_name}"

        # 3. Priority scoring (independent of category source)
        priority = score_priority(
            sender, subject, body, predicted_label,
            model=self.model, vectorizer=self.vectorizer, clean_text=text_clean
        )

        return {
            "sender": sender,
            "subject": subject,
            "predicted_category": predicted_label,
            "decision_source": source,
            "confidence_note": confidence_note,
            "explanation": explanation,
            "explanation_text": explanation_text,
            "priority_score": priority["score"],
            "priority_band": priority["band"],
            "priority_reasons": priority["reasons"],
        }
