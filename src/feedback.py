"""
feedback.py
-----------
"Personalized learning from corrections" - when the AI gets a category
wrong for a particular user, that user can correct it once, and the model
incorporates the correction on the next retrain. This is what makes the
system "personal" rather than a fixed, one-size-fits-all classifier like
Gmail's built-in tabs.

Design (kept intentionally simple for a capstone prototype):
  1. Every correction is appended to data/corrections.csv (sender, subject,
     body, correct_label, source="user_correction").
  2. retrain_with_feedback() merges corrections.csv into the base training
     set (data/emails.csv) and re-runs the full train_and_compare pipeline,
     so corrected examples directly influence the next model.
  3. Corrections are additive and never deleted, so retraining is always
     reproducible from data/emails.csv + data/corrections.csv.

In a production system you would weight recent corrections more heavily,
deduplicate near-identical corrections, and retrain incrementally (e.g.
MultinomialNB.partial_fit) instead of a full batch retrain - noted as a
"Future Work" item in README.md.
"""

import os
import csv
import pandas as pd
import paths

EMAILS_CSV = paths.emails_csv()
CORRECTIONS_CSV = paths.corrections_csv()
MERGED_CSV = paths.merged_csv()

VALID_LABELS = ["Primary", "Social", "Promotions", "Spam", "Finance_Bills", "Job_Academic"]


def record_correction(sender: str, subject: str, body: str, correct_label: str):
    if correct_label not in VALID_LABELS:
        raise ValueError(f"'{correct_label}' is not a valid category. Choose from {VALID_LABELS}")

    file_exists = os.path.exists(CORRECTIONS_CSV)
    with open(CORRECTIONS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["sender", "subject", "body", "label"])
        writer.writerow([sender, subject, body, correct_label])
    print(f"Correction recorded: '{subject[:50]}' -> {correct_label}")


def retrain_with_feedback():
    """Merge base dataset + all recorded corrections, retrain, and save the
    new models (overwriting the ones in models/). Returns the same
    (results, best_name) tuple as classifiers.train_and_compare."""
    from classifiers import train_and_compare, print_comparison_report

    base_df = pd.read_csv(EMAILS_CSV)
    if os.path.exists(CORRECTIONS_CSV):
        corr_df = pd.read_csv(CORRECTIONS_CSV)
        n_corrections = len(corr_df)
        merged = pd.concat([base_df, corr_df], ignore_index=True)
    else:
        n_corrections = 0
        merged = base_df

    merged.to_csv(MERGED_CSV, index=False)
    print(f"Retraining on {len(merged)} emails ({n_corrections} from user corrections)...")

    results, best_name, models, vectorizer = train_and_compare(MERGED_CSV)
    print_comparison_report(results, best_name)
    return results, best_name


if __name__ == "__main__":
    # Small demonstration: the base rule-engine + ML model has no way of
    # knowing that *this particular user* treats a specific newsletter as
    # important Primary mail rather than a Promotion. We simulate a user
    # correction and show the model adapting after retraining.
    from pipeline import EmailClassifierPipeline

    sender = "updates@myfavoritenewsletter.com"
    subject = "This week's AI research digest"
    body = "Here are the top 5 AI papers this week, curated just for subscribers like you."

    pipeline = EmailClassifierPipeline()
    before = pipeline.classify_email(sender, subject, body)
    print(f"BEFORE correction -> predicted: {before['predicted_category']} ({before['decision_source']})")

    # User says: "actually this is Primary to me, I read it every week"
    record_correction(sender, subject, body, "Primary")
    # add a couple more similar examples so the ML model has enough signal
    record_correction("updates@myfavoritenewsletter.com", "Last week's AI research digest",
                       "Here are the top 5 AI papers from last week, curated for subscribers.", "Primary")
    record_correction("updates@myfavoritenewsletter.com", "AI research digest - new issue",
                       "This week's curated list of must-read AI papers is ready for you.", "Primary")

    retrain_with_feedback()

    pipeline_after = EmailClassifierPipeline()
    after = pipeline_after.classify_email(sender, subject, body)
    print(f"AFTER correction  -> predicted: {after['predicted_category']} ({after['decision_source']})")
