"""
bootstrap.py
------------
First-run setup: makes sure a training dataset and trained models exist in
the user's data directory (see paths.py). This is what lets the GUI (and
the packaged .exe) "just work" the very first time someone opens it, with
no manual command-line steps beforehand.

Called automatically by pipeline.EmailClassifierPipeline() and by
gui_app.py at startup.
"""

import os
import paths


def is_ready() -> bool:
    return (
        os.path.exists(paths.emails_csv())
        and os.path.exists(paths.vectorizer_path())
        and os.path.exists(paths.best_model_name_path())
    )


def ensure_ready(progress_callback=None):
    """
    Generates data/emails.csv (if missing) and trains+saves all three
    models (if missing), writing everything into the per-user app-data
    folder. `progress_callback(message)` is called with human-readable
    status strings, e.g. for a GUI to display "Training models...".
    """
    def report(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    if not os.path.exists(paths.emails_csv()):
        report("First run: generating training dataset...")
        import generate_dataset
        generate_dataset.main()

    if not (os.path.exists(paths.vectorizer_path()) and os.path.exists(paths.best_model_name_path())):
        report("First run: training AI models (Naive Bayes / SVM / Decision Tree)...")
        import classifiers
        results, best_name, models, vec = classifiers.train_and_compare(paths.emails_csv())
        classifiers.print_comparison_report(results, best_name)
        report(f"Model ready: {best_name} selected as the deployed classifier.")


if __name__ == "__main__":
    ensure_ready()
