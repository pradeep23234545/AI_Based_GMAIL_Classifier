"""
gui_app.py
----------
Desktop GUI for the AI Gmail Classifier capstone project, built with
Tkinter (bundled with Python - no extra GUI toolkit needed, and the easiest
to package into a standalone .exe with PyInstaller).

Tabs:
  1. Classify Email  - type/paste an email, see category + priority +
                        plain-English explanation, and correct it if wrong
                        (personalized learning).
  2. Model Training   - regenerate the dataset and re-run the Naive Bayes /
                        SVM / Decision Tree comparison, with results shown
                        in a table.
  3. Gmail Inbox      - connect to a real Gmail account, fetch recent mail,
                        see it classified in a table, and (opt-in) apply
                        the labels for real.
  4. About            - project/report summary for quick reference during
                        a viva or demo.

Run directly with:  python gui_app.py
Package into a Windows .exe with: build_exe.bat  (see README.md)
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import paths
import bootstrap
import feedback
import classifiers
from pipeline import EmailClassifierPipeline

APP_TITLE = "AI Gmail Classifier"

EXAMPLE_EMAILS = [
    dict(sender="notifications@linkedin.com", subject="You appeared in 9 searches this week",
         body="See who's been looking at your profile and grow your network."),
    dict(sender="offers@myntra.com", subject="Flat 60% OFF Ends Tonight",
         body="Hurry! Use code SAVE60 to get flat 60% off on all fashion brands, sale ends tonight."),
    dict(sender="alerts@hdfcbank.com", subject="Debit Alert",
         body="Rs.4599 has been debited from your account ending 4321 towards EMI payment."),
    dict(sender="careers@google.com", subject="Interview Invitation",
         body="You have been shortlisted for the AI Intern role. Please join the interview tomorrow at 10 AM."),
    dict(sender="priya.sharma@gmail.com", subject="Dinner this weekend?",
         body="Hey! Are you free for dinner this Saturday? Let me know, would love to catch up."),
]

ABOUT_TEXT = """AI-Based Gmail Email Filtering & Prioritization System
========================================================

An AI Capstone Project built entirely in Python.

WHAT IT DOES
------------
Classifies emails into 6 categories - the 4 Gmail already has (Primary,
Social, Promotions, Spam) plus two new ones this project adds:
  - Finance_Bills : bank alerts, EMIs, invoices, subscription renewals
  - Job_Academic  : interview calls, offer letters, exam/semester notices

It also scores every email's PRIORITY (Low/Medium/High/Critical)
independently of its category, EXPLAINS every decision in plain language,
lets you CORRECT a wrong prediction so the model personalizes itself, and
COMPARES three ML algorithms head-to-head to justify which one is deployed.

SYLLABUS MODULE MAPPING
------------------------
Module 1 - Intelligent Agents: the whole app is a goal-based agent that
           perceives (reads mail), reasons (classifies/prioritizes), and
           acts (applies Gmail labels).
Module 4 - Knowledge Representation & Reasoning: rule_filter.py is a
           rule-based / forward-chaining reasoning layer used as a fast,
           transparent first pass before the ML model runs.
Module 5 - Machine Learning: classifiers.py trains and compares Naive
           Bayes, Linear SVM, and a Decision Tree on TF-IDF features of
           the email text; feedback.py implements personalized learning
           from user corrections.
Module 6 - AI Ethics & Explainable AI: explainability.py shows exactly
           which words drove each decision; the Gmail integration
           defaults to a safe dry-run mode and requests only the minimum
           OAuth scope needed (read + label, never send/delete).

HOW TO USE THIS APP
--------------------
1. "Classify Email" tab - paste in any email's sender/subject/body and
   click Classify. If the category or priority looks wrong, pick the
   correct one and click "Submit Correction & Retrain" - the app learns
   from that immediately.
2. "Model Training" tab - regenerate the training dataset or re-run the
   algorithm comparison at any time; the best-performing algorithm
   (by macro F1 score) is automatically redeployed.
3. "Gmail Inbox" tab - connect this app to a real Gmail account (one-time
   setup with your own Google Cloud OAuth credentials, see README.md) and
   see your real inbox classified. Nothing is changed in your mailbox
   until you tick off "dry run" and click Apply.

All trained models, the training dataset, and Gmail login tokens are
stored privately on this computer in:
    {user_data_dir}
"""


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1000x700")
        self.minsize(860, 600)

        self.pipeline = None
        self._last_result = None
        self._last_email = None
        self._gmail_service = None
        self._gmail_results = []
        self._busy_widgets = []

        self._build_statusbar()
        self._build_tabs()

        self.after(150, self._startup_bootstrap)

    # ------------------------------------------------------------------
    # Layout scaffolding
    # ------------------------------------------------------------------
    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="Starting...")
        bar = ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken", padding=(6, 3))
        bar.pack(side="bottom", fill="x")

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.tab_classify = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_gmail = ttk.Frame(self.notebook)
        self.tab_about = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_classify, text="Classify Email")
        self.notebook.add(self.tab_train, text="Model Training")
        self.notebook.add(self.tab_gmail, text="Gmail Inbox")
        self.notebook.add(self.tab_about, text="About")

        self._build_classify_tab()
        self._build_train_tab()
        self._build_gmail_tab()
        self._build_about_tab()

    # ------------------------------------------------------------------
    # Background task helper (keeps the GUI responsive)
    # ------------------------------------------------------------------
    def _run_bg(self, fn, on_done=None):
        def worker():
            try:
                result = fn()
                err = None
            except Exception as e:  # noqa - surfaced to the user via messagebox
                result = None
                err = e
            self.after(0, lambda: on_done(result, err) if on_done else None)

        threading.Thread(target=worker, daemon=True).start()

    def _set_busy(self, busy: bool):
        try:
            self.config(cursor="watch" if busy else "")
        except Exception:
            pass
        state = "disabled" if busy else "normal"
        for w in self._busy_widgets:
            try:
                w.configure(state=state)
            except Exception:
                pass
        self.update_idletasks()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def _startup_bootstrap(self):
        self._set_busy(True)
        self.status_var.set("Preparing AI models (first run generates data & trains models)...")

        def task():
            return EmailClassifierPipeline(
                progress_callback=lambda m: self.after(0, lambda: self.status_var.set(m))
            )

        def done(result, err):
            self._set_busy(False)
            if err:
                messagebox.showerror("Startup error", str(err))
                self.status_var.set("Startup failed - see error dialog.")
                return
            self.pipeline = result
            self.status_var.set(f"Ready. Deployed model: {self.pipeline.model_name}")
            self._refresh_train_tab_labels()

        self._run_bg(task, done)

    # ------------------------------------------------------------------
    # Tab 1: Classify Email
    # ------------------------------------------------------------------
    def _build_classify_tab(self):
        frame = self.tab_classify

        input_frame = ttk.LabelFrame(frame, text="Email Input")
        input_frame.pack(fill="x", padx=10, pady=10)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="From (sender email):").grid(row=0, column=0, sticky="w", padx=5, pady=4)
        self.sender_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.sender_var).grid(row=0, column=1, sticky="we", padx=5, pady=4)

        ttk.Label(input_frame, text="Subject:").grid(row=1, column=0, sticky="w", padx=5, pady=4)
        self.subject_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.subject_var).grid(row=1, column=1, sticky="we", padx=5, pady=4)

        ttk.Label(input_frame, text="Body:").grid(row=2, column=0, sticky="nw", padx=5, pady=4)
        self.body_text = tk.Text(input_frame, height=6, wrap="word")
        self.body_text.grid(row=2, column=1, sticky="we", padx=5, pady=4)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", padx=10)
        btn_classify = ttk.Button(btn_frame, text="Classify Email", command=self._on_classify)
        btn_classify.pack(side="left")
        ttk.Button(btn_frame, text="Load Example", command=self._on_load_example).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Clear", command=self._on_clear_classify).pack(side="left")
        self._busy_widgets.append(btn_classify)

        result_frame = ttk.LabelFrame(frame, text="Result")
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)

        top_row = ttk.Frame(result_frame)
        top_row.pack(fill="x", padx=5, pady=6)
        self.result_category_var = tk.StringVar(value="-")
        self.result_priority_var = tk.StringVar(value="-")
        self.result_source_var = tk.StringVar(value="-")
        for col, (label, var) in enumerate([
            ("Category:", self.result_category_var),
            ("Priority:", self.result_priority_var),
            ("Decision source:", self.result_source_var),
        ]):
            ttk.Label(top_row, text=label, font=("", 10, "bold")).grid(row=0, column=col * 2, sticky="w")
            ttk.Label(top_row, textvariable=var).grid(row=0, column=col * 2 + 1, sticky="w", padx=(4, 25))

        ttk.Label(result_frame, text="Why this category:", font=("", 10, "bold")).pack(anchor="w", padx=5)
        self.explanation_box = scrolledtext.ScrolledText(result_frame, height=3, wrap="word", state="disabled")
        self.explanation_box.pack(fill="x", padx=5, pady=(0, 8))

        ttk.Label(result_frame, text="Why this priority:", font=("", 10, "bold")).pack(anchor="w", padx=5)
        self.priority_box = scrolledtext.ScrolledText(result_frame, height=3, wrap="word", state="disabled")
        self.priority_box.pack(fill="x", padx=5, pady=(0, 8))

        corr_frame = ttk.LabelFrame(frame, text="Not right? Correct it (the model personalizes to you)")
        corr_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(corr_frame, text="Correct category:").pack(side="left", padx=5, pady=6)
        self.correction_var = tk.StringVar()
        self.correction_combo = ttk.Combobox(corr_frame, textvariable=self.correction_var,
                                              values=feedback.VALID_LABELS, state="readonly", width=18)
        self.correction_combo.pack(side="left", padx=5)
        btn_correct = ttk.Button(corr_frame, text="Submit Correction & Retrain", command=self._on_submit_correction)
        btn_correct.pack(side="left", padx=10)
        self._busy_widgets.append(btn_correct)

    def _on_load_example(self):
        import random
        ex = random.choice(EXAMPLE_EMAILS)
        self.sender_var.set(ex["sender"])
        self.subject_var.set(ex["subject"])
        self.body_text.delete("1.0", "end")
        self.body_text.insert("1.0", ex["body"])

    def _on_clear_classify(self):
        self.sender_var.set("")
        self.subject_var.set("")
        self.body_text.delete("1.0", "end")
        self.result_category_var.set("-")
        self.result_priority_var.set("-")
        self.result_source_var.set("-")
        self._set_readonly(self.explanation_box, "")
        self._set_readonly(self.priority_box, "")
        self._last_result = None
        self._last_email = None

    def _set_readonly(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _on_classify(self):
        if not self.pipeline:
            messagebox.showwarning("Please wait", "AI models are still loading, try again in a moment.")
            return
        sender = self.sender_var.get().strip()
        subject = self.subject_var.get().strip()
        body = self.body_text.get("1.0", "end").strip()
        if not (sender or subject or body):
            messagebox.showwarning("Empty email", "Please enter at least a subject or a body.")
            return

        result = self.pipeline.classify_email(sender, subject, body)
        self._last_result = result
        self._last_email = dict(sender=sender, subject=subject, body=body)

        self.result_category_var.set(result["predicted_category"])
        self.result_priority_var.set(f"{result['priority_band']} ({result['priority_score']}/100)")
        self.result_source_var.set(result["decision_source"])
        self._set_readonly(self.explanation_box, result["explanation_text"])
        self._set_readonly(self.priority_box, "; ".join(result["priority_reasons"]))
        self.correction_var.set(result["predicted_category"])

    def _on_submit_correction(self):
        if not self._last_email:
            messagebox.showinfo("Nothing to correct", "Classify an email first, then correct it if needed.")
            return
        correct_label = self.correction_var.get()
        if not correct_label:
            messagebox.showwarning("Choose a category", "Select the correct category first.")
            return

        self._set_busy(True)
        self.status_var.set("Recording your correction and retraining the model...")

        def task():
            feedback.record_correction(self._last_email["sender"], self._last_email["subject"],
                                        self._last_email["body"], correct_label)
            feedback.retrain_with_feedback()
            return EmailClassifierPipeline()

        def done(result, err):
            self._set_busy(False)
            if err:
                messagebox.showerror("Retrain error", str(err))
                self.status_var.set("Error during retraining.")
                return
            self.pipeline = result
            self.status_var.set(f"Correction learned. Deployed model: {self.pipeline.model_name}")
            self._refresh_train_tab_labels()
            messagebox.showinfo("Retrained", "Thanks! The model has been retrained with your correction.")

        self._run_bg(task, done)

    # ------------------------------------------------------------------
    # Tab 2: Model Training
    # ------------------------------------------------------------------
    def _build_train_tab(self):
        frame = self.tab_train

        ttk.Label(frame, wraplength=850, justify="left", text=(
            "Retrain and compare three Module-5 algorithms - Naive Bayes, Linear SVM, and a "
            "Decision Tree - on the current training dataset. The best one (by macro F1 score) "
            "is automatically redeployed for classification."
        )).pack(anchor="w", padx=10, pady=10)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", padx=10)
        btn_regen = ttk.Button(btn_frame, text="Regenerate Sample Dataset", command=self._on_regenerate_dataset)
        btn_regen.pack(side="left")
        btn_train = ttk.Button(btn_frame, text="Train & Compare Models", command=self._on_train_models)
        btn_train.pack(side="left", padx=8)
        self._busy_widgets.extend([btn_regen, btn_train])

        self.dataset_size_var = tk.StringVar(value="Dataset: (loading...)")
        ttk.Label(frame, textvariable=self.dataset_size_var).pack(anchor="w", padx=10, pady=(10, 0))

        self.deployed_model_var = tk.StringVar(value="Deployed model: (loading...)")
        ttk.Label(frame, textvariable=self.deployed_model_var, font=("", 10, "bold")).pack(anchor="w", padx=10, pady=(2, 10))

        columns = ("algorithm", "accuracy", "macro_f1", "weighted_f1")
        self.tree_results = ttk.Treeview(frame, columns=columns, show="headings", height=6)
        for col, label in zip(columns, ["Algorithm", "Accuracy", "Macro F1", "Weighted F1"]):
            self.tree_results.heading(col, text=label)
            self.tree_results.column(col, width=180, anchor="center")
        self.tree_results.pack(fill="x", padx=10, pady=10)

    def _on_regenerate_dataset(self):
        self._set_busy(True)
        self.status_var.set("Regenerating training dataset...")

        def task():
            import generate_dataset
            generate_dataset.main()

        def done(result, err):
            self._set_busy(False)
            if err:
                messagebox.showerror("Error", str(err))
                return
            self.status_var.set("Dataset regenerated. Click 'Train & Compare Models' to retrain on it.")
            self._refresh_train_tab_labels()

        self._run_bg(task, done)

    def _on_train_models(self):
        self._set_busy(True)
        self.status_var.set("Training Naive Bayes / SVM / Decision Tree and comparing...")

        def task():
            results, best_name, models, vec = classifiers.train_and_compare(paths.emails_csv())
            return results, best_name

        def done(result, err):
            self._set_busy(False)
            if err:
                messagebox.showerror("Training error", str(err))
                return
            results, best_name = result
            for row in self.tree_results.get_children():
                self.tree_results.delete(row)
            for name, r in results.items():
                marker = "  <-- selected" if name == best_name else ""
                self.tree_results.insert("", "end", values=(
                    name + marker, f"{r['accuracy']*100:.2f}%", f"{r['macro_f1']*100:.2f}%", f"{r['weighted_f1']*100:.2f}%"
                ))
            self.status_var.set("Training complete. Reloading deployed model...")
            self.pipeline = EmailClassifierPipeline()
            self.deployed_model_var.set(f"Deployed model: {self.pipeline.model_name}")
            self.status_var.set(f"Ready. Deployed model: {self.pipeline.model_name}")

        self._run_bg(task, done)

    def _refresh_train_tab_labels(self):
        try:
            import pandas as pd
            df = pd.read_csv(paths.emails_csv())
            self.dataset_size_var.set(f"Dataset: {len(df)} emails across {df['label'].nunique()} categories")
        except Exception:
            self.dataset_size_var.set("Dataset: (not generated yet)")
        if self.pipeline:
            self.deployed_model_var.set(f"Deployed model: {self.pipeline.model_name}")

    # ------------------------------------------------------------------
    # Tab 3: Gmail Inbox
    # ------------------------------------------------------------------
    def _build_gmail_tab(self):
        frame = self.tab_gmail

        ttk.Label(frame, justify="left", wraplength=900, text=(
            "One-time setup: create a Gmail API OAuth 'credentials.json' (see README.md) and place it "
            "in the folder below. Then click Connect - the first time, a browser window will ask you to "
            "sign in and approve access."
        )).pack(anchor="w", padx=10, pady=10)

        path_frame = ttk.Frame(frame)
        path_frame.pack(fill="x", padx=10)
        ttk.Label(path_frame, text="App data / credentials folder:").pack(side="left")
        self.creds_folder_var = tk.StringVar(value=paths.get_user_data_dir())
        ttk.Entry(path_frame, textvariable=self.creds_folder_var, state="readonly").pack(
            side="left", padx=5, fill="x", expand=True)
        ttk.Button(path_frame, text="Open Folder", command=self._on_open_creds_folder).pack(side="left")

        ctrl_frame = ttk.Frame(frame)
        ctrl_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(ctrl_frame, text="Emails to fetch:").pack(side="left")
        self.max_emails_var = tk.IntVar(value=15)
        ttk.Spinbox(ctrl_frame, from_=1, to=100, textvariable=self.max_emails_var, width=6).pack(side="left", padx=5)
        self.dry_run_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Dry run (preview only - no mailbox changes)",
                         variable=self.dry_run_var).pack(side="left", padx=15)
        btn_connect = ttk.Button(ctrl_frame, text="Connect & Fetch Inbox", command=self._on_connect_fetch)
        btn_connect.pack(side="left", padx=5)
        self.apply_btn = ttk.Button(ctrl_frame, text="Apply Labels to Gmail", command=self._on_apply_labels,
                                     state="disabled")
        self.apply_btn.pack(side="left", padx=5)
        self._busy_widgets.extend([btn_connect, self.apply_btn])

        columns = ("subject", "sender", "category", "priority", "source")
        self.tree_gmail = ttk.Treeview(frame, columns=columns, show="headings", height=14)
        for col, label, w in zip(columns, ["Subject", "From", "Category", "Priority", "Source"],
                                  [300, 220, 130, 110, 150]):
            self.tree_gmail.heading(col, text=label)
            self.tree_gmail.column(col, width=w, anchor="w")
        self.tree_gmail.pack(fill="both", expand=True, padx=10, pady=10)

    def _on_open_creds_folder(self):
        folder = paths.get_user_data_dir()
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')
        except Exception:
            messagebox.showinfo("Folder location", folder)

    def _on_connect_fetch(self):
        if not self.pipeline:
            messagebox.showwarning("Please wait", "AI models are still loading, try again in a moment.")
            return
        self._set_busy(True)
        self.status_var.set("Connecting to Gmail (a browser window may open for sign-in)...")
        max_n = self.max_emails_var.get()

        def task():
            import gmail_client
            service = gmail_client.get_gmail_service()
            ids = gmail_client.list_recent_messages(service, max_results=max_n)
            classified = []
            for msg_id in ids:
                content = gmail_client.get_message_content(service, msg_id)
                result = self.pipeline.classify_email(content["sender"], content["subject"], content["body"])
                classified.append((content, result))
            return service, classified

        def done(result, err):
            self._set_busy(False)
            if err:
                messagebox.showerror("Gmail connection failed", str(err))
                self.status_var.set("Gmail connection failed - see error dialog.")
                return
            service, classified = result
            self._gmail_service = service
            self._gmail_results = classified
            for row in self.tree_gmail.get_children():
                self.tree_gmail.delete(row)
            for content, res in classified:
                self.tree_gmail.insert("", "end", values=(
                    content["subject"][:70], content["sender"][:45],
                    res["predicted_category"], res["priority_band"], res["decision_source"]
                ))
            self.apply_btn.configure(state="normal")
            self.status_var.set(f"Fetched & classified {len(classified)} emails. Review, then Apply if correct.")

        self._run_bg(task, done)

    def _on_apply_labels(self):
        if not self._gmail_results or not self._gmail_service:
            messagebox.showinfo("Nothing to apply", "Connect and fetch your inbox first.")
            return
        dry = self.dry_run_var.get()
        if not dry:
            if not messagebox.askyesno("Confirm",
                                        "This will create/apply labels on your REAL Gmail account. Continue?"):
                return

        self._set_busy(True)
        self.status_var.set("Applying labels..." if not dry else "Dry run: computing intended actions...")

        def task():
            import gmail_client
            actions = []
            for content, res in self._gmail_results:
                a = gmail_client.apply_label(self._gmail_service, content["id"], res["predicted_category"],
                                              dry_run=dry)
                p = gmail_client.apply_label(self._gmail_service, content["id"], f"Priority-{res['priority_band']}",
                                              dry_run=dry)
                actions.append((a, p))
            return actions

        def done(result, err):
            self._set_busy(False)
            if err:
                messagebox.showerror("Error applying labels", str(err))
                return
            msg = "Dry run complete - no mailbox changes were made." if dry else "Labels applied to your Gmail account."
            messagebox.showinfo("Done", msg)
            self.status_var.set(msg)

        self._run_bg(task, done)

    # ------------------------------------------------------------------
    # Tab 4: About
    # ------------------------------------------------------------------
    def _build_about_tab(self):
        frame = self.tab_about
        box = scrolledtext.ScrolledText(frame, wrap="word")
        box.pack(fill="both", expand=True, padx=10, pady=10)
        box.insert("1.0", ABOUT_TEXT.format(user_data_dir=paths.get_user_data_dir()))
        box.configure(state="disabled")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
