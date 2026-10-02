# AI-Based Gmail Email Filtering & Prioritization Agent

An AI Capstone Project (pure Python) built as an **autonomous background
agent** — no GUI to operate, no .exe to install. Point it at your Gmail
account once, and it keeps classifying and labeling new mail on its own.

## 1. Problem Statement

Gmail already sorts mail into Primary/Social/Promotions and flags spam, but
it is a black box: it doesn't explain *why* an email was filed a certain
way, it doesn't visibly learn from an individual user's corrections, and it
has no notion of *urgency* independent of category. This project builds an
intelligent **agent** that runs quietly in the background and:

1. Classifies each new email into **6 categories**: `Primary`, `Social`,
   `Promotions`, `Spam`, and two **new, non-Gmail categories** —
   `Finance_Bills` (bank alerts, invoices, EMIs, subscriptions) and
   `Job_Academic` (interview calls, offer letters, exam/semester notices).
2. **Explains** every decision in plain language (which words/rule drove it).
3. Scores every email's **priority** (Low/Medium/High/Critical) using a
   transparent hybrid rule + ML-confidence model — independent of category.
4. **Compares** three ML algorithms head-to-head and picks the best one.
5. **Learns from user corrections** — a lightweight personalization loop.
6. Runs **on its own, on a schedule**, applying real Gmail labels — never
   reprocessing the same email twice.

## 2. Why this is not "just Gmail again"

| Feature | Gmail has it? | This project |
|---|---|---|
| Primary/Social/Promotions | Yes | Reproduced as a baseline, not the novelty |
| Spam detection | Yes | Reproduced as a baseline, not the novelty |
| **Custom categories** (Finance_Bills, Job_Academic) | No | **New** |
| **Explainable classification** (why this label?) | No | **New** |
| **Custom priority model** (urgency score, reasons shown) | Partial/opaque | **New, transparent** |
| **Compare ML algorithms** (NB vs SVM vs Decision Tree) | N/A | **New, academic core** |
| **Personalized learning from corrections** | Partial/opaque | **New, lightweight version** |
| **Runs autonomously in the background** | N/A | **New** |

## 3. Syllabus Module Mapping

- **Module 1** — The whole app is a goal-based/utility-based *intelligent
  agent*: it perceives (checks the inbox on a timer), reasons (classifies
  and prioritizes), and acts (applies Gmail labels) with no human in the
  loop once it's turned on — the clearest possible demonstration of Module
  1's agent architecture.
- **Module 4** — `rule_filter.py` is a rule-based/forward-chaining reasoning
  layer used as a fast, explainable first pass before ML.
- **Module 5** — `classifiers.py` trains and compares **Naive Bayes**,
  **Linear SVM**, and **Decision Tree** on TF-IDF features; `feedback.py`
  implements a simple personalized-learning loop.
- **Module 6** — `explainability.py` (Explainable AI) and the dry-run /
  least-privilege / audit-log design of the agent (AI Ethics).

## 4. Project Structure

```
gmail_ai_classifier/
├── src/
│   ├── agent.py                 # >>> THE AGENT - this is what you run <<<
│   ├── paths.py                 # per-user app-data folder handling
│   ├── bootstrap.py             # auto-generates dataset + trains models on first run
│   ├── generate_dataset.py      # builds the synthetic labeled dataset (270 emails, 6 classes)
│   ├── preprocessing.py         # text cleaning + TF-IDF
│   ├── rule_filter.py           # Module 4: rule-based reasoning layer
│   ├── classifiers.py           # Module 5: train & compare NB / SVM / Decision Tree
│   ├── explainability.py        # Module 6: explain any prediction
│   ├── priority_model.py        # custom priority scoring
│   ├── pipeline.py              # ties rules + ML + explainability + priority together
│   ├── feedback.py              # personalized correction + retrain loop
│   ├── gmail_client.py          # Gmail API wrapper (OAuth, fetch, label)
│   ├── gui_app.py               # optional: a manual point-and-click tester (not required)
│   ├── demo.py                  # offline command-line demo (no Gmail needed)
│   └── main_live.py             # one-shot manual CLI version of the Gmail integration
├── install_windows_task.bat     # makes the agent run automatically (Windows)
├── install_cron.sh              # makes the agent run automatically (macOS/Linux)
├── requirements.txt
└── README.md
```

Everything the agent creates at runtime — trained models, the training
dataset, your corrections, the cached Gmail login token, its config, and
its log file — lives in a per-user folder, not inside this project folder:

```
Windows      C:\Users\<you>\.gmail_ai_classifier\
macOS/Linux  ~/.gmail_ai_classifier/
```

## 5. One-Time Setup

```bash
cd gmail_ai_classifier
pip install -r requirements.txt
```

Then connect a real Gmail account (needed since the agent's whole job is
to act on your real inbox):

1. Go to **Google Cloud Console** → create a new project.
2. Enable the **Gmail API** for that project.
3. Go to **APIs & Services → Credentials → Create Credentials → OAuth
   client ID**, choose **Desktop app**, and download the resulting JSON.
4. Rename it to `credentials.json` and place it in your app-data folder:
   `~/.gmail_ai_classifier/credentials.json` (Windows:
   `C:\Users\<you>\.gmail_ai_classifier\credentials.json`). Create the
   folder if it doesn't exist yet, or just run the agent once first (step
   6 below) and it will create the folder for you.
5. Only the `gmail.modify` scope is ever requested (read + label, never
   send or delete) — a deliberate least-privilege choice discussed in the
   AI Ethics section of the report (Module 6).
6. Run it once by hand to sign in and see it work:
   ```bash
   python src/agent.py --once
   ```
   The first time, a browser window opens asking you to sign in and
   approve access — this is the standard Google consent screen, and the
   login is cached afterward so you won't need to repeat it.

## 6. How It Behaves — Safe By Default

The agent starts in **dry-run mode**. It reads your inbox, classifies each
new email, and **logs exactly what it would do** — but makes no changes to
your mailbox at all. You can run `python src/agent.py --once` as many
times as you like to watch its predictions build up in the log with zero
risk.

When you're happy with what you see, turn on real changes **once**:
```bash
python src/agent.py --once --enable
```
This is saved permanently in its config — every future run (including
scheduled ones) will now actually apply labels, until you run `--disable`.

Every email it touches gets marked with an `AI/Processed` label, so a
message is never re-classified or reprocessed on a later run — the agent
always only looks at genuinely new mail (`in:inbox -label:AI/Processed`).

## 7. Running It Automatically In The Background

This is the whole point — set it up once, forget about it.

**Windows:** double-click `install_windows_task.bat`. It registers a
Windows Scheduled Task that silently runs the agent every 15 minutes
whenever you're logged in — no window, no console, nothing visible.
- Change how often it runs by editing `INTERVAL` near the top of the
  `.bat` file, then run it again (it replaces the old schedule).
- Remove it later: `schtasks /delete /tn "GmailAIAgent" /f`

**macOS/Linux:** run `./install_cron.sh`. It adds a cron entry that runs
the agent every 15 minutes.
- Change the interval by editing `INTERVAL_MINUTES` near the top of the
  script, then run it again.
- Remove it later: `crontab -e` and delete the line mentioning `agent.py`.

**Either OS, simplest option:** just leave a terminal open running:
```bash
python src/agent.py --loop --interval-minutes 15
```
Stop it any time with Ctrl+C.

All three approaches call the exact same `agent.py` — the scheduler just
decides when it wakes up.

## 8. Checking On It / Auditing What It Did

Every run appends to a plain-text log:
```
~/.gmail_ai_classifier/logs/agent.log
```
Each line shows the timestamp, the email's subject/sender, the category
and priority it was given, and whether the change was actually applied or
just logged (dry-run). This log is your evidence trail for the report and
your way of sanity-checking the agent without opening Gmail.

## 9. Other ways to use the same pipeline (optional)

The agent shares its brain with a few other entry points, useful for
building/demoing the project itself rather than running it long-term:

```bash
python src/demo.py       # classify 8 unseen sample emails, no Gmail needed
python src/feedback.py   # demo: correct a wrong prediction & retrain
python src/gui_app.py    # optional point-and-click tester, if you want to
                          # manually try single emails or re-run the model
                          # comparison — not required for the agent to work
python src/main_live.py  # a manual one-shot Gmail run with a --apply flag,
                          # if you want more control than agent.py's config
```

## 10. How Classification Works (pipeline.py)

```
Email (sender, subject, body)
        │
        ▼
1. Rule-based layer (rule_filter.py)
   e.g. sender domain = linkedin.com -> Social (confidence 0.97)
        │  (no confident rule fired)
        ▼
2. ML fallback (best of Naive Bayes / SVM / Decision Tree, chosen
   automatically by classifiers.py based on macro-F1 on a held-out test set)
        │
        ▼
3. Explainability layer (explainability.py)
   -> top words that drove the decision, algorithm-appropriate:
      Naive Bayes: log-likelihood ratio · SVM: coef_ x tfidf · Tree: feature_importances_
        │
        ▼
4. Priority scoring (priority_model.py)
   -> category base weight + urgency keywords + model confidence
   -> Low / Medium / High / Critical, with reasons logged
```

## 11. Algorithm Comparison (reproducible — run `python src/classifiers.py`)

On the bundled 270-email dataset (25% held out for testing):

| Algorithm | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes | ~100% | ~100% |
| Linear SVM | ~98.5% | ~98.5% |
| Decision Tree | ~97% | ~97% |

Naive Bayes is selected automatically (highest macro-F1). This is expected
academically: Naive Bayes is historically very strong on short, keyword-rich
text like emails (the same reason it's the classic textbook algorithm for
spam filtering), while Decision Trees tend to overfit sparse TF-IDF features
without pruning. **Your project report should present this table and this
justification — it is the strongest "compare ML algorithms" evidence.**

## 12. Ethics Notes (Module 6 — include in your report)

- Reading and auto-labeling personal email raises real privacy/autonomy
  concerns; this is exactly why the agent defaults to dry-run and requires
  an explicit, one-time `--enable` before it touches your mailbox for real.
- Only `gmail.modify` is requested — never `gmail.send` or delete
  permissions.
- Every action is logged to a plain-text, human-readable file, so an
  autonomous agent never acts invisibly.
- The `AI/Processed` marker ensures the agent's actions are idempotent —
  running it more times than intended never double-labels or re-archives
  the same email.
- The dataset used to train the models is synthetic; a real deployment
  should get explicit consent before scanning a user's inbox content.

## 13. Possible Extensions (for viva / future work)

- Replace batch retraining in `feedback.py` with true incremental learning
  (`MultinomialNB.partial_fit`), and wire a way to submit corrections
  without needing the optional GUI (e.g. by replying to a summary email).
- Use Gmail push notifications (Cloud Pub/Sub) instead of polling, so the
  agent reacts within seconds of a new email arriving instead of waiting
  for the next scheduled run.
- Extend `priority_model.py` with a learned (not hand-weighted) priority
  model trained on which emails a user actually opens/replies to quickly.
- Add action-item extraction (e.g. detect "please reply by Friday") using
  simple NLP — noted in the brief as a possible but non-essential feature,
  left out of this prototype to keep scope focused.
