"""
paths.py
--------
Centralised path handling so the project works IDENTICALLY whether it's run
as plain Python scripts (`python src/demo.py`) or as a packaged standalone
.exe (built with PyInstaller - see build_exe.bat).

Why this file exists: a frozen .exe's own folder is not reliably writable
(Program Files, or a temp extraction folder for a --onefile build), so all
data this app creates or changes at runtime - the trained models, the
training dataset, user corrections, and the cached Gmail OAuth token - is
kept in a per-user application-data folder instead:

    Windows : C:\\Users\\<you>\\.gmail_ai_classifier\\
    macOS/Linux : ~/.gmail_ai_classifier/

On first run this folder is empty; `bootstrap.py` notices that and
generates the dataset + trains the models straight into it. Nothing is
ever written next to the .exe itself, so the app works even when installed
to a read-only location and needs no admin rights.
"""

import os

APP_DIR_NAME = ".gmail_ai_classifier"


def get_user_data_dir() -> str:
    base = os.path.join(os.path.expanduser("~"), APP_DIR_NAME)
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "data"), exist_ok=True)
    os.makedirs(os.path.join(base, "models"), exist_ok=True)
    return base


def data_dir() -> str:
    d = os.path.join(get_user_data_dir(), "data")
    os.makedirs(d, exist_ok=True)
    return d


def models_dir() -> str:
    d = os.path.join(get_user_data_dir(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def emails_csv() -> str:
    return os.path.join(data_dir(), "emails.csv")


def corrections_csv() -> str:
    return os.path.join(data_dir(), "corrections.csv")


def merged_csv() -> str:
    return os.path.join(data_dir(), "emails_with_feedback.csv")


def credentials_path() -> str:
    """Where the user should drop their Gmail OAuth credentials.json.
    Kept in the user data dir (not the install folder) so it survives
    reinstalls/updates of the .exe."""
    return os.path.join(get_user_data_dir(), "credentials.json")


def token_path() -> str:
    return os.path.join(get_user_data_dir(), "token.json")


def agent_config_path() -> str:
    return os.path.join(get_user_data_dir(), "agent_config.json")


def logs_dir() -> str:
    d = os.path.join(get_user_data_dir(), "logs")
    os.makedirs(d, exist_ok=True)
    return d


def agent_log_path() -> str:
    return os.path.join(logs_dir(), "agent.log")


def best_model_name_path() -> str:
    return os.path.join(models_dir(), "best_model_name.joblib")


def vectorizer_path() -> str:
    return os.path.join(models_dir(), "vectorizer.joblib")


def model_path(model_name: str) -> str:
    fname = model_name.lower().replace(" ", "_") + ".joblib"
    return os.path.join(models_dir(), fname)
