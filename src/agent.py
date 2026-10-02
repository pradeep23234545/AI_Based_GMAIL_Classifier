"""
agent.py
--------
The autonomous Gmail AI agent - no GUI, no exe, nothing to keep open.

WHAT IT DOES
------------
Each time it runs, it:
  1. Signs in to your Gmail account (using the cached token after the first
     time).
  2. Finds inbox mail it hasn't handled yet (it marks every message it
     processes with an "AI/Processed" label, so re-running never repeats
     work).
  3. Runs each new email through the same pipeline as the rest of this
     project (rule engine -> ML model -> explainability -> priority score).
  4. Applies the category + priority as real Gmail labels (and, if you've
     opted in, archives low-priority mail out of the inbox).
  5. Logs everything it did (or would do) to a log file for you to audit.

SAFETY BY DESIGN (see README's Ethics section / Module 6)
-----------------------------------------------------------
The agent starts in **dry-run mode**: it classifies mail and logs exactly
what it *would* label, but changes nothing in your mailbox and does not
mark anything as processed (so nothing is missed once you do turn it on).
You explicitly opt in to real changes with `--enable`. This mirrors the
least-privilege / transparency choices made throughout this project.

HOW TO RUN IT
-------------
One-off manual run (e.g. to check the log):
    python agent.py --once

Turn on real label application from now on:
    python agent.py --once --enable

Run continuously in a terminal you leave open, checking every 15 minutes:
    python agent.py --loop --interval-minutes 15

Run automatically in the background with no terminal open at all: use
install_windows_task.bat (Windows Task Scheduler) or install_cron.sh
(macOS/Linux cron) in the project root - see README.md section "Running
as a background agent".
"""

import argparse
import json
import logging
import os
import sys
import time

import paths
import gmail_client
from pipeline import EmailClassifierPipeline

DEFAULT_CONFIG = {
    "auto_apply": False,        # False = dry run (log only, never touches Gmail)
    "max_per_run": 25,          # how many new emails to process per run
    "archive_low_priority": False,  # also remove Promotions/Spam/Social from Inbox
    "interval_minutes": 15,     # only used by --loop
}

PROCESSED_LABEL = "Processed"
LOW_PRIORITY_CATEGORIES = {"Promotions", "Spam", "Social"}


# ----------------------------------------------------------------------
# Config persistence (~/.gmail_ai_classifier/agent_config.json)
# ----------------------------------------------------------------------
def load_config() -> dict:
    path = paths.agent_config_path()
    if not os.path.exists(path):
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        cfg = {}
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    return merged


def save_config(cfg: dict):
    with open(paths.agent_config_path(), "w") as f:
        json.dump(cfg, f, indent=2)


# ----------------------------------------------------------------------
# Logging (console + persistent log file, so a scheduled/background run
# still leaves a readable trail)
# ----------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("gmail_agent")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(paths.agent_log_path())
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)
    return logger


# ----------------------------------------------------------------------
# Core work
# ----------------------------------------------------------------------
def run_once(logger: logging.Logger, cfg: dict, pipeline: EmailClassifierPipeline, service) -> int:
    """Process one batch of not-yet-handled inbox mail. Returns how many
    emails were found (processed or, in dry-run, just reported)."""
    query = f"in:inbox -label:AI/{PROCESSED_LABEL}"
    ids = gmail_client.list_recent_messages(service, max_results=cfg["max_per_run"], query=query)

    if not ids:
        logger.info("No new emails to process.")
        return 0

    dry = not cfg["auto_apply"]
    logger.info(f"Found {len(ids)} new email(s) to process. Mode: {'DRY-RUN' if dry else 'APPLYING'}")

    for msg_id in ids:
        try:
            content = gmail_client.get_message_content(service, msg_id)
            result = pipeline.classify_email(content["sender"], content["subject"], content["body"])

            remove_from_inbox = cfg["archive_low_priority"] and result["predicted_category"] in LOW_PRIORITY_CATEGORIES

            gmail_client.apply_label(service, msg_id, result["predicted_category"],
                                      remove_from_inbox=remove_from_inbox, dry_run=dry)
            gmail_client.apply_label(service, msg_id, f"Priority-{result['priority_band']}", dry_run=dry)
            if not dry:
                # Mark as handled so future runs never reprocess this email.
                gmail_client.apply_label(service, msg_id, PROCESSED_LABEL, dry_run=False)

            mode = "APPLIED" if not dry else "would-apply"
            archived_note = " [archived]" if (not dry and remove_from_inbox) else ""
            logger.info(
                f"  [{mode}] \"{content['subject'][:60]}\" from {content['sender']} "
                f"-> {result['predicted_category']} / {result['priority_band']}{archived_note} "
                f"({result['decision_source']})"
            )
        except Exception as e:  # keep the batch going even if one message fails
            logger.info(f"  ERROR processing message {msg_id}: {e}")

    return len(ids)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Autonomous Gmail AI classification agent")
    parser.add_argument("--once", action="store_true",
                         help="Process the current batch once and exit (used by Task Scheduler/cron)")
    parser.add_argument("--loop", action="store_true",
                         help="Run forever, checking every --interval-minutes (for a terminal you leave open)")
    parser.add_argument("--interval-minutes", type=int, default=None, help="Override the loop interval")
    parser.add_argument("--enable", action="store_true",
                         help="Turn ON auto-apply (real label/archive changes) from now on, and save that setting")
    parser.add_argument("--disable", action="store_true",
                         help="Turn OFF auto-apply (back to safe dry-run/logging only), and save that setting")
    parser.add_argument("--max", type=int, default=None, help="Override how many emails to process per run")
    parser.add_argument("--archive-low-priority", action="store_true",
                         help="Also archive Promotions/Spam/Social out of the Inbox (saved setting)")
    args = parser.parse_args()

    cfg = load_config()
    changed = False
    if args.interval_minutes is not None:
        cfg["interval_minutes"] = args.interval_minutes
        changed = True
    if args.max is not None:
        cfg["max_per_run"] = args.max
        changed = True
    if args.archive_low_priority:
        cfg["archive_low_priority"] = True
        changed = True
    if args.enable:
        cfg["auto_apply"] = True
        changed = True
    if args.disable:
        cfg["auto_apply"] = False
        changed = True
    if changed:
        save_config(cfg)

    logger = setup_logging()
    logger.info("=" * 78)
    logger.info(
        f"Gmail AI Agent starting | auto_apply={cfg['auto_apply']} | "
        f"max_per_run={cfg['max_per_run']} | archive_low_priority={cfg['archive_low_priority']}"
    )
    if not cfg["auto_apply"]:
        logger.info("Running in DRY-RUN mode - no mailbox changes will be made. "
                     "Run with --enable once you're happy with the predictions in this log.")

    logger.info("Authenticating with Gmail...")
    service = gmail_client.get_gmail_service()
    pipeline = EmailClassifierPipeline()
    logger.info(f"Model loaded: {pipeline.model_name}")

    if args.loop:
        interval = cfg["interval_minutes"]
        logger.info(f"Looping every {interval} minute(s). Press Ctrl+C to stop.")
        try:
            while True:
                run_once(logger, cfg, pipeline, service)
                time.sleep(interval * 60)
        except KeyboardInterrupt:
            logger.info("Agent stopped by user (Ctrl+C).")
    else:
        # Default behaviour, and what --once uses: a single pass then exit.
        run_once(logger, cfg, pipeline, service)
        logger.info("Run complete.")


if __name__ == "__main__":
    main()
