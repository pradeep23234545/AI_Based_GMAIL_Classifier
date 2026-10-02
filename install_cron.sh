#!/usr/bin/env bash
# ============================================================
# install_cron.sh
# ------------------------------------------------------------
# Installs a cron job that runs the Gmail AI Agent automatically
# every N minutes on macOS/Linux - no terminal window needed,
# nothing to keep open. Your machine just needs to be on (cron
# does not run while the computer is fully asleep/off).
#
# Run this once:
#   chmod +x install_cron.sh
#   ./install_cron.sh
#
# To remove it later:
#   crontab -e     (then delete the line mentioning agent.py)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(command -v python3 || command -v python)"
INTERVAL_MINUTES=15
LOG_DIR="$HOME/.gmail_ai_classifier/logs"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Gmail AI Agent - cron installer"
echo "============================================================"
echo "This will run:"
echo "    $PYTHON_BIN $SCRIPT_DIR/src/agent.py --once"
echo "every $INTERVAL_MINUTES minutes via cron."
echo
echo "IMPORTANT: the agent starts in DRY-RUN mode (it only logs what"
echo "it would do). Once you've checked $LOG_DIR/agent.log and are"
echo "happy with the predictions, enable real changes with:"
echo "    $PYTHON_BIN $SCRIPT_DIR/src/agent.py --once --enable"
echo
read -p "Continue installing the cron job? (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

CRON_LINE="*/$INTERVAL_MINUTES * * * * $PYTHON_BIN $SCRIPT_DIR/src/agent.py --once >> $LOG_DIR/cron_stdout.log 2>&1"

( crontab -l 2>/dev/null | grep -v "gmail_ai_classifier/src/agent.py" ; echo "$CRON_LINE" ) | crontab -

echo
echo "SUCCESS: cron job installed - the agent will run every $INTERVAL_MINUTES minutes."
echo "Log file: $LOG_DIR/agent.log"
echo "To remove it later: crontab -e   (then delete the line mentioning agent.py)"
