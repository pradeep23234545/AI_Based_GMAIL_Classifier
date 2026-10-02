"""
main_live.py
------------
Runs the full pipeline against YOUR REAL Gmail inbox.

Run this on your own machine (not in a cloud sandbox) after completing the
one-time Gmail API setup in README.md. By default it runs in DRY-RUN mode:
it will fetch and classify real emails and print exactly what it *would*
do, but will NOT create labels or move anything until you pass --apply.

Usage:
    python main_live.py                # dry run, first 15 inbox emails
    python main_live.py --apply        # actually create/apply labels
    python main_live.py --max 30 --apply
    python main_live.py --archive-low-priority --apply
        (also removes the INBOX label from Promotions/Spam/Social mail,
         i.e. "moves" them out of the primary inbox view - opt-in only)
"""

import argparse
from gmail_client import get_gmail_service, list_recent_messages, get_message_content, apply_label
from pipeline import EmailClassifierPipeline

LOW_PRIORITY_CATEGORIES = {"Promotions", "Spam", "Social"}


def main():
    parser = argparse.ArgumentParser(description="AI Gmail classifier - live mode")
    parser.add_argument("--max", type=int, default=15, help="Number of recent inbox emails to process")
    parser.add_argument("--apply", action="store_true", help="Actually create/apply Gmail labels (default: dry run)")
    parser.add_argument("--archive-low-priority", action="store_true",
                         help="Also remove Promotions/Spam/Social mail from Inbox view (only with --apply)")
    args = parser.parse_args()

    print("Authenticating with Gmail...")
    service = get_gmail_service()

    print(f"Fetching {args.max} recent inbox messages...")
    ids = list_recent_messages(service, max_results=args.max)

    pipeline = EmailClassifierPipeline()
    print(f"Loaded model: {pipeline.model_name}\n")
    print(f"Mode: {'APPLYING CHANGES' if args.apply else 'DRY RUN (no mailbox changes)'}")
    print("=" * 100)

    for msg_id in ids:
        content = get_message_content(service, msg_id)
        result = pipeline.classify_email(content["sender"], content["subject"], content["body"])

        remove_from_inbox = args.archive_low_priority and result["predicted_category"] in LOW_PRIORITY_CATEGORIES

        cat_action = apply_label(service, msg_id, result["predicted_category"],
                                  remove_from_inbox=remove_from_inbox, dry_run=not args.apply)
        pri_action = apply_label(service, msg_id, f"Priority-{result['priority_band']}",
                                  dry_run=not args.apply)

        print(f"Subject   : {content['subject'][:80]}")
        print(f"From      : {content['sender']}")
        print(f"Category  : {result['predicted_category']}  ({result['decision_source']})")
        print(f"Priority  : {result['priority_band']} ({result['priority_score']}/100)")
        print(f"Why       : {result['explanation_text']}")
        print(f"Label action : {cat_action}")
        print(f"Priority label: {pri_action}")
        print("-" * 100)

    if not args.apply:
        print("\nThis was a DRY RUN - no labels were created and nothing moved.")
        print("Re-run with --apply once you're happy with the predictions above.")


if __name__ == "__main__":
    main()
