"""
demo.py
-------
Offline end-to-end demo: runs the full pipeline (rule engine -> ML model ->
explainability -> priority scoring) on a handful of brand-new example
emails the model has never seen, so you can see real predictions without
needing a live Gmail connection. Run this any time to sanity-check the
prototype.
"""

from pipeline import EmailClassifierPipeline

TEST_EMAILS = [
    dict(sender="notifications@linkedin.com", subject="You appeared in 9 searches this week",
         body="See who's been looking at your profile and grow your network."),
    dict(sender="offers@myntra.com", subject="Flat 60% OFF Ends Tonight",
         body="Hurry! Use code SAVE60 to get flat 60% off on all fashion brands, sale ends tonight."),
    dict(sender="alerts@hdfcbank.com", subject="Debit Alert",
         body="Rs.4599 has been debited from your account ending 4321 towards EMI payment."),
    dict(sender="win@prize-mail.net", subject="You Have Won!!!",
         body="Congratulations! You have WON a lottery of $1,000,000. Click here to claim now!!!"),
    dict(sender="careers@google.com", subject="Interview Invitation",
         body="You have been shortlisted for the AI Intern role. Please join the interview tomorrow at 10 AM, this is time sensitive."),
    dict(sender="priya.sharma@gmail.com", subject="Dinner this weekend?",
         body="Hey! Are you free for dinner this Saturday? Let me know, would love to catch up."),
    dict(sender="billing@netflix.com", subject="Your subscription renews soon",
         body="Your Netflix subscription of Rs.649 will renew in 3 days. Update payment method if needed."),
    dict(sender="exam-cell@university.edu", subject="Semester Exam Schedule",
         body="Your end-semester examination for Artificial Intelligence is scheduled next week, please check the portal immediately."),
]


def main():
    pipeline = EmailClassifierPipeline()
    print(f"\nLoaded deployed model: {pipeline.model_name}\n")
    print("=" * 100)
    for email in TEST_EMAILS:
        result = pipeline.classify_email(**email)
        print(f"From      : {email['sender']}")
        print(f"Subject   : {email['subject']}")
        print(f"Category  : {result['predicted_category']}   (source: {result['decision_source']})")
        print(f"Priority  : {result['priority_band']}  (score {result['priority_score']}/100)")
        print(f"Why       : {result['explanation_text']}")
        print(f"Priority why: {'; '.join(result['priority_reasons'])}")
        print("-" * 100)


if __name__ == "__main__":
    main()
