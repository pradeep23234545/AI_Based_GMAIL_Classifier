"""
generate_dataset.py
--------------------
Builds a synthetic-but-realistic labeled email dataset for training/demoing
the classifier. Six categories are used:

  1. Primary        - genuine personal / work correspondence
  2. Social          - social network notifications
  3. Promotions      - marketing / sales emails
  4. Spam            - phishing / scam / junk
  5. Finance_Bills   - NEW category: bank alerts, bills, invoices, subscriptions
  6. Job_Academic    - NEW category: job applications/offers, exam/course notices

In a real deployment this file would be replaced by emails pulled from the
user's own Gmail account (see gmail_client.py) with labels supplied either by
existing Gmail category tabs (bootstrap) or by manual correction feedback
(feedback.py). For a reproducible academic prototype, we generate data from
templates so the pipeline can be trained and evaluated end-to-end offline.
"""

import random
import csv
import os

import paths

random.seed(42)

OUT_PATH = paths.emails_csv()

FIRST_NAMES = ["Rahul", "Sneha", "Aman", "Priya", "Karan", "Isha", "Rohan",
               "Neha", "Arjun", "Divya", "Vikram", "Anjali", "Suresh", "Meera"]

COMPANIES = ["Amazon", "Flipkart", "Myntra", "Zomato", "Swiggy", "Ajio", "Nykaa"]
SOCIAL_PLATFORMS = ["LinkedIn", "Facebook", "Instagram", "Twitter/X", "Pinterest", "Quora"]
BANKS = ["HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Mahindra Bank", "PayPal"]
COMPANIES_JOB = ["TCS", "Infosys", "Wipro", "Google", "Microsoft", "Amazon India", "Accenture"]
COLLEGES = ["your University", "the Examination Cell", "the Placement Cell", "your Department"]

def primary_samples(n):
    rows = []
    templates = [
        ("Meeting reschedule", "Hi {name}, can we move tomorrow's meeting to 3 PM? Let me know if that works for you."),
        ("Project update", "Hey, attaching the latest draft of the report. Please review before Friday's sync."),
        ("Re: Weekend plans", "Sounds good! Let's meet at the cafe near the station around 6."),
        ("Quick question", "Hi, do you have the notes from today's class? I missed the second half."),
        ("Family dinner this Sunday", "Hey, mom wants everyone over for dinner this Sunday at 7pm. Can you make it?"),
        ("Following up on our call", "Thanks for the call earlier. Sending over the document we discussed."),
        ("Happy Birthday!", "Just wanted to wish you a very happy birthday! Let's catch up soon."),
        ("Your feedback needed", "Could you review the attached document and share your thoughts by tomorrow?"),
        ("Lunch tomorrow?", "Are you free for lunch tomorrow around 1? Would love to catch up."),
        ("Travel plans", "I booked the tickets for our trip next month, sending the itinerary shortly."),
    ]
    for i in range(n):
        subj, body = random.choice(templates)
        name = random.choice(FIRST_NAMES)
        sender = f"{name.lower()}.{random.choice(FIRST_NAMES).lower()}@gmail.com"
        rows.append((sender, subj, body.format(name=name), "Primary"))
    return rows

def social_samples(n):
    rows = []
    templates = [
        "{name} commented on your post: \"Great work, keep it up!\"",
        "You have {num} new connection request on {platform}.",
        "{name} tagged you in a photo on {platform}.",
        "{name} liked your post from last week.",
        "Your {platform} weekly digest: 5 posts from people you follow.",
        "{name} started following you on {platform}.",
        "You have a new message request on {platform}.",
        "{name} mentioned you in a comment on {platform}.",
        "See what's trending on {platform} this week.",
        "{name} invited you to join a group on {platform}.",
    ]
    for i in range(n):
        platform = random.choice(SOCIAL_PLATFORMS)
        name = random.choice(FIRST_NAMES)
        body = random.choice(templates).format(name=name, platform=platform, num=random.randint(1, 9))
        subj = f"New activity on {platform}"
        sender = f"notifications@{platform.split('/')[0].lower()}.com"
        rows.append((sender, subj, body, "Social"))
    return rows

def promotions_samples(n):
    rows = []
    templates = [
        "Flat {pct}% OFF on your favourite brands. Sale ends tonight, don't miss out!",
        "Exclusive deal just for you: Buy 1 Get 1 Free, only today at {company}.",
        "Your cart is waiting! Complete your purchase and get extra {pct}% off.",
        "Big Billion Days is here! Up to {pct}% off storewide at {company}.",
        "Last chance: Clearance sale up to {pct}% off ends in 24 hours.",
        "New arrivals just dropped at {company} - shop the collection now.",
        "Use code SAVE{pct} to get {pct}% off your next order at {company}.",
        "Weekend special: Free delivery + {pct}% cashback on orders above 999.",
        "Members only: early access to the {company} festive sale starts now.",
        "We miss you! Here's {pct}% off to welcome you back to {company}.",
    ]
    for i in range(n):
        company = random.choice(COMPANIES)
        pct = random.choice([10, 20, 30, 40, 50, 60, 70])
        body = random.choice(templates).format(pct=pct, company=company)
        subj = f"{company}: {pct}% OFF - Limited Time Offer"
        sender = f"offers@{company.lower()}.com"
        rows.append((sender, subj, body, "Promotions"))
    return rows

def spam_samples(n):
    rows = []
    templates = [
        "Congratulations! You have WON a lottery of $1,000,000. Click here to claim now!!!",
        "URGENT: Your account will be suspended. Verify your password immediately at this link.",
        "You've been selected for a FREE iPhone 16. Claim your prize before it expires!",
        "Dear Winner, your email was randomly selected for a cash reward of $500,000.",
        "Get rich quick! Invest $100 and earn $10,000 in a week guaranteed!!!",
        "Your package could not be delivered. Pay a small fee here to reschedule.",
        "hot singles in your area want to chat with you now, click here",
        "Nigerian prince needs your help transferring $20 million, reply for details.",
        "Your bank account has been compromised, click this link to secure it now!",
        "FINAL NOTICE: claim your inheritance of $2,000,000 within 24 hours.",
    ]
    for i in range(n):
        body = random.choice(templates)
        subj = random.choice(["You Have Won!!!", "URGENT ACTION REQUIRED", "Claim Your Prize Now",
                               "Security Alert!!!", "Act Now - Limited Time"])
        sender = f"no-reply{random.randint(100,999)}@{random.choice(['prize-mail.net','secure-verify.info','win-big.co','freegift.biz'])}"
        rows.append((sender, subj, body, "Spam"))
    return rows

def finance_samples(n):
    rows = []
    templates = [
        "Your credit card statement for this month is now available. Total due: Rs.{amt}.",
        "An amount of Rs.{amt} has been debited from your account ending in {last4}.",
        "Your electricity bill of Rs.{amt} is due on the {day}th of this month.",
        "Payment reminder: Your subscription renews for Rs.{amt} in 3 days.",
        "Your EMI of Rs.{amt} has been successfully processed.",
        "Invoice #{inv} of Rs.{amt} has been generated for your recent purchase.",
        "Your monthly bank statement is ready to view. Closing balance: Rs.{amt}.",
        "Rs.{amt} was credited to your account. Available balance: Rs.{amt2}.",
        "Your mobile recharge of Rs.{amt} was successful.",
        "Reminder: Your insurance premium of Rs.{amt} is due this week.",
    ]
    for i in range(n):
        bank = random.choice(BANKS)
        amt = random.choice([499, 999, 1499, 2499, 4999, 7999, 12000, 25000])
        amt2 = amt + random.randint(1000, 50000)
        body = random.choice(templates).format(amt=amt, amt2=amt2, last4=random.randint(1000,9999),
                                                 day=random.randint(1,28), inv=random.randint(10000,99999))
        subj = f"{bank}: Transaction / Bill Alert"
        sender = f"alerts@{bank.lower().replace(' ', '')}.com"
        rows.append((sender, subj, body, "Finance_Bills"))
    return rows

def job_academic_samples(n):
    rows = []
    templates = [
        "We are pleased to inform you that you have been shortlisted for the {role} position. Next round: technical interview.",
        "Thank you for applying to {company}. Your application is under review.",
        "Congratulations! Your offer letter for the role of {role} at {company} is attached.",
        "Reminder: Your end-semester examination for {subject} is scheduled next week.",
        "{college} has released the internal assessment marks. Please check the portal.",
        "Interview scheduled: Please join the technical interview for {role} at {company} on Monday, 10 AM.",
        "Your assignment submission for {subject} is due by Friday midnight.",
        "Placement drive: {company} will be visiting campus for {role} recruitment next week.",
        "Your semester result has been declared. Login to the portal to check your grade card.",
        "Workshop registration confirmed: You are registered for the AI & Machine Learning workshop.",
    ]
    for i in range(n):
        role = random.choice(["Software Engineer", "Data Analyst", "AI Intern", "Backend Developer", "Research Intern"])
        company = random.choice(COMPANIES_JOB)
        college = random.choice(COLLEGES)
        subject = random.choice(["Artificial Intelligence", "Database Systems", "Operating Systems", "Mathematics"])
        body = random.choice(templates).format(role=role, company=company, college=college, subject=subject)
        subj = f"{company if 'Interview' in body or 'shortlisted' in body or 'offer' in body else college}: Update"
        sender = f"careers@{company.lower().replace(' ', '')}.com" if random.random() > 0.4 else f"exam-cell@{'university'}.edu"
        rows.append((sender, subj, body, "Job_Academic"))
    return rows

def main():
    n_per_class = 45
    rows = []
    rows += primary_samples(n_per_class)
    rows += social_samples(n_per_class)
    rows += promotions_samples(n_per_class)
    rows += spam_samples(n_per_class)
    rows += finance_samples(n_per_class)
    rows += job_academic_samples(n_per_class)
    random.shuffle(rows)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sender", "subject", "body", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    from collections import Counter
    print(Counter(r[3] for r in rows))

if __name__ == "__main__":
    main()
