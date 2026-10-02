"""
rule_filter.py
--------------
A small rule-based reasoning layer (Module 4: Knowledge Representation &
Reasoning - Rule-Based Systems, forward chaining). This acts as a fast,
transparent first-pass filter BEFORE the statistical ML classifier runs:

  - Obvious, high-confidence cases (e.g. sender domain is a known bank, or
    subject contains classic spam trigger phrases) are resolved instantly by
    IF-THEN rules, with a human-readable reason.
  - Anything the rules are not confident about is passed through to the ML
    pipeline (Naive Bayes / SVM / Decision Tree) for a learned decision.

This hybrid design (symbolic rules + statistical learning) mirrors classic
AI architectures taught in Module 4 and gives the whole system an
explainability boost for the "easy" cases.
"""

import re

KNOWN_SOCIAL_DOMAINS = ["linkedin.com", "facebook.com", "instagram.com", "twitter.com",
                         "x.com", "pinterest.com", "quora.com"]
KNOWN_BANK_DOMAINS = ["hdfcbank.com", "icicibank.com", "sbi.com", "axisbank.com",
                       "kotakmahindrabank.com", "paypal.com"]
PROMO_KEYWORDS = ["% off", "sale", "discount", "coupon", "deal", "cashback", "clearance",
                   "buy 1 get 1", "flat ", "offer"]
SPAM_KEYWORDS = ["you have won", "click here", "verify your password", "free iphone",
                  "lottery", "inheritance", "nigerian prince", "act now", "urgent action",
                  "hot singles", "guaranteed", "final notice"]
FINANCE_KEYWORDS = ["debited", "credited", "emi", "statement", "invoice", "bill", "premium due",
                     "balance", "recharge"]
JOB_ACADEMIC_KEYWORDS = ["shortlisted", "interview", "offer letter", "placement", "exam",
                          "semester", "assignment", "grade card", "internship"]


def domain_of(sender: str) -> str:
    m = re.search(r"@([\w\.-]+)", sender or "")
    return m.group(1).lower() if m else ""


def _domain_matches(domain: str, known_domain: str) -> bool:
    """True only if `domain` IS `known_domain` or a proper subdomain of it
    (e.g. 'mail.linkedin.com' matches 'linkedin.com'), never a loose
    substring match. This avoids false positives like 'x.com' (Twitter/X)
    wrongly matching inside 'netflix.com'."""
    return domain == known_domain or domain.endswith("." + known_domain)


def apply_rules(sender: str, subject: str, body: str):
    """
    Returns (label, confidence, reason) if a rule fires with high confidence,
    otherwise (None, 0.0, None) meaning 'defer to the ML model'.
    """
    text = f"{subject} {body}".lower()
    domain = domain_of(sender)

    # Rule 1: known social platform domain -> Social
    for d in KNOWN_SOCIAL_DOMAINS:
        if _domain_matches(domain, d):
            return "Social", 0.97, f"Sender domain '{domain}' matches known social platform"

    # Rule 2: spam trigger phrases -> Spam (checked before finance/bank, since
    # phishing mails often *impersonate* banks)
    hits = [k for k in SPAM_KEYWORDS if k in text]
    if len(hits) >= 1 and not any(_domain_matches(domain, d) for d in KNOWN_BANK_DOMAINS):
        return "Spam", 0.9, f"Matched spam trigger phrase(s): {hits}"

    # Rule 3: known bank/payment domain + finance keywords -> Finance_Bills
    if any(_domain_matches(domain, d) for d in KNOWN_BANK_DOMAINS):
        return "Finance_Bills", 0.95, f"Sender domain '{domain}' is a known bank/payment provider"

    # Rule 4: strong finance keywords -> Finance_Bills
    fhits = [k for k in FINANCE_KEYWORDS if k in text]
    if len(fhits) >= 2:
        return "Finance_Bills", 0.85, f"Matched finance/billing keyword(s): {fhits}"

    # Rule 5: strong promo keywords -> Promotions
    phits = [k for k in PROMO_KEYWORDS if k in text]
    if len(phits) >= 1:
        return "Promotions", 0.85, f"Matched promotional keyword(s): {phits}"

    # Rule 6: job/academic keywords -> Job_Academic
    jhits = [k for k in JOB_ACADEMIC_KEYWORDS if k in text]
    if len(jhits) >= 1:
        return "Job_Academic", 0.8, f"Matched job/academic keyword(s): {jhits}"

    # No confident rule fired -> defer to ML
    return None, 0.0, None
