"""
gmail_client.py
----------------
Thin wrapper around the Gmail API (google-api-python-client) for:
  - OAuth2 authentication (runs a local browser consent flow once, then
    reuses a cached token.json)
  - Fetching recent inbox messages (sender, subject, body)
  - Creating custom Gmail labels for our categories/priority bands
  - Applying a label to a message (our "segmentation"/"filtering" action -
    Gmail doesn't have folders, it has labels, and removing the INBOX label
    is the equivalent of "moving it out of the inbox")

IMPORTANT - this file talks to a REAL Gmail account and must be run on YOUR
OWN machine, not inside a sandboxed cloud session, because:
  1. It needs a `credentials.json` OAuth client file that only you can
     generate from your own Google Cloud Console project (Anthropic/Claude
     has no access to your Gmail account and never will).
  2. The OAuth consent step opens a real browser window on your machine.

See ../README.md for the exact one-time setup steps.
"""

import os
import base64
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import paths

# Minimum scopes needed: read messages + modify labels. We deliberately do
# NOT request the broader gmail.settings or delete/send scopes - this project
# only reads and re-labels mail, in line with the ethics/least-privilege
# discussion in Module 6.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

CREDENTIALS_PATH = paths.credentials_path()
TOKEN_PATH = paths.token_path()

LABEL_PREFIX = "AI"  # all labels we create are namespaced under "AI/..."


def get_gmail_service():
    """Authenticate (via cached token, or a fresh OAuth consent flow the
    first time) and return an authorized Gmail API service object."""
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as exc:
                error_response = next(
                    (arg for arg in exc.args if isinstance(arg, dict)), {}
                )
                if error_response.get("error") != "invalid_grant":
                    raise
                creds = None

        if not creds or not creds.valid:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    "credentials.json not found. Download your OAuth client "
                    "credentials from Google Cloud Console and place them at "
                    f"{os.path.abspath(CREDENTIALS_PATH)}. See README.md."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def _decode_part(data: str) -> str:
    return base64.urlsafe_b64decode(data.encode("ASCII")).decode("utf-8", errors="ignore")


def _extract_body(payload) -> str:
    """Walk the (possibly multipart) MIME payload and return the best-effort
    plain-text body."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return _decode_part(payload["body"]["data"])

    if "parts" in payload:
        # Prefer text/plain; fall back to text/html stripped of tags.
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                return _decode_part(part["body"]["data"])
        for part in payload["parts"]:
            text = _extract_body(part)
            if text:
                return text

    if payload.get("mimeType") == "text/html" and payload.get("body", {}).get("data"):
        import re
        html = _decode_part(payload["body"]["data"])
        return re.sub("<[^<]+?>", " ", html)

    return ""


def list_recent_messages(service, max_results: int = 20, query: str = "in:inbox"):
    """Return a list of message IDs matching the query (default: inbox)."""
    resp = service.users().messages().list(userId="me", maxResults=max_results, q=query).execute()
    return [m["id"] for m in resp.get("messages", [])]


def get_message_content(service, msg_id: str):
    """Fetch one message and return {sender, subject, body}."""
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
    sender = headers.get("from", "")
    subject = headers.get("subject", "")
    body = _extract_body(msg["payload"])
    return {"id": msg_id, "sender": sender, "subject": subject, "body": body}


_label_cache = {}


def ensure_label(service, label_name: str) -> str:
    """Get-or-create a Gmail label (namespaced e.g. 'AI/Finance_Bills') and
    return its label ID."""
    full_name = f"{LABEL_PREFIX}/{label_name}"
    if full_name in _label_cache:
        return _label_cache[full_name]

    resp = service.users().labels().list(userId="me").execute()
    for label in resp.get("labels", []):
        if label["name"] == full_name:
            _label_cache[full_name] = label["id"]
            return label["id"]

    created = service.users().labels().create(
        userId="me",
        body={"name": full_name, "labelListVisibility": "labelShow", "messageListVisibility": "show"},
    ).execute()
    _label_cache[full_name] = created["id"]
    return created["id"]


def apply_label(service, msg_id: str, label_name: str, remove_from_inbox: bool = False, dry_run: bool = True):
    """Apply a (namespaced) label to a message, optionally archiving it out
    of the inbox. When dry_run=True (the default), NO changes are made to
    the mailbox - the intended action is only returned/printed. This keeps
    the tool safe to demo without risking real inbox changes until the user
    explicitly opts in, per the ethics discussion in Module 6."""
    label_id = ensure_label(service, label_name) if not dry_run else f"(would-create) AI/{label_name}"

    if dry_run:
        return {"applied": False, "label": f"AI/{label_name}", "removed_from_inbox": remove_from_inbox}

    body = {"addLabelIds": [label_id]}
    if remove_from_inbox:
        body["removeLabelIds"] = ["INBOX"]
    service.users().messages().modify(userId="me", id=msg_id, body=body).execute()
    return {"applied": True, "label": f"AI/{label_name}", "removed_from_inbox": remove_from_inbox}
