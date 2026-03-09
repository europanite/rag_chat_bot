#!/usr/bin/env python3
"""Post the latest generated feed to X (Twitter).

Reads JSON from LATEST_PATH (default: frontend/app/public/latest.json) and posts
a tweet to X API v2 using OAuth 1.0a user context.

Required env vars:
  - X_API_KEY
  - X_API_SECRET
  - X_ACCESS_TOKEN
  - X_ACCESS_TOKEN_SECRET

Optional env vars:
  - LATEST_PATH (default: frontend/app/public/latest.json)
  - SITE_URL (default: https://<owner>.github.io)
  - BASE_PATH (default: <repo> for project pages, empty for <owner>.github.io)
  - HASHTAGS (space-separated hashtags; used as fallback if latest.json text has none)
  - DRY_RUN=1  (prints the tweet without posting)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

API_URL = "https://api.x.com/2/tweets"
MEDIA_UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# X counts any URL (https/http) as a fixed t.co length (usually 23).
TCO_URL_LEN = int(os.environ.get("TCO_URL_LEN", "23"))


def x_weighted_length(s: str) -> int:
    """Approximate X tweet length by counting any URL as TCO_URL_LEN characters."""
    if not s:
        return 0
    return len(URL_RE.sub(lambda m: "x" * TCO_URL_LEN, s))


def pct(s: str) -> str:
    return urllib.parse.quote(s, safe="~")


def strip_urls(s: str) -> str:
    s = URL_RE.sub("", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def truncate_utf8(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    # Keep it simple: truncate by characters (Python uses Unicode codepoints)
    return s[: max_chars - 1].rstrip() + "…"


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def clean_link_list(val: object) -> List[str]:
    if isinstance(val, str):
        vals = [val]
    elif isinstance(val, list):
        vals = [x for x in val if isinstance(x, str)]
    else:
        vals = []

    out: List[str] = []
    seen = set()
    for raw in vals:
        u = raw.strip()
        if not u or not u.startswith(("http://", "https://")):
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def parse_hashtags(s: str) -> List[str]:
    tags = [t.strip() for t in (s or "").split() if t.strip().startswith("#")]
    out: List[str] = []
    seen = set()
    for t in tags:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def pick_hashtags(tags: Sequence[str], *, k: int = 3, seed: str = "") -> str:
    uniq = [t for t in tags if t]
    if not uniq:
        return ""
    rng = random.Random(seed)
    if len(uniq) <= k:
        return " ".join(uniq)
    return " ".join(rng.sample(uniq, k))


def append_if_fit(text: str, extra: str, *, max_chars: int) -> str:
    if not extra:
        return text
    candidate = f"{text} {extra}".strip()
    return candidate if x_weighted_length(candidate) <= max_chars else text


def dedupe_keep_order(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def compose_tweet(*, raw_text: str, links: Sequence[str], share_url: str, hashtags_env: str, seed: str) -> str:
    base_text = normalize_spaces(raw_text)
    core = strip_urls(base_text) or "GOODDAY YOKOSUKA"

    if "#" not in core:
        picked = pick_hashtags(parse_hashtags(hashtags_env), k=3, seed=seed)
        core = append_if_fit(core, picked, max_chars=280)

    candidate_urls = dedupe_keep_order([*clean_link_list(list(links)), share_url])

    selected_urls: List[str] = []
    reserve = 0
    for url in candidate_urls:
        add_cost = (1 if core or selected_urls else 0) + TCO_URL_LEN
        if reserve + add_cost > 280:
            break
        reserve += add_cost
        selected_urls.append(url)

    safety = int(os.environ.get("X_TWEET_SAFETY", "0"))
    max_core = max(0, 280 - reserve - safety)
    if x_weighted_length(core) > max_core:
        core2 = truncate_utf8(core, max_core)
        while max_core > 0 and x_weighted_length(core2) > max_core:
            max_core -= 1
            core2 = truncate_utf8(core, max_core)
        core = core2

    parts: List[str] = []
    if core:
        parts.append(core)
    parts.extend(selected_urls)
    tweet = "\n".join(parts).strip()

    if x_weighted_length(tweet) > 280:
        return share_url
    return tweet


def compute_site_and_base() -> Tuple[str, str]:
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "").strip()
    repo_full = os.environ.get("GITHUB_REPOSITORY", "").strip()  # owner/repo
    repo = repo_full.split("/", 1)[1] if "/" in repo_full else os.environ.get("GITHUB_EVENT_REPOSITORY_NAME", "").strip()

    site = (os.environ.get("SITE_URL") or "").strip().rstrip("/")
    base = (os.environ.get("BASE_PATH") or "").strip().strip("/")

    if not site and owner:
        site = f"https://{owner}.github.io"
    if base == "" and owner and repo:
        # Default base path: project pages -> /repo, user/organization pages -> ""
        if repo.lower() == f"{owner.lower()}.github.io":
            base = ""
        else:
            base = repo

    return site, base


def make_share_url(site: str, base: str, post_id: str) -> str:
    base_part = f"/{base}" if base else ""
    if post_id:
        # Keep the URL clickable even if post_id contains ":" or other reserved chars.
        post_id_q = urllib.parse.quote(post_id, safe="")
        return f"{site}{base_part}/p/{post_id_q}/"
    return f"{site}{base_part}/"


def resolve_local_image_path(latest_path: str, obj: Dict[str, object]) -> Optional[Path]:
    latest_file = Path(latest_path).resolve()
    public_dir = latest_file.parent
    candidates = [
        str(obj.get("image") or "").strip(),
        str(obj.get("image_url") or "").strip(),
        str(obj.get("fixed_image") or "").strip(),
        str(obj.get("avatar_image") or "").strip(),
    ]

    for raw in candidates:
        if not raw:
            continue
        if raw.startswith(("http://", "https://")):
            continue
        rel = raw.lstrip("/")
        path = (public_dir / rel).resolve()
        if path.is_file():
            return path
    return None


def build_multipart_form(field_name: str, filename: str, content_type: str, data: bytes) -> Tuple[bytes, str]:
    boundary = f"----ragchatbot{uuid.uuid4().hex}"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(data)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), boundary


def upload_media(
    image_path: Path,
    *,
    consumer_key: str,
    consumer_secret: str,
    token: str,
    token_secret: str,
) -> Optional[str]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    body, boundary = build_multipart_form("media", image_path.name, mime_type, image_path.read_bytes())
    auth = oauth1_header("POST", MEDIA_UPLOAD_URL, consumer_key, consumer_secret, token, token_secret)
    req = urllib.request.Request(
        MEDIA_UPLOAD_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": auth,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
            "User-Agent": "rag_chat_bot-gha",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        print(f"WARN: media upload failed: HTTP {e.code}: {body}", file=sys.stderr)
        return None

    media_id = str(payload.get("media_id_string") or payload.get("media_id") or "").strip()
    return media_id or None


def oauth1_header(
    method: str,
    url: str,
    consumer_key: str,
    consumer_secret: str,
    token: str,
    token_secret: str,
) -> str:
    nonce = base64.b64encode(os.urandom(16)).decode("ascii").rstrip("=")
    timestamp = str(int(time.time()))

    oauth_params: Dict[str, str] = {
        "oauth_consumer_key": consumer_key,
        "oauth_nonce": nonce,
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": timestamp,
        "oauth_token": token,
        "oauth_version": "1.0",
    }

    # Signature base string
    # For JSON body requests, request parameters are empty; only OAuth params are included.
    param_items = [(pct(k), pct(v)) for k, v in oauth_params.items()]
    param_items.sort()
    param_str = "&".join([f"{k}={v}" for k, v in param_items])

    base_elems = [method.upper(), pct(url), pct(param_str)]
    base_str = "&".join(base_elems)

    signing_key = f"{pct(consumer_secret)}&{pct(token_secret)}"
    signature = base64.b64encode(hmac.new(signing_key.encode("utf-8"), base_str.encode("utf-8"), hashlib.sha1).digest()).decode("ascii")
    oauth_params["oauth_signature"] = signature

    # Authorization header
    header_params = ", ".join([f'{pct(k)}="{pct(v)}"' for k, v in sorted(oauth_params.items())])
    return f"OAuth {header_params}"


def main() -> int:
    api_key = os.environ.get("X_API_KEY", "")
    api_secret = os.environ.get("X_API_SECRET", "")
    access_token = os.environ.get("X_ACCESS_TOKEN", "")
    access_secret = os.environ.get("X_ACCESS_TOKEN_SECRET", "")

    missing = [k for k, v in [
        ("X_API_KEY", api_key),
        ("X_API_SECRET", api_secret),
        ("X_ACCESS_TOKEN", access_token),
        ("X_ACCESS_TOKEN_SECRET", access_secret),
    ] if not v]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

    latest_path = os.environ.get("LATEST_PATH", "frontend/app/public/latest.json")
    with open(latest_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    post_id = str(obj.get("id") or obj.get("focus_id") or obj.get("doc_id") or "").strip()
    raw_text = str(obj.get("text") or obj.get("title") or obj.get("summary") or "").strip()
    links = clean_link_list(obj.get("links") or [])
    hashtags_env = os.environ.get("HASHTAGS", "")

    site, base = compute_site_and_base()
    if not site:
        raise SystemExit("Could not determine SITE_URL. Set SITE_URL env var or provide GITHUB_REPOSITORY_OWNER.")

    share_url = make_share_url(site, base, post_id)

    tweet = compose_tweet(
        raw_text=raw_text,
        links=links,
        share_url=share_url,
        hashtags_env=hashtags_env,
        seed=post_id or raw_text or share_url,
    )

    if os.environ.get("DRY_RUN", "").strip() == "1":
        print(tweet)
        return 0

    image_path = resolve_local_image_path(latest_path, obj)
    media_id = None
    if image_path is not None:
        media_id = upload_media(
            image_path,
            consumer_key=api_key,
            consumer_secret=api_secret,
            token=access_token,
            token_secret=access_secret,
        )

    payload_obj = {"text": tweet}
    if media_id:
        payload_obj["media"] = {"media_ids": [media_id]}

    payload = json.dumps(payload_obj).encode("utf-8")
    auth = oauth1_header("POST", API_URL, api_key, api_secret, access_token, access_secret)

    req = urllib.request.Request(
        API_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": auth,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "rag_chat_bot-gha",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status not in (200, 201):
                raise SystemExit(f"X API error: HTTP {resp.status}: {body}")
            print(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise SystemExit(f"X API HTTPError: {e.code}: {body}") from e

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
