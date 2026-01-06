from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from rag_store import RAGChunk


def truthy_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_bot_name() -> str:
    return os.getenv("BOT_NAME", "YokosukaRAG")


def get_bot_hashtags() -> str:
    return os.getenv("BOT_HASHTAGS", "#Yokosuka #MiuraPeninsula #Kanagawa #Japan")


def get_output_style_default() -> str:
    return os.getenv("OUTPUT_STYLE_DEFAULT", "tweet_bot")


def get_max_chars_default(style: str) -> int:
    return int(os.getenv("MAX_CHARS_TWEET", "280")) if style == "tweet_bot" else int(
        os.getenv("MAX_CHARS_DEFAULT", "2000")
    )


def safe_zoneinfo(tz_name: Optional[str]) -> Optional[ZoneInfo]:
    if not tz_name:
        return None
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return None


def resolve_now_datetime(payload_datetime: Optional[str], tz_name: Optional[str]) -> tuple[datetime, str]:
    tz = safe_zoneinfo(tz_name) or ZoneInfo("Asia/Tokyo")
    if payload_datetime:
        try:
            dt = datetime.fromisoformat(payload_datetime.replace("Z", "+00:00"))
            dt = dt.astimezone(tz) if dt.tzinfo else dt.replace(tzinfo=tz)
        except Exception:
            dt = datetime.now(tz)
    else:
        dt = datetime.now(tz)
    return dt, f"{dt.strftime('%A %Y-%m-%d %H:%M')} {tz.key}"


_URL_RE = re.compile(r"https?://[^\s\)\]\}>\",']+", re.IGNORECASE)
_BARE_SCHEME_RE = re.compile(r"(?<!\w)(https?://)(?![A-Za-z0-9])", re.IGNORECASE)
_URL_TRIM_CHARS = " \t\r\n.,;:!?)\"]}'＞》〉」』】"


def normalize_url(url: str) -> str:
    return (url or "").strip().strip(_URL_TRIM_CHARS)


def extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    urls = [normalize_url(m.group(0)) for m in _URL_RE.finditer(text)]
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def collect_allowed_urls(*parts: str, extra_urls: Optional[Iterable[str]] = None) -> set[str]:
    allowed: set[str] = set()
    for p in parts:
        for u in extract_urls_from_text(p or ""):
            allowed.add(u)
    if extra_urls:
        for u in extra_urls:
            if u:
                allowed.add(normalize_url(u))
    return allowed


def filter_answer_urls(answer: str, allowed_urls: set[str]) -> tuple[str, list[str]]:
    if not answer:
        return answer, []
    removed: list[str] = []
    filtered = answer
    for u in extract_urls_from_text(answer):
        if u not in allowed_urls:
            removed.append(u)
            filtered = filtered.replace(u, "")
    filtered = re.sub(r"\s{2,}", " ", filtered).strip()
    return filtered, removed


def chunk_id(meta: dict[str, Any], text: str) -> str:
    meta = meta or {}
    for k in ("id", "chunk_id", "uid"):
        if meta.get(k):
            return str(meta[k])
    if meta.get("url"):
        return str(meta["url"])
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:10]


def select_required_context(chunks: list[RAGChunk]) -> tuple[str, str]:
    if not chunks:
        return ("", "")
    best = chunks[0]
    meta = best.metadata or {}
    url = meta.get("url") or (extract_urls_from_text(best.text)[0] if extract_urls_from_text(best.text) else "")
    title = meta.get("title") or ""
    place = meta.get("place") or ""

    mention = title.strip() or (best.text.strip().splitlines()[0][:80] if best.text else "")
    if place.strip() and place.lower() not in mention.lower():
        mention = f"{mention} ({place.strip()})".strip()
    return mention, normalize_url(url)


def project_context_for_prompt(chunks: list[RAGChunk], *, style: str, max_items: int = 8) -> str:
    items: list[str] = []
    for c in chunks[:max_items]:
        t = (c.text or "").strip()
        if not t:
            continue
        if style == "tweet_bot":
            t = re.sub(r"\s+", " ", t)[:280]
        items.append(f"- {t}")
    return "\n".join(items)


def build_chat_prompts(
    *,
    question: str,
    now_block: str,
    chunks: list[RAGChunk],
    extra_context: str,
    output_style: str,
    required_mention: str,
    required_url: str,
    allowed_urls: set[str],
    max_chars: int,
) -> tuple[str, str]:
    bot_name = get_bot_name()
    hashtags = get_bot_hashtags() if output_style == "tweet_bot" else ""

    allowed_list = "\n".join(f"- {u}" for u in sorted(allowed_urls)) if allowed_urls else "(none)"

    system_prompt = f"""You are {bot_name}.
Hard requirements:
- You MUST mention: "{required_mention}"
- You MUST include this URL exactly once: "{required_url}"
- tweet_bot => single line, <= {max_chars} chars
- Do NOT invent URLs. Use only allowed URLs.
"""

    ctx_block = project_context_for_prompt(chunks, style=output_style)

    user_prompt = f"""NOW: {now_block}

QUESTION:
{question}

EXTRA CONTEXT:
{extra_context.strip() if extra_context else "(none)"}

RAG CONTEXT:
{ctx_block}

ALLOWED URLs:
{allowed_list}

Hashtags you may use if they fit: {hashtags}
"""
    return system_prompt, user_prompt


def finalize_answer(
    *,
    raw_answer: str,
    output_style: str,
    max_chars: int,
    required_mention: str,
    required_url: str,
) -> str:
    s = (raw_answer or "").strip().strip('"').strip()
    s = _BARE_SCHEME_RE.sub("", s).replace("()", "").strip()

    if output_style == "tweet_bot":
        s = re.sub(r"\s+", " ", s.replace("\n", " ").replace("\r", " ")).strip()

    if required_mention and required_mention.lower() not in s.lower():
        s = f"{required_mention} — {s}".strip(" —")
    if required_url and required_url not in s:
        s = f"{s} {required_url}".strip()

    if output_style == "tweet_bot" and len(s) > max_chars:
        suffix = f" {required_url}".strip() if required_url and required_url in s else ""
        base = s.replace(suffix, "").strip() if suffix else s
        reserve = len(suffix) + 2
        avail = max(0, max_chars - reserve)
        trimmed = base[:avail].rstrip()
        if " " in trimmed:
            trimmed = trimmed.rsplit(" ", 1)[0].rstrip()
        s = f"{trimmed} … {suffix}".strip()[:max_chars]

    return s.strip()
