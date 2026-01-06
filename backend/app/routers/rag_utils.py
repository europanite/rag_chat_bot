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


def collect_allowed_urls(*, user_links: list[str], rag_chunks: list[RAGChunk], extra_text: Optional[str]) -> set[str]:
    allowed = set()
    for u in user_links or []:
        u2 = normalize_url(u)
        if u2:
            allowed.add(u2)

    # from retrieved chunks metadata
    for c in rag_chunks or []:
        md = c.metadata or {}
        for k in ("url", "source", "link"):
            if md.get(k):
                allowed.add(normalize_url(str(md.get(k))))

    # from extra_text
    for u in extract_urls_from_text(extra_text or ""):
        allowed.add(u)

    # remove weird bare schemes
    allowed = {u for u in allowed if u and not _BARE_SCHEME_RE.search(u)}
    return allowed


def filter_answer_urls(answer: str, allowed_urls: set[str]) -> tuple[str, list[str]]:
    if not answer:
        return answer, []
    allowed_norm = {normalize_url(u) for u in allowed_urls if u}
    removed: list[str] = []

    def repl(m: re.Match) -> str:
        u = normalize_url(m.group(0))
        if u and u not in allowed_norm:
            removed.append(u)
            return ""
        return m.group(0)

    cleaned = _URL_RE.sub(repl, answer)
    cleaned = re.sub(r"\s+\n", "\n", cleaned).strip()
    return cleaned, removed


def _chunk_best_url(chunks: list[RAGChunk], allowed_urls: set[str]) -> Optional[str]:
    if not chunks:
        return None
    allowed_norm = {normalize_url(u) for u in allowed_urls if u}
    for c in chunks:
        md = c.metadata or {}
        u = md.get("url") or md.get("source") or md.get("link")
        if u:
            u2 = normalize_url(str(u))
            if u2 and u2 in allowed_norm:
                return u2
    return None


def _pick_required_mention(chunks: list[RAGChunk]) -> str:
    if not chunks:
        return "RAG"
    c0 = chunks[0]
    md = c0.metadata or {}
    title = md.get("title") or md.get("name") or md.get("spot") or md.get("event")
    if title:
        return str(title)
    # fallback: hash prefix from text
    t = (c0.text or "").strip()
    if not t:
        return "RAG"
    h = hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"RAG:{h}"


def select_required_context(chunks: list[RAGChunk], allowed_urls: set[str]) -> tuple[str, str]:
    mention = _pick_required_mention(chunks)
    url = _chunk_best_url(chunks, allowed_urls)
    if not url:
        # fallback: any allowed url
        url = next(iter(allowed_urls), "https://example.com")
    return mention, url


def _join_context(context_texts: Iterable[str], limit_chars: int = 6000) -> str:
    out: list[str] = []
    used = 0
    for t in context_texts:
        s = (t or "").strip()
        if not s:
            continue
        if used + len(s) + 2 > limit_chars:
            break
        out.append(s)
        used += len(s) + 2
    return "\n\n".join(out)


def build_chat_prompts(
    *,
    question: str,
    now_block: str,
    context_texts: list[str],
    live_extra: Optional[str],
    required_mention: str,
    required_url: str,
    output_style: str,
    max_chars: int,
    audit_feedback: Optional[str] = None,
    previous_answer: Optional[str] = None,
) -> tuple[str, str]:
    sys_lines = [
        "You are a helpful assistant.",
        "You MUST follow the requirements exactly.",
        "If you are unsure, be conservative and avoid hallucinations.",
        f"Output style: {output_style}",
    ]

    user_lines = [
        now_block.strip(),
        "",
        f"[QUESTION]\n{question}\n[/QUESTION]",
        "",
        f"[RAG_CONTEXT]\n{_join_context(context_texts)}\n[/RAG_CONTEXT]",
    ]

    if live_extra:
        user_lines += ["", f"[LIVE_CONTEXT]\n{live_extra.strip()}\n[/LIVE_CONTEXT]"]

    # strict requirements
    user_lines += [
        "",
        "[REQUIREMENTS]",
        f"- You must mention: {required_mention}",
        f"- You must include this URL exactly once: {required_url}",
        f"- Keep within {max_chars} characters (hard limit).",
        "- Do not include any other URLs.",
        "[/REQUIREMENTS]",
    ]

    if previous_answer:
        user_lines += ["", f"[PREVIOUS_ANSWER]\n{previous_answer}\n[/PREVIOUS_ANSWER]"]

    if audit_feedback:
        user_lines += ["", f"[AUDIT_FEEDBACK]\n{audit_feedback}\n[/AUDIT_FEEDBACK]"]

    user_lines += ["", "Answer now."]

    return "\n".join(sys_lines).strip(), "\n".join(user_lines).strip()


def _ensure_required(answer: str, required_mention: str, required_url: str) -> str:
    a = (answer or "").strip()
    if required_mention and required_mention not in a:
        # append a short mention
        a = (a + f"\n\n({required_mention})").strip()
    if required_url and required_url not in a:
        a = (a + f"\n{required_url}").strip()
    return a


def _trim_to_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    s = text or ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def finalize_answer(
    *,
    raw_answer: str,
    output_style: str,
    max_chars: int,
    required_mention: str,
    required_url: str,
) -> str:
    a = (raw_answer or "").strip()

    # normalize whitespace a bit
    a = re.sub(r"[ \t]+\n", "\n", a)
    a = re.sub(r"\n{3,}", "\n\n", a).strip()

    a = _ensure_required(a, required_mention, required_url)

    if output_style == "tweet_bot":
        a = _trim_to_chars(a, max_chars)
    else:
        # keep max_chars for safety
        a = _trim_to_chars(a, max_chars)

    return a.strip()
