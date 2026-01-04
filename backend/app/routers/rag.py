from __future__ import annotations

import logging
import os
import re
from http import HTTPStatus
from typing import Any, Literal
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import rag_store
import requests
import hashlib
import random
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from rag_store import RAGChunk
from weather_service import *

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Use a module-level session so tests can monkeypatch it.
_session = requests.Session()



def _safe_zoneinfo(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("Asia/Tokyo")

def _extract_now_from_live_weather(live_weather: str | None) -> tuple[datetime | None, str | None]:
    """
    Try to parse Open-Meteo snapshot JSON:
      { "timezone": "Asia/Tokyo", "current": { "time": "2025-12-28T21:00", ... } }
    """
    if not live_weather or not live_weather.strip():
        return None, None
    try:
        obj = json.loads(live_weather)
        if not isinstance(obj, dict):
            return None, None
        tz_name = obj.get("timezone") or obj.get("timezone_abbreviation") or "Asia/Tokyo"
        cur = obj.get("current") or {}
        t = cur.get("time")
        if not isinstance(t, str) or not t:
            return None, tz_name
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_safe_zoneinfo(tz_name))
        else:
            dt = dt.astimezone(_safe_zoneinfo(tz_name))
        return dt, tz_name
    except Exception:
        return None, None

def _format_now_block(live_weather: str | None) -> str:
    dt, tz_name = _extract_now_from_live_weather(live_weather)
    if dt is None:
        tz_name = tz_name or (os.getenv("TZ_NAME") or "Asia/Tokyo")
        dt = datetime.now(_safe_zoneinfo(tz_name))

    today = dt.date()
    tomorrow = today + timedelta(days=1)

    # to fix "this weekend"
    # weekday(): Mon=0 ... Sun=6
    days_until_sat = (5 - dt.weekday()) % 7
    sat = today + timedelta(days=days_until_sat)
    sun = sat + timedelta(days=1)

    wd = _WEEKDAYS[dt.weekday()]
    return (
        f"- local_datetime: {dt.strftime('%Y-%m-%d %H:%M')} {dt.tzname() or ''} ({wd})\n"
        f"- timezone: {tz_name}\n"
        f"- today: {today.isoformat()}\n"
        f"- tomorrow: {tomorrow.isoformat()}\n"
        f"- this_weekend: {sat.isoformat()}–{sun.isoformat()} (Sat–Sun)\n"
    )

def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# -------------------------------------------------------------------
# Request / response models
# -------------------------------------------------------------------


class IngestRequest(BaseModel):
    documents: list[str] = Field(..., description="Raw texts to ingest into the vector store.")


class IngestResponse(BaseModel):
    ingested: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


    # caller-provided metadata (NOT stored in RAG)
    datetime: str | None = Field(
        default=None,
        description="Optional ISO datetime string provided by the caller (e.g. 2026-01-03T17:43:54+09:00).",
    )
    links: list[str] | None = Field(
        default=None,
        description="Optional list of URLs provided by the caller (for allowlisting / UI display).",
    )

    top_k: int = Field(
        5,
        ge=1,
        le=128,
        description="Number of similar chunks to retrieve from the vector store.",
    )

    extra_context: str | None = Field(
        default=None,
        description=(
            "Optional short-lived context that should NOT be stored in RAG "
            "(e.g., live weather fetched at runtime)."
        ),
    )

    # Backward-compat aliases for older clients/scripts
    context: str | None = Field(default=None, description="Alias for extra_context")
    user_context: str | None = Field(default=None, description="Alias for extra_context")

    include_debug: bool = Field(
        default=False,
        description="If true, include retrieved RAG context and chunk metadata in the response.",
    )

    output_style: Literal["default", "tweet_bot"] = Field(
        default="tweet_bot",
        description="Controls output style. 'tweet_bot' produces a single friendly tweet-like post.",
    )

    max_words: int = Field(
        default=512,
        ge=50,
        le=1024,
        description="Maximum characters for the generated post (tweet is 280).",
    )

    audit: bool | None = Field(
        default=None,
        description=(
            "Enable auditing of the generated answer by a second LLM call. "
            "If None, uses env RAG_AUDIT_DEFAULT."
        ),
    )

    audit_rewrite: bool | None = Field(
        default=None,
        description=(
            "If true, and the auditor provides a fixed_answer, replace answer with it. "
            "If None, uses env RAG_AUDIT_REWRITE_DEFAULT."
        ),
    )

    audit_model: str | None = Field(
        default=None,
        description=(
            "Optional override model name for auditing (defaults to RAG_AUDIT_MODEL / OLLAMA_CHAT_MODEL)."
        ),
    )


class ChunkOut(BaseModel):
    id: str | None = None
    text: str
    distance: float | None = None
    metadata: dict = Field(default_factory=dict)


class AuditResult(BaseModel):
    model: str | None = None
    passed: bool = Field(..., description="True if the answer is supported by the provided context.")
    score: int = Field(..., ge=0, le=100, description="0-100 quality score of support & faithfulness.")
    confidence: Literal["low", "medium", "high"] | None = None
    issues: list[str] = Field(default_factory=list)
    fixed_answer: str | None = None
    original_answer: str | None = None
    raw: str | None = None


class QueryResponse(BaseModel):
    answer: str
    links: list[str] | None = None
    context: list[str] | None = None
    chunks: list[ChunkOut] | None = None
    removed_urls: list[str] | None = None
    audit: AuditResult | None = None


class StatusResponse(BaseModel):
    docs_dir: str
    json_files: int
    chunks_in_store: int
    files: list[str]


class ReindexResponse(BaseModel):
    documents: int
    chunks: int
    files: int


# -------------------------------------------------------------------
# Ollama chat wrapper
# -------------------------------------------------------------------


def _get_ollama_chat_model() -> str:
    return os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")


def _get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")

def _get_ollama_chat_timeout() -> int:
    """
    Seconds for requests timeout to Ollama /api/chat.
    """
    raw = os.getenv("OLLAMA_CHAT_TIMEOUT", "300")
    try:
        v = int(raw)
        return v if v > 0 else 300
    except Exception:
        return 300

def _call_ollama_chat(*, question: str, system_prompt: str, user_prompt: str) -> str:
    """Call Ollama's /api/chat endpoint."""
    base_url = _get_ollama_base_url()
    model = _get_ollama_chat_model()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    try:
        timeout_s = _get_ollama_chat_timeout()
        resp = _session.post(f"{base_url}/api/chat", json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama chat response missing 'message.content'")
    except Exception as e:
        logger.exception("Ollama chat failed: %s", e)
        raise RuntimeError(f"Ollama chat failed: {e}") from e

    return content


def _truthy_env(name: str, default: str = "false") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _get_audit_default_enabled() -> bool:
    return _truthy_env("RAG_AUDIT_DEFAULT", "false")


def _get_audit_default_rewrite() -> bool:
    return _truthy_env("RAG_AUDIT_REWRITE_DEFAULT", "false")


def _get_ollama_audit_model() -> str:
    return os.getenv("RAG_AUDIT_MODEL") or _get_ollama_chat_model()


def _call_ollama_chat_with_model(*, model: str, system_prompt: str, user_prompt: str) -> str:
    """Call Ollama's /api/chat endpoint with an explicit model override."""
    base_url = _get_ollama_base_url()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    try:
        timeout_s = _get_ollama_chat_timeout()
        resp = _session.post(f"{base_url}/api/chat", json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama chat response missing 'message'")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama chat response missing 'message.content'")
        return content
    except Exception as e:
        logger.exception("Ollama audit chat failed: %s", e)
        raise RuntimeError(f"Ollama audit chat failed: {e}") from e


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _parse_first_json_object(text: str) -> dict | None:
    candidate = _strip_code_fences(text)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find the first {...} block that parses as JSON.
    for m in re.finditer(r"\{.*?\}", candidate, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _build_audit_prompts(
    *,
    question: str,
    answer: str,
    rag_context: list[str],
    live_weather: str | None,
    allowed_urls: list[str],
    output_style: str,
    max_chars: int,
) -> tuple[str, str]:
    system_prompt = (
        "You are a strict auditor for a Retrieval-Augmented Generation (RAG) assistant.\n"
        "You will be given: QUESTION, ANSWER, and CONTEXT.\n"
        "Your job: judge whether the ANSWER is fully supported by the CONTEXT.\n"
        "Rules:\n"
        "- Use ONLY the provided CONTEXT; do NOT rely on outside knowledge.\n"
        "- If a claim is not clearly supported, mark it as an issue.\n"
        "- If the answer contains URLs not listed in ALLOWED_URLS, mark it as an issue.\n"
        "- Output MUST be a single JSON object and nothing else.\n"
        "\n"
        "JSON schema (keys must exist):\n"
        "{\n"
        "  \"passed\": boolean,\n"
        "  \"score\": integer 0-100,\n"
        "  \"confidence\": \"low\"|\"medium\"|\"high\",\n"
        "  \"issues\": array of strings,\n"
        "  \"fixed_answer\": string|null\n"
        "}\n"
        "\n"
        "If passed=false, provide fixed_answer that removes unsupported claims and stays within max_chars. "
        "Keep the same output_style as the original answer."
    )

    ctx_parts: list[str] = []
    for i, c in enumerate(rag_context, start=1):
        if not isinstance(c, str):
            continue
        c = c.strip()
        if not c:
            continue
        # Cap each context block to avoid runaway tokens
        if len(c) > 1500:
            c = c[:1500] + "…"
        ctx_parts.append(f"[{i}] {c}")

    user_prompt = (
        f"OUTPUT_STYLE: {output_style}\n"
        f"MAX_CHARS: {max_chars}\n"
        f"ALLOWED_URLS: {allowed_urls}\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"CONTEXT:\n" + "\n\n".join(ctx_parts) + "\n\n"
        + (f"LIVE_WEATHER:\n{live_weather}\n\n" if live_weather else "")
    )

    return system_prompt, user_prompt


def _run_answer_audit(
    *,
    question: str,
    answer: str,
    rag_context: list[str],
    live_weather: str | None,
    allowed_urls: list[str],
    output_style: str,
    max_chars: int,
    audit_model: str,
    include_raw: bool,
) -> AuditResult:
    system_prompt, user_prompt = _build_audit_prompts(
        question=question,
        answer=answer,
        rag_context=rag_context,
        live_weather=live_weather,
        allowed_urls=allowed_urls,
        output_style=output_style,
        max_chars=max_chars,
    )

    raw = _call_ollama_chat_with_model(
        model=audit_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    obj = _parse_first_json_object(raw)
    if not isinstance(obj, dict):
        return AuditResult(
            model=audit_model,
            passed=False,
            score=0,
            confidence="low",
            issues=["Audit model did not return valid JSON."],
            fixed_answer=None,
            raw=(raw if include_raw else None),
        )

    passed = bool(obj.get("passed", False))
    score = obj.get("score")
    try:
        score_i = int(score)
    except Exception:
        score_i = 0
    score_i = max(0, min(100, score_i))

    conf = obj.get("confidence")
    if conf not in {"low", "medium", "high"}:
        conf = "low"

    issues = obj.get("issues")
    if not isinstance(issues, list):
        issues_list: list[str] = []
    else:
        issues_list = [str(x) for x in issues if str(x).strip()]

    fixed_answer = obj.get("fixed_answer")
    if fixed_answer is not None and not isinstance(fixed_answer, str):
        fixed_answer = str(fixed_answer)

    return AuditResult(
        model=audit_model,
        passed=passed,
        score=score_i,
        confidence=conf,
        issues=issues_list,
        fixed_answer=fixed_answer,
        raw=(raw if include_raw else None),
    )




def _chunk_id_from_metadata(meta: dict) -> str | None:
    doc_id = meta.get("doc_id")
    idx = meta.get("chunk_index")
    if idx is None:
        idx = meta.get("index")
    if isinstance(doc_id, str) and doc_id and idx is not None:
        return f"{doc_id}:{idx}"
    return None


# -------------------------------------------------------------------
# Tweet-bot helpers (output only; no citations/sources in the text)
# -------------------------------------------------------------------


def _get_bot_name() -> str:
    return os.getenv("WEATHER_BOT_NAME", "YokoWeather")


def _get_bot_hashtags() -> str:
    # Space-separated or comma-separated accepted; we pass through to the model.
    return os.getenv("HASHTAGS", "#Yokosuka #MiuraPeninsula #Kanagawa")


def _clean_single_line(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").replace("\t", " ").split()).strip()


def _strip_wrapping_quotes(text: str) -> str:
    s = text.strip()
    pairs = [
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
        ("`", "`"),
    ]
    for a, b in pairs:
        if s.startswith(a) and s.endswith(b) and len(s) >= 2:
            s = s[1:-1].strip()
    return s


def _enforce_max_chars(text: str, max_words: int) -> str:
    if max_words <= 0 or len(text) <= max_words:
        return text
    cut = text[: max_words - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "…"


# -------------------------------------------------------------------
# URL safety: only allow URLs that appear in retrieved context (or explicit allowlists)
# -------------------------------------------------------------------

# NOTE: the double-quote inside the character class must be escaped for Python string syntax.
_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")

_TRAILING_PUNCT_RE = re.compile(r"[\]\)\}\>,\.;:!\?\"']+$")


def _normalize_url(url: str) -> str:
    """Normalize URLs for comparison (strip common trailing punctuation)."""
    u = url.strip()
    # Remove common trailing punctuation that appears in prose.
    u = _TRAILING_PUNCT_RE.sub("", u)
    return u


def _extract_urls_from_text(text: str) -> set[str]:
    return { _normalize_url(m.group(0)) for m in _URL_RE.finditer(text or "") }


def _get_extra_allowed_urls() -> set[str]:
    raw = os.getenv("RAG_EXTRA_ALLOWED_URLS", "")
    if not raw.strip():
        return set()
    parts = re.split(r"[\n,]+", raw)
    return {_normalize_url(p) for p in parts if p.strip()}


def _get_allowlist_regexes() -> list[re.Pattern[str]]:
    raw = os.getenv("RAG_URL_ALLOWLIST_REGEXES", "")
    if not raw.strip():
        return []
    patterns: list[re.Pattern[str]] = []
    for p in re.split(r"[\n,]+", raw):
        p = p.strip()
        if not p:
            continue
        try:
            patterns.append(re.compile(p))
        except re.error:
            logger.warning("Invalid URL allowlist regex ignored: %s", p)
    return patterns


def _is_allowed_url(url: str, *, allowed_urls: set[str], allowlist_regexes: list[re.Pattern[str]]) -> bool:
    u = _normalize_url(url)
    if u in allowed_urls:
        return True
    for rx in allowlist_regexes:
        if rx.search(u):
            return True
    return False

_INCOMPLETE_SCHEME_RE = re.compile(r"https?://(?=[\s\)\]\}\>,\.;:!\?\"']|$)")
_EMPTY_PARENS_RE = re.compile(r"\(\s*\)")

def _filter_answer_urls(
    answer: str,
    *,
    allowed_urls: set[str],
    allowlist_regexes: list[re.Pattern[str]],
) -> tuple[str, list[str]]:
    """Remove/harden URLs that are not allowed.

    - Markdown links: keep the label, drop the URL if not allowed.
    - Raw URLs: remove if not allowed.
    """
    removed: list[str] = []

    def _md_repl(m: re.Match[str]) -> str:
        label = m.group(1)
        url = _normalize_url(m.group(2))
        if _is_allowed_url(url, allowed_urls=allowed_urls, allowlist_regexes=allowlist_regexes):
            return f"[{label}]({url})"
        removed.append(url)
        return label  # drop the link, keep text

    out = _MD_LINK_RE.sub(_md_repl, answer or "")

    def _raw_repl(m: re.Match[str]) -> str:
        url = _normalize_url(m.group(0))
        if _is_allowed_url(url, allowed_urls=allowed_urls, allowlist_regexes=allowlist_regexes):
            return url
        removed.append(url)
        return ""

    out = _URL_RE.sub(_raw_repl, out)
    out = _INCOMPLETE_SCHEME_RE.sub("", out)     # "https://)", "https://"
    out = _EMPTY_PARENS_RE.sub("", out)          # "( )" , "()"

    # Cleanup: collapse extra spaces created by URL removal.
    out = re.sub(r"\s{2,}", " ", out).strip()
    removed_dedup = sorted({u for u in removed if u})
    return out, removed_dedup


def _collect_allowed_urls(context_texts: list[str], live_extra: str | None) -> set[str]:
    urls: set[str] = set()
    for t in context_texts:
        urls |= _extract_urls_from_text(t)
    if live_extra:
        urls |= _extract_urls_from_text(live_extra)
    urls |= _get_extra_allowed_urls()
    return {u for u in urls if u}


def _augment_prompts_with_url_policy(
    system_prompt: str,
    user_prompt: str,
    *,
    allowed_urls: set[str],
) -> tuple[str, str]:
    policy = (
        "\nURL POLICY:\n"
        "- Do NOT invent or guess URLs.\n"
        "- Only include URLs that appear in the 'Allowed URLs' list (verbatim).\n"
        "- If the Allowed URLs list is empty, do not include any URLs.\n"
    )
    sys2 = (system_prompt or "") + policy

    # Keep the list short to avoid bloating prompts.
    max_urls = int(os.getenv("RAG_MAX_ALLOWED_URLS_IN_PROMPT", "25") or "25")
    allowed_list = sorted(allowed_urls)[: max_urls if max_urls > 0 else 25]
    if allowed_list:
        urls_block = "\nAllowed URLs (copy/paste only):\n" + "\n".join(f"- {u}" for u in allowed_list) + "\n"
    else:
        urls_block = "\nAllowed URLs (copy/paste only):\n- (none)\n"

    usr2 = (user_prompt or "") + urls_block
    return sys2, usr2

def _build_chat_prompts(
    *,
    question: str,
    rag_context: list[str],
    live_weather: str | None,
    output_style: str,
    max_words: int,
    place_hint: str | None,
) -> tuple[str, str]:
    if output_style != "tweet_bot":
        system = "You answer using the given context."
        ctx = "\n\n".join(rag_context + ([live_weather] if live_weather else []))
        user = (
            "Use the context below aside from general knowledge to answer the question.\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question:\n{question}\n"
        )
        return system, user

    bot_name = _get_bot_name()
    hashtags = _get_bot_hashtags()
    place = place_hint or os.getenv("PLACE") or "your area"

    now_block = _format_now_block(live_weather)

    system = (
        f"You are {bot_name}, a friendly English local story bot for {place} (locals, familes and tourists). "
        f"Write one tweet in English within {max_words} characters. "
        "No markdown, no lists, no extra commentary, no quotes.\n"
        "Show only real existing URLs.\n"
        "TIME AWARENESS:\n"
        "- Treat NOW (in user prompt) as the current local datetime.\n"
        "- Do NOT recommend events that are already in the past relative to NOW.\n"
        "- If you use words like 'today', 'tomorrow', or 'this weekend', they must match NOW.\n"
        "STYLE:\n"
        "- Warm, upbeat, practical.\n"
        "- Use emojis.\n"
        f"- If you add hashtags, pick 1-3 from: {hashtags}.\n"
    )

    def _sample_context(ctx: list[str], seed_text: str, k: int = 8, pool: int = 18) -> list[str]:
        if not ctx:
            return []
        cand = ctx[: min(len(ctx), pool)]
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        rng.shuffle(cand)
        return cand[: min(k, len(cand))]

    sampled = _sample_context(rag_context, question, k=8, pool=18)
    rag_lines = "\n".join(f"- {c}" for c in sampled) if sampled else "- (none)"
    live_block = live_weather.strip() if isinstance(live_weather, str) and live_weather.strip() else "(not available)"

    user = (
        "NOW (for time reasoning):\n"
        f"{now_block}\n"
        "LIVE WEATHER:\n"
        f"{live_block}\n\n"
        "RAG CONTEXT:\n"
        f"{rag_lines}\n\n"
        "TASK:\n"
        f"{question}\n\n"
        "Remember: output ONLY the tweet text."
    )

    return system, user


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    docs_dir = os.getenv("RAG_DOCS_DIR") or os.getenv("DOCS_DIR", "/data/json")
    file_paths = rag_store.list_json_files(docs_dir)
    file_names = [os.path.basename(p) for p in file_paths][:50]

    return StatusResponse(
        docs_dir=docs_dir,
        json_files=len(file_paths),
        chunks_in_store=rag_store.get_collection_count(),
        files=file_names,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_rag(request: IngestRequest) -> IngestResponse:
    """(Optional) Ingest raw texts (kept for backwards-compat / testing).

    In production, prefer indexing from JSON files via /rag/reindex or startup auto-index.
    """
    docs = [d.strip() for d in request.documents if d and d.strip()]

    if not docs:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="No documents provided.",
        )

    successes = 0
    last_error: Exception | None = None

    for text in docs:
        try:
            rag_store.add_document(text)
            successes += 1
        except Exception as exc:
            logger.exception("Failed to ingest document", exc_info=exc)
            last_error = exc

    if successes == 0 and last_error is not None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=f"Document ingestion failed: {last_error}",
        )

    return IngestResponse(ingested=successes)


@router.post("/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    """Clear and rebuild the vector DB from JSON files in DOCS_DIR."""
    docs_dir = os.getenv("RAG_DOCS_DIR") or os.getenv("DOCS_DIR", "/data/json")
    enabled = _truthy(os.getenv("RAG_REINDEX_ENABLED", "true"))
    if not enabled:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail="Reindex is disabled by configuration.",
        )

    try:
        stats = rag_store.rebuild_from_json_dir(docs_dir)
        return ReindexResponse(**stats)
    except Exception as exc:
        logger.exception("Reindex failed", exc_info=exc)
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=str(exc),
        ) from exc


def _enrich_context_text_with_links(text: str, meta: dict[str, Any]) -> tuple[str, set[str]]:
    """Attach doc-level metadata (especially links) to every chunk's context.

    Why: chunking can separate the URL/title/time/place/tag from the relevant content chunk.
    By re-attaching metadata, the LLM can always see the full document context and URL allow-listing works.
    """

    def _clean_list(val: Any, *, split_commas: bool = False) -> list[str]:
        if val is None:
            return []
        if isinstance(val, list):
            return [str(x).strip() for x in val if isinstance(x, (str, int, float)) and str(x).strip()]
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return []
            if split_commas and "," in s:
                return [p.strip() for p in s.split(",") if p.strip()]
            return [s]
        if isinstance(val, (int, float)):
            return [str(val)]
        return []

    # Collect links from metadata
    links: list[str] = []
    for k in ("links", "link", "url", "permalink", "href", "source_url", "sourceUrl"):
        links.extend(_clean_list((meta or {}).get(k), split_commas=True))
    links = [u for u in links if isinstance(u, str) and u and ("http://" in u or "https://" in u)]
    links = list(dict.fromkeys(links))

    # Collect tags (supports new `tag` and legacy `tags`)
    tags: list[str] = []
    for k in ("tag", "tags"):
        tags.extend(_clean_list((meta or {}).get(k), split_commas=True))
    tags = list(dict.fromkeys([t for t in tags if t]))

    # Build metadata block (keep it compact but complete)
    meta_lines: list[str] = []
    for key, label in (("title", "TITLE"), ("datetime", "DATETIME"), ("place", "PLACE")):
        v = (meta or {}).get(key)
        if isinstance(v, str) and v.strip():
            if not re.search(rf"^\s*{label}\s*:", text, flags=re.IGNORECASE | re.MULTILINE):
                meta_lines.append(f"{label}: {v.strip()}")

    if tags and not re.search(r"^\s*TAG\s*:", text, flags=re.IGNORECASE | re.MULTILINE):
        meta_lines.append("TAG: " + ", ".join(tags))

    if links:
        missing = [u for u in links if u not in text]
        if missing:
            meta_lines.append("LINK:")
            meta_lines.extend(missing)

    # NOTE: 全metadataをJSONとしてコンテクストへ流し込む（＝data.jsonの“コンテクスト全部”がRAGに渡る）
    meta_json_obj = dict(meta or {})
    try:
        meta_json = json.dumps(meta_json_obj, ensure_ascii=False, default=str)
    except Exception:
        meta_json = str(meta_json_obj)
    if meta_json and meta_json != "{}":
        meta_lines.append("META_JSON: " + meta_json)

    enriched = text.rstrip()
    if meta_lines:
        enriched = enriched + "\n\n----\n" + "\n".join(meta_lines)

    all_links = set(links) | set(_extract_urls_from_text(enriched))
    return enriched, all_links



@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query_rag(payload: QueryRequest, http_request: Request) -> QueryResponse:
    """Run a full RAG cycle: retrieve similar chunks and ask the chat model."""
    try:
        tweet_pool = int(os.getenv("RAG_TWEET_CONTEXT_POOL", "18"))
        raw_k = max(payload.top_k * 4, tweet_pool * 4) if payload.output_style == "tweet_bot" else payload.top_k

        chunks: list[RAGChunk] = rag_store.query_similar_chunks(
            payload.question,
            top_k=raw_k,
        )

        if payload.output_style == "tweet_bot":
            seen = set()
            diversified = []
            for c in sorted(chunks, key=lambda x: x.distance):
                meta = c.metadata if isinstance(c.metadata, dict) else {}
                key = meta.get("file") or meta.get("doc_id") or c.text[:40]
                if key in seen:
                    continue
                diversified.append(c)
                seen.add(key)
                if len(diversified) >= tweet_pool:
                    break
            chunks = diversified or chunks[:tweet_pool]
    except Exception as exc:
        logger.exception("Vector search failed", exc_info=exc)
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=f"Vector search failed: {exc}",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="No relevant context found for the given question.",
        )

    context_texts: list[str] = []
    doc_links: set[str] = set()
    for c in chunks:
        meta = c.metadata if isinstance(c.metadata, dict) else {}
        enriched, links = _enrich_context_text_with_links(c.text, meta)
        context_texts.append(enriched)
        doc_links |= links

    # Compat: accept `context` / `user_context` as aliases for `extra_context`
    extra_ctx: str | None = payload.extra_context or payload.context or payload.user_context

    def _looks_like_weather_json(s: str) -> bool:
        try:
            obj = json.loads(s)
        except Exception:
            return False
        if not isinstance(obj, dict):
            return False
        cur = obj.get("current")
        return (
            isinstance(cur, dict)
            and isinstance(cur.get("time"), str)
            and bool(cur.get("time"))
        )

    live_extra: str | None = None
    user_extra: str | None = None
    if isinstance(extra_ctx, str) and extra_ctx.strip():
        if _looks_like_weather_json(extra_ctx):
            live_extra = extra_ctx
        else:
            user_extra = extra_ctx

    if (live_extra is None or not live_extra.strip()):
        try:
            live_extra = get_live_weather_context(
                http_request=http_request,
                session=_session,
            )
        except Exception as exc:
            logger.exception("Live weather fetch failed", exc_info=exc)
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Live weather fetch failed: {exc}",
            ) from exc


    place_hint = http_request.query_params.get("place") or os.getenv("PLACE")

    # If caller provided non-weather extra context, keep it, but don't poison LIVE WEATHER.
    if user_extra:
        context_texts = context_texts + [f"EXTRA CONTEXT:\n{user_extra}"]

    # normalize request links
    req_links: list[str] = []
    for raw in (payload.links or []):
        if not isinstance(raw, str):
            continue
        u = _normalize_url(raw.strip())
        if u and (u.startswith("http://") or u.startswith("https://")):
            req_links.append(u)
        if len(req_links) >= 50:
            break

    allowed_urls = _collect_allowed_urls(context_texts, live_extra) | set(req_links)


    system_prompt, user_prompt = _build_chat_prompts(
        question=payload.question,
        rag_context=context_texts,
        live_weather=live_extra,
        output_style=payload.output_style,
        max_words=payload.max_words,
        place_hint=place_hint,
    )

    system_prompt, user_prompt = _augment_prompts_with_url_policy(
        system_prompt,
        user_prompt,
        allowed_urls=allowed_urls,
    )

    try:
        answer = _call_ollama_chat(
            question=payload.question,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.exception("Ollama chat failed", exc_info=exc)
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    answer = _strip_wrapping_quotes(_clean_single_line(answer))
    answer, removed_urls = _filter_answer_urls(
        answer,
        allowed_urls=allowed_urls,
        allowlist_regexes=_get_allowlist_regexes(),
    )
    answer = _enforce_max_chars(answer, payload.max_words)

    # Optional: audit the answer with a second LLM call.
    audit_enabled = payload.audit if payload.audit is not None else _get_audit_default_enabled()
    audit_rewrite = payload.audit_rewrite if payload.audit_rewrite is not None else _get_audit_default_rewrite()
    audit_result: AuditResult | None = None
    if audit_enabled:
        audit_model = payload.audit_model or _get_ollama_audit_model()

        # Keep audit context bounded: reuse what we already sent to the writer model.
        audit_context = context_texts[:20]

        try:

            # keep auditor seeing the same request metadata (optional)
            audit_question = payload.question
            if payload.datetime:
                audit_question = f"REQUEST_DATETIME: {payload.datetime}\n" + audit_question
            if req_links:
                audit_question = "REQUEST_LINKS:\n" + "\n".join(f"- {u}" for u in req_links[:10])

            audit_result = _run_answer_audit(
                question=payload.question,
                answer=answer,
                rag_context=audit_context,
                live_weather=live_extra,
                allowed_urls=allowed_urls,
                output_style=payload.output_style,
                max_chars=payload.max_words,
                audit_model=audit_model,
                include_raw=bool(payload.include_debug),
            )
        except Exception as exc:
            # Don't fail the main request if auditing fails; surface the failure via audit_result.
            audit_result = AuditResult(
                model=audit_model,
                passed=False,
                score=0,
                confidence="low",
                issues=[f"Audit failed: {exc}"],
                fixed_answer=None,
                raw=(str(exc) if payload.include_debug else None),
            )

        if audit_rewrite and audit_result.fixed_answer:
            audit_result.original_answer = answer
            answer = _strip_wrapping_quotes(_clean_single_line(audit_result.fixed_answer))
            answer, removed_urls_2 = _filter_answer_urls(
                answer,
                allowed_urls=allowed_urls,
                allowlist_regexes=_get_allowlist_regexes(),
            )
            if removed_urls_2:
                removed_urls = (removed_urls or []) + removed_urls_2
            answer = _enforce_max_chars(answer, payload.max_words)

    debug_context: list[str] | None = None
    debug_chunks: list[ChunkOut] | None = None

    if payload.include_debug:
        debug_context = list(context_texts)
        if live_extra and live_extra.strip():
            debug_context.append(f"[Live context]\n{live_extra.strip()}")

        chunk_out: list[ChunkOut] = []
        for c in chunks:
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            chunk_out.append(
                ChunkOut(
                    id=_chunk_id_from_metadata(meta),
                    text=c.text,
                    distance=c.distance,
                    metadata=meta,
                )
            )
        debug_chunks = chunk_out

    # links for UI/debug: request links first, then doc_links
    links_out: list[str] = []
    seen = set()
    for u in req_links:
        if u not in seen:
            links_out.append(u)
            seen.add(u)
    for u in (sorted(doc_links) if doc_links else []):
        if u not in seen:
            links_out.append(u)
            seen.add(u)

    return QueryResponse(
        answer=answer,
        links=(links_out if links_out else None),
        context=debug_context,
        chunks=debug_chunks,
        removed_urls=(removed_urls if payload.include_debug and removed_urls else None),
        audit=audit_result,
    )