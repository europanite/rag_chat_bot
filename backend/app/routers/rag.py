"""
FastAPI router for Retrieval-Augmented Generation (RAG).

This router provides:
- /rag/status   : health/status for RAG store (expected by scripts)
- /rag/reindex  : rebuild index from JSON docs directory (expected by scripts)
- /rag/ingest   : ingest ad-hoc documents
- /rag/query    : answer a question using retrieved context + optional live context

The design here matches the test helpers and scripts in this repository:
- Uses rag_store.query_similar_chunks (not rag_store.query).
- Uses rag_store.add_document for ingestion.

Note: The actual LLM calls are to an Ollama server at {OLLAMA_BASE_URL}/api/chat.
"""
from __future__ import annotations

import os
import json
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

import rag_store
from .rag_utils import (
    build_chat_prompts,
    collect_source_links,
    collect_allowed_urls,
    filter_answer_urls,
    finalize_answer,
    select_required_context,
)
from .rag_audit import AuditLite, run_answer_audit

OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL")
RAG_MODEL = os.getenv("RAG_MODEL")
AUDIT_MODEL = os.getenv("AUDIT_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "256"))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])

# Reused HTTP session for Ollama calls (tests monkeypatch this).
_session = requests.Session()

def _ollama_chat_payload(*, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "num_predict": 256,
            "temperature": 0.2,
        },
    }


def _call_ollama_chat_with_model(*, model: str, system_prompt: str, user_prompt: str) -> str:
    """Direct Ollama call for a specific model (tests may monkeypatch this)."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = _ollama_chat_payload(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
    resp = _session.post(url, json=payload, timeout=(5, OLLAMA_TIMEOUT_S))
    resp.raise_for_status()
    data = resp.json() or {}
    return ((data.get("message") or {}).get("content") or "").strip()


def _call_ollama_chat(
    *,
    question: str,
    context: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> str:
    """
    Flexible helper used by both scripts and routes.

    - If system_prompt/user_prompt are provided, those are sent to Ollama.
    - Else, 'context' is used to build a minimal prompt.
    """
    if system_prompt is None or user_prompt is None:
        system_prompt = (
            "You are a careful assistant. Use ONLY the provided context. "
            "If you don't know, say you don't know."
        )
        user_prompt = f"Question:\n{question}\n\nContext:\n{context or ''}\n\nAnswer:"
    return _call_ollama_chat_with_model(
        model=RAG_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


DOCS_DIR = (os.getenv("DOCS_DIR") or "/data/json").strip()

def _now_block(req: Request, payload_datetime: Optional[str]) -> str:
    """
    Provide a consistent 'NOW' header for both the generator and audit.
    Scripts may pass query params: place, lat, lon, tz.
    """
    qp = req.query_params
    place = qp.get("place")
    lat = qp.get("lat")
    lon = qp.get("lon")
    tz = qp.get("tz")

    parts = []
    if payload_datetime:
        parts.append(f"datetime: {payload_datetime}")
    if tz:
        parts.append(f"tz: {tz}")
    if place:
        parts.append(f"place: {place}")
    if lat and lon:
        parts.append(f"lat,lon: {lat},{lon}")

    if not parts:
        return "NOW: (not provided)"
    return "NOW:\n" + "\n".join(parts)


# -----------------------------
# Temporal helpers (event hygiene)
# -----------------------------

_TOPIC_FAMILY_RE = re.compile(r"TOPIC\s*FAMILY:\s*([^\n]+)", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_MONTH_DAY_RE = re.compile(
    r"\b("
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _safe_parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Accept Z as UTC.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _safe_parse_date_like(value: Any, *, now_dt: Optional[datetime] = None) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s:
        return None

    # Date only
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return date.fromisoformat(s)
        except Exception:
            return None

    # Datetime-like
    if s.endswith("Z"):
        s2 = s[:-1] + "+00:00"
    else:
        s2 = s
    try:
        return datetime.fromisoformat(s2).date()
    except Exception:
        return None


def _extract_topic_family(question: str) -> Optional[str]:
    q = (question or "")
    m = _TOPIC_FAMILY_RE.search(q)
    if not m:
        return None
    raw = (m.group(1) or "").strip()
    if not raw:
        return None
    # e.g. "event (event/place/chat)." -> "event"
    token = re.split(r"[\s\(]", raw, maxsplit=1)[0].strip().lower()
    return token or None


def _wants_future_events(question: str) -> bool:
    q = (question or "").lower()
    # Our generator prompt includes "upcoming events" for event family.
    if "upcoming" in q:
        return True
    if "do not mention past" in q:
        return True
    if "future event" in q:
        return True
    return False


def _chunk_event_date(chunk: Any, *, now_dt: Optional[datetime]) -> Optional[date]:
    meta = getattr(chunk, "metadata", None) or {}
    if not isinstance(meta, dict):
        return None
    for key in ("datetime", "date", "event_datetime", "event_date", "start_datetime", "start_date"):
        if key in meta:
            d = _safe_parse_date_like(meta.get(key), now_dt=now_dt)
            if d:
                return d
    return None


def _postprocess_retrieved_chunks(
    chunks: List[Any],
    *,
    question: str,
    now_dt: Optional[datetime],
) -> List[Any]:
    """Post-process retrieved chunks to avoid selecting past events for 'upcoming' prompts.

    This is intentionally conservative: it only activates when the prompt explicitly
    requests upcoming events and declares TOPIC FAMILY: event.
    """
    family = _extract_topic_family(question)
    if family != "event":
        return chunks
    if not _wants_future_events(question):
        return chunks
    if not now_dt:
        return chunks

    today = now_dt.date()
    try:
        horizon_days = int(os.getenv("EVENT_HORIZON_DAYS", "0") or "0")
    except Exception:
        horizon_days = 0
    horizon_date = today + timedelta(days=horizon_days) if horizon_days > 0 else None

    ranked: list[tuple[int, float, Any]] = []
    for c in chunks:
        dist = getattr(c, "distance", 0.0) or 0.0
        try:
            dist_f = float(dist)
        except Exception:
            dist_f = 0.0

        d = _chunk_event_date(c, now_dt=now_dt)
        if d is None:
            ranked.append((1, dist_f, c))  # unknown date
            continue
        if d < today:
            continue  # drop past
        if horizon_date and d > horizon_date:
            ranked.append((2, dist_f, c))  # too far; de-prioritize
            continue
        ranked.append((0, dist_f, c))  # upcoming/today

    # If everything got filtered out, fall back to original to avoid empty context.
    if not ranked:
        return chunks

    ranked.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in ranked]


def _temporal_issues_future_event_answer(answer: str, *, now_dt: datetime) -> List[str]:
    """Detect obvious 'past event' mentions in the generated answer."""
    issues: list[str] = []
    text = (answer or "")
    if not text.strip():
        return issues

    today = now_dt.date()

    # 1) ISO dates: 2026-01-01
    for iso in _ISO_DATE_RE.findall(text):
        try:
            d = date.fromisoformat(iso)
        except Exception:
            continue
        if d < today:
            issues.append(f"mentions past date: {iso}")

    # 2) Month day: Jan 1st
    for mon, day_s in _MONTH_DAY_RE.findall(text):
        key = mon.strip().lower().rstrip(".")
        month_num = _MONTHS.get(key)
        if not month_num:
            continue
        try:
            day_num = int(day_s)
        except Exception:
            continue
        year = now_dt.year
        # handle year wrap for Dec -> Jan if needed
        if month_num < now_dt.month and (now_dt.month - month_num) >= 6:
            year += 1
        try:
            d = date(year, month_num, day_num)
        except Exception:
            continue
        if d < today:
            issues.append(f"mentions past date: {mon} {day_num}")

    # 3) Numeric dates: 1/1 or 01/01/2026
    for m_s, d_s, y_s in _SLASH_DATE_RE.findall(text):
        try:
            mm = int(m_s)
            dd = int(d_s)
            yy = int(y_s) if y_s else now_dt.year
            if yy < 100:
                yy += 2000
            # if year missing, assume upcoming if it looks like a wrap
            if not y_s and mm < now_dt.month and (now_dt.month - mm) >= 6:
                yy = now_dt.year + 1
            d = date(yy, mm, dd)
        except Exception:
            continue
        if d < today:
            issues.append(f"mentions past date: {m_s}/{d_s}{('/'+y_s) if y_s else ''}")

    return issues

class IngestRequest(BaseModel):
    documents: List[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    ingested: int


class AuditResult(BaseModel):
    model: str
    passed: bool
    score: int = Field(default=0, ge=0, le=100)
    confidence: str = "low"
    issues: List[str] = Field(default_factory=list)
    fixed_answer: Optional[str] = None
    original_answer: Optional[str] = None
    raw: Optional[str] = None


class QueryRequest(BaseModel):
    question: str

    # scripts/generate_talk.py sends these
    datetime: Optional[str] = None
    links: List[str] = Field(default_factory=list)
    top_k: int = 5
    extra_context: Optional[str] = None

    output_style: str = "tweet_bot"
    max_chars: Optional[int] = None

    include_debug: bool = False

    # audit
    audit: bool = False
    audit_model: Optional[str] = None
    audit_rewrite: bool = False
    audit_max_attempts: int = 1

    # if True, audit expects no unsupported claims
    strict_context: bool = True


class QueryResponse(BaseModel):
    answer: str
    context: List[str]
    links: List[str] = Field(default_factory=list)
    removed_urls: List[str] = Field(default_factory=list)
    audit: Optional[AuditResult] = None
    debug: Optional[Dict[str, Any]] = None


@router.get("/status")
def status() -> Dict[str, Any]:
    """
    Simple status endpoint expected by scripts.
    """
    try:
        n = rag_store.get_collection_count()
    except Exception as e:
        logger.exception("get_collection_count failed")
        raise HTTPException(status_code=500, detail=f"RAG store error: {e}")
    return {"ok": True, "chunks_in_store": n, "docs_dir": DOCS_DIR}


@router.post("/reindex")
def reindex() -> Dict[str, Any]:
    """
    Rebuild the index from docs dir JSON files (expected by scripts).
    """
    docs = DOCS_DIR
    if not os.path.isdir(docs):
        raise HTTPException(status_code=404, detail=f"Docs dir not found: {docs}")
    try:
        ingested = rag_store.rebuild_from_json_dir(docs)
        n = rag_store.get_collection_count()
        return {"ok": True, "ingested": ingested, "chunks_in_store": n, "docs_dir": docs}
    except Exception as e:
        logger.exception("reindex failed")
        raise HTTPException(status_code=502, detail=f"reindex failed: {e}")


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    docs = [d for d in (req.documents or []) if isinstance(d, str) and d.strip()]
    if not docs:
        raise HTTPException(status_code=400, detail="No documents provided.")
    ok = 0
    errors: List[str] = []
    for d in docs:
        try:
            rag_store.add_document(d)
            ok += 1
        except Exception as e:
            errors.append(str(e))
    if ok == 0:
        raise HTTPException(status_code=502, detail=f"Ingest failed: {errors[:3]}")
    return IngestResponse(ingested=ok)


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        top_k = int(payload.top_k or 5)
        top_k = max(1, min(20, top_k))
    except Exception:
        top_k = 5

    now_dt = _safe_parse_datetime(payload.datetime)
    topic_family = _extract_topic_family(question)
    wants_future_events = (topic_family == "event") and _wants_future_events(question)

    # Retrieve context
    try:
        chunks = rag_store.query_similar_chunks(question, top_k=top_k)
        chunks = _postprocess_retrieved_chunks(
            chunks,
            question=question,
            now_dt=now_dt,
        )
    except Exception as e:
        logger.exception("query_similar_chunks failed")
        raise HTTPException(status_code=502, detail=f"RAG query failed: {e}")

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant context.")

    context_texts = [(getattr(c, "text", "") or "").strip() for c in chunks]
    context_texts = [t for t in context_texts if t]

    # 1) links from data.json metadata (preferred for UI)
    source_links = collect_source_links(chunks=chunks, limit=64)

    # allow-list URLs
    allowed_urls: Set[str] = collect_allowed_urls(
        user_links=payload.links,
        chunk_links=source_links,
        context_texts=context_texts,
        extra_text=payload.extra_context,
    )

    required_mention, required_url = select_required_context(
        chunks=chunks,
        allowed_urls=allowed_urls,
    )

    now_block = _now_block(request, payload.datetime)
    max_chars = int(payload.max_chars)

    sys_prompt, user_prompt = build_chat_prompts(
        question=question,
        now_block=now_block,
        context_texts=context_texts,
        extra_context=payload.extra_context,
        required_mention=required_mention,
        required_url=required_url,
        allowed_urls=allowed_urls,
        output_style=payload.output_style or "tweet_bot",
        max_chars=max_chars,
    )

    # Generation (+ optional audit loop)
    removed_urls_total: List[str] = []
    original_answer: Optional[str] = None
    last_audit: Optional[AuditResult] = None

    attempts = max(1, int(payload.audit_max_attempts or 1)) if payload.audit else 1
    # For 'upcoming event' prompts, allow at least one extra internal retry when rewrite is enabled.
    if payload.audit and wants_future_events and payload.audit_rewrite:
        attempts = max(attempts, 2)

    answer = ""
    for attempt in range(1, attempts + 1):
        try:
            candidate = _call_ollama_chat(
                question=question,
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
            )
        except Exception as e:
            logger.exception("ollama chat failed")
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

        if original_answer is None:
            original_answer = candidate

        candidate = finalize_answer(
            answer=candidate,
            required_mention=required_mention,
            required_url=required_url,
            max_chars=max_chars,
            output_style=payload.output_style or "tweet_bot",
        )

        keep_allowed = (payload.output_style or "tweet_bot") != "tweet_bot"
        candidate, removed = filter_answer_urls(
            candidate,
            allowed_urls,
            keep_allowed=keep_allowed,
        )

        removed_urls_total.extend(removed)

        candidate = finalize_answer(
            answer=candidate,
            required_mention=required_mention,
            required_url=required_url,
            max_chars=max_chars,
            output_style=payload.output_style or "tweet_bot",
        )


        # Deterministic temporal lint for "upcoming event" prompts (catches obvious past-date mentions).
        if wants_future_events and now_dt:
            temporal_issues = _temporal_issues_future_event_answer(candidate, now_dt=now_dt)
            if temporal_issues:
                last_audit = AuditResult(
                    model="temporal_lint",
                    passed=False,
                    score=0,
                    confidence="high",
                    issues=temporal_issues,
                    fixed_answer=None,
                    original_answer=original_answer if payload.audit_rewrite else None,
                    raw="temporal_lint_failed" if payload.include_debug else None,
                )
                answer = candidate
                if attempt < attempts and payload.audit_rewrite:
                    feedback = "; ".join(temporal_issues[:5])
                    user_prompt = (
                        user_prompt
                        + "\n\n"
                        + f"Audit feedback: {feedback}\n"
                        + "Rewrite to comply with rules. If mentioning a date, it must be today or in the future.\n"
                        + "Do not call past events 'upcoming'. Do not add new facts.\n"
                    )
                    continue
                break

        # If no audit requested, accept immediately
        if not payload.audit:
            answer = candidate
            break

        audit_model = AUDIT_MODEL
        audit_lite: AuditLite = run_answer_audit(
            call_chat_with_model=lambda m, sp, up: _call_ollama_chat_with_model(
                model=m, system_prompt=sp, user_prompt=up
            ),
            model=audit_model,
            answer=candidate,
            question=question,
            now_block=now_block,
            allowed_urls=allowed_urls,
            required_url=required_url,
            strict_context=bool(payload.strict_context),
            allow_rewrite=bool(payload.audit_rewrite),
            max_chars=max_chars,
            require_required_url_in_answer=((payload.output_style or "tweet_bot") != "tweet_bot"),
            forbid_urls_in_answer=((payload.output_style or "tweet_bot") == "tweet_bot"),
        )

        last_audit = AuditResult(
            model=audit_model,
            passed=audit_lite.passed,
            score=audit_lite.score,
            confidence=audit_lite.confidence,
            issues=audit_lite.issues,
            fixed_answer=audit_lite.fixed_answer,
            original_answer=original_answer if payload.audit_rewrite else None,
            raw=audit_lite.raw if payload.include_debug else None,
        )

        if audit_lite.passed:
            answer = candidate
            break

        # If audit suggests a fixed answer, apply it once and re-audit (without rewrite).
        if audit_lite.fixed_answer:
            fixed = finalize_answer(
                answer=audit_lite.fixed_answer,
                required_mention=required_mention,
                required_url=required_url,
                max_chars=max_chars,
                output_style=payload.output_style or "tweet_bot",
            )
                                  
            keep_allowed = (payload.output_style or "tweet_bot") != "tweet_bot"
            fixed, removed2 = filter_answer_urls(
                fixed,
                allowed_urls,
                keep_allowed=keep_allowed,
            )

            removed_urls_total.extend(removed2)
            fixed = finalize_answer(
                answer=fixed,
                required_mention=required_mention,
                required_url=required_url,
                max_chars=max_chars,
                output_style=payload.output_style or "tweet_bot",
            )
            candidate = fixed

            audit2 = run_answer_audit(
                call_chat_with_model=lambda m, sp, up: _call_ollama_chat_with_model(
                    model=m, system_prompt=sp, user_prompt=up
                ),
                model=audit_model,
                answer=candidate,
                question=question,
                now_block=now_block,
                allowed_urls=allowed_urls,
                required_url=required_url,
                strict_context=bool(payload.strict_context),
                allow_rewrite=False,
                max_chars=max_chars,
                require_required_url_in_answer=((payload.output_style or "tweet_bot") != "tweet_bot"),
                forbid_urls_in_answer=((payload.output_style or "tweet_bot") == "tweet_bot"),
            )
            last_audit = AuditResult(
                model=audit_model,
                passed=audit2.passed,
                score=audit2.score,
                confidence=audit2.confidence,
                issues=audit2.issues,
                fixed_answer=None,
                original_answer=original_answer if payload.audit_rewrite else None,
                raw=audit2.raw if payload.include_debug else None,
            )
            answer = candidate
            if audit2.passed:
                break

        # If we can try again, add audit feedback to the prompt and regenerate.
        answer = candidate
        if attempt < attempts and payload.audit_rewrite:
            feedback = "; ".join(last_audit.issues[:5]) if last_audit else "audit_failed"
            # Tighten user prompt without altering base structure too much
            user_prompt = (
                user_prompt
                + "\n\n"
                + f"Audit feedback: {feedback}\n"
                + "Rewrite to comply with rules. Do not add new facts.\n"
            )
            continue

        break

    # Build links out:
    # - explicit links (caller)
    # - source links (data.json metadata)
    # - ensure required_url is included
    links_out: List[str] = []
    seen: Set[str] = set()
    for u in (payload.links or []):
        if u and u not in seen:
            seen.add(u)
            links_out.append(u)
    for u in (source_links or []):
        if u and u not in seen:
            seen.add(u)
            links_out.append(u)

    if required_url and required_url not in seen:
        links_out.append(required_url)
        seen.add(required_url)

    debug: Optional[Dict[str, Any]] = None
    if payload.include_debug:
        debug = {
            "required_mention": required_mention,
            "required_url": required_url,
            "allowed_urls": sorted(allowed_urls),
            "source_links": source_links,
            "top_k": top_k,
        }

    return QueryResponse(
        answer=answer,
        context=context_texts,
        links=links_out,
        removed_urls=sorted(set(removed_urls_total)),
        audit=last_audit,
        debug=debug,
    )
