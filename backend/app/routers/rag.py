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
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import requests
from functools import lru_cache
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

import rag_store
from .rag_utils import *
from .rag_audit import AuditLite, run_answer_audit

RAG_MODEL = os.getenv("RAG_MODEL")
AUDIT_MODEL = os.getenv("AUDIT_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S"))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])

# Reused HTTP session for Ollama calls (tests monkeypatch this).
_session = requests.Session()

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default

def _ollama_chat_payload(*, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    num_predict = int((os.getenv("OLLAMA_NUM_PREDICT")))
    temperature = float((os.getenv("OLLAMA_TEMPERATURE")))
    num_thread = int((os.getenv("OLLAMA_NUM_THREAD")))
    return {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_thread": num_thread,
        },
    }


def _call_ollama_chat_with_model(*, model: str, system_prompt: str, user_prompt: str) -> str:
    llm = _get_ollama_llm(model=model)

    payload = _ollama_chat_payload(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
    lc_messages = []
    for m in payload["messages"]:
        role = m.get("role")
        content = m.get("content", "") or ""
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    result = llm.invoke(lc_messages)
    return (getattr(result, "content", "") or "").strip()


@lru_cache(maxsize=16)
def _get_ollama_llm(
    model: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    num_thread: Optional[int] = None,
    timeout_s: Optional[float] = None,
) -> ChatOllama:
    model = model or os.getenv("RAG_MODEL") or "llama3.1"
    base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")

    temperature = temperature if temperature is not None else _env_float("OLLAMA_TEMPERATURE", 0.2)
    num_predict = num_predict if num_predict is not None else _env_int("OLLAMA_NUM_PREDICT", 256)
    num_thread = num_thread if num_thread is not None else _env_int("OLLAMA_NUM_THREAD", 4)
    timeout_s = timeout_s if timeout_s is not None else float(_env_int("OLLAMA_TIMEOUT_S", 60))

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_predict=num_predict,
        num_thread=num_thread,
        client_kwargs={"timeout": timeout_s},
    )

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
    # Post-process retrieved chunks to avoid selecting past events for 'upcoming' prompts.
    family = _extract_topic_family(question)
    if family and family != "event":
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
    datetime: Optional[str] = None
    links: List[str] = Field(default_factory=list)
    top_k: int = 5
    extra_context: Optional[str] = None
    blocked_urls: List[str] = Field(default_factory=list)
    max_chars: Optional[int] = None
    include_debug: bool = True
    # Variety controls (optional)
    variety: float = 0.0          # 0.0 = legacy(top1), 0.2~0.5 = good
    seed: Optional[int] = None    # deterministic shuffle/sampling
    anchor_top_n: int = 8         # sample within top N (<= top_k )

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
    wants_future_events = _wants_future_events(question) and (topic_family in (None, "event"))

    # Retrieve context
    try:
        chunks = rag_store.query_similar_chunks(question, top_k=top_k)

        # Normalize blocked URLs (recently used primary links)
        blocked: Set[str] = set()
        for u in (payload.blocked_urls or []):
            nu = normalize_url(u)
            if nu:
                blocked.add(nu)

        # If we can, prefer chunks that don't point to recently-used URLs
        if blocked and chunks:
            filtered = []
            for c in chunks:
                c_links = collect_source_links(chunks=[c], limit=16)
                if any(u in blocked for u in c_links):
                    continue
                filtered.append(c)
            if filtered:
                chunks = filtered

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


    # Optional: diversify topic by reordering chunks before picking required_mention
    try:
        v = float(getattr(payload, "variety", 0.0) or 0.0)
    except Exception:
        v = 0.0
    if v > 0 and chunks:
        s = getattr(payload, "seed", None)
        if s is None:
            # deterministic-ish rotation by hour if datetime exists
            s = int(now_dt.strftime("%Y%m%d%H")) if now_dt else 0
        top_n = int(getattr(payload, "anchor_top_n", 8) or 8)
        top_n = max(2, min(top_n, len(chunks), int(payload.top_k or 5)))
        chunks = reorder_chunks_for_variety(chunks=chunks, seed=int(s), top_n=top_n, variety=v)
 

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

    # Remove blocked URLs from the allow-list (prevents reusing the same URL)
    if blocked:
        allowed_urls = {u for u in allowed_urls if u not in blocked}
        source_links = [u for u in source_links if u not in blocked]

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
        max_chars=max_chars,
    )

    # Generation (+ optional audit loop)
    removed_urls_total: List[str] = []
    original_answer: Optional[str] = None
    last_audit: Optional[AuditResult] = None

    audit_enabled = os.getenv("RAG_AUDIT")
    rewrite_enabled = os.getenv("RAG_AUDIT_REWRITE")
    try:
        attempts = int(os.getenv("RAG_AUDIT_MAX_ATTEMPTS") or ("2" if rewrite_enabled else "1"))
    except Exception:
        attempts = 2 if rewrite_enabled else 1
    attempts = max(1, attempts) if (audit_enabled or rewrite_enabled) else 1

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
            max_chars=max_chars,
            now_dt=now_dt,
        )

        candidate, removed = filter_answer_urls(
            candidate,
            allowed_urls,
        )

        removed_urls_total.extend(removed)

        candidate = finalize_answer(
            answer=candidate,
            required_mention=required_mention,
            max_chars=max_chars,
            now_dt=now_dt,
        )

        issues: List[str] = []
        if wants_future_events and now_dt:
            issues.extend(_temporal_issues_future_event_answer(candidate, now_dt=now_dt) or [])

        t = (candidate or "").strip()
        if "\n" in t or "\r" in t:
            issues.append("no line breaks (single paragraph)")
        if "http://" in t or "https://" in t:
            issues.append("no URLs in text")
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        if len(sents) != 3:
            issues.append("exactly 3 sentences")
        low = t.lower()
        if not re.search(r"\b(sunny|cloudy|windy|chilly|rainy)\b", low):
            issues.append("sentence 2 must include weather word")
        if not re.search(r"\b-?\d{1,2}\s*°\s*c\b", low):
            issues.append("sentence 2 must include temperature like 10°C")

        answer = candidate
        if issues and rewrite_enabled and attempt < attempts:
            feedback = "; ".join(issues[:5])
            user_prompt = (
                user_prompt
                + "\n\n"
                + f"Format feedback: {feedback}\n"
                + "Rewrite to comply with rules. Do not add new facts.\n"
            )
            continue
        if issues:
            break

        if not audit_enabled:
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
            allow_rewrite=bool(rewrite_enabled),
            max_chars=max_chars,
        )

        last_audit = AuditResult(
            model=audit_model,
            passed=audit_lite.passed,
            score=audit_lite.score,
            confidence=audit_lite.confidence,
            issues=audit_lite.issues,
            fixed_answer=audit_lite.fixed_answer,
            original_answer=original_answer if rewrite_enabled else None,
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
                max_chars=max_chars,
                now_dt=now_dt,
            )
                                  
            fixed, removed2 = filter_answer_urls(
                fixed,
                allowed_urls,
            )

            removed_urls_total.extend(removed2)
            fixed = finalize_answer(
                answer=fixed,
                required_mention=required_mention,
                max_chars=max_chars,
                now_dt=now_dt,
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
            )
            last_audit = AuditResult(
                model=audit_model,
                passed=audit2.passed,
                score=audit2.score,
                confidence=audit2.confidence,
                issues=audit2.issues,
                fixed_answer=None,
                original_answer=original_answer if rewrite_enabled else None,
                raw=audit2.raw if payload.include_debug else None,
            )
            answer = candidate
            if audit2.passed:
                break

        # If we can try again, add audit feedback to the prompt and regenerate.
        answer = candidate
        if attempt < attempts and rewrite_enabled:
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

    # Keep ONLY ONE link for UI output (prefer required_url / top-chunk URL)
    primary_link = required_url or (links_out[0] if links_out else "")
    links_out = [primary_link] if primary_link else []

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
