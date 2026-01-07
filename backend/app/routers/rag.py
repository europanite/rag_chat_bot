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
- Provides _get_ollama_base_url/_get_ollama_chat_model and _call_ollama_chat helpers.

Note: The actual LLM calls are to an Ollama server at {OLLAMA_BASE_URL}/api/chat.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

import rag_store
from .rag_utils import (
    build_chat_prompts,
    collect_allowed_urls,
    filter_answer_urls,
    finalize_answer,
    select_required_context,
)
from .rag_audit import AuditLite, run_answer_audit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])

# Reused HTTP session for Ollama calls (tests monkeypatch this).
_session = requests.Session()

# Defaults (tests rely on these env keys)
DEFAULT_BASE_URL = "http://ollama:11434"
DEFAULT_CHAT_MODEL = "llama3.1"
DEFAULT_AUDIT_MODEL = "llama3.1"


def _get_ollama_base_url() -> str:
    base = (os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).strip()
    # normalize trailing slash
    return base[:-1] if base.endswith("/") else base


def _get_ollama_chat_model() -> str:
    return (os.getenv("OLLAMA_CHAT_MODEL") or DEFAULT_CHAT_MODEL).strip()


def _get_rag_model() -> str:
    # allow overriding RAG model independently
    return (os.getenv("RAG_MODEL") or _get_ollama_chat_model()).strip()


def _get_audit_model() -> str:
    return (os.getenv("AUDIT_MODEL") or DEFAULT_AUDIT_MODEL).strip()


def _get_timeout_s() -> int:
    # tests expect an int timeout argument
    try:
        return int(os.getenv("OLLAMA_TIMEOUT_S") or "90")
    except Exception:
        return 90


def _ollama_chat_payload(*, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }


def _call_ollama_chat_with_model(*, model: str, system_prompt: str, user_prompt: str) -> str:
    """Direct Ollama call for a specific model (tests may monkeypatch this)."""
    url = f"{_get_ollama_base_url()}/api/chat"
    payload = _ollama_chat_payload(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
    resp = _session.post(url, json=payload, timeout=_get_timeout_s())
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
        model=_get_rag_model(),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _docs_dir() -> str:
    return (os.getenv("RAG_DOCS_DIR") or "docs").strip()


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


def _max_chars_default() -> int:
    for k in ("RAG_MAX_CHARS", "MAX_CHARS", "TWEET_MAX_CHARS"):
        v = os.getenv(k)
        if v:
            try:
                return max(40, min(2000, int(v)))
            except Exception:
                pass
    return 280


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
    return {"ok": True, "chunks_in_store": n, "docs_dir": _docs_dir()}


@router.post("/reindex")
def reindex() -> Dict[str, Any]:
    """
    Rebuild the index from docs dir JSON files (expected by scripts).
    """
    docs = _docs_dir()
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

    # Retrieve context
    try:
        chunks = rag_store.query_similar_chunks(question, top_k=top_k)
    except Exception as e:
        logger.exception("query_similar_chunks failed")
        raise HTTPException(status_code=502, detail=f"RAG query failed: {e}")

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant context.")

    context_texts = [(getattr(c, "text", "") or "").strip() for c in chunks]
    context_texts = [t for t in context_texts if t]

    # allow-list URLs
    allowed_urls: Set[str] = collect_allowed_urls(
        user_links=payload.links,
        context_texts=context_texts,
        extra_text=payload.extra_context,
    )

    required_mention, required_url = select_required_context(
        chunks=chunks,
        allowed_urls=allowed_urls,
    )

    now_block = _now_block(request, payload.datetime)
    max_chars = int(payload.max_chars or _max_chars_default())

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
        )
        candidate, removed = filter_answer_urls(candidate, allowed_urls)
        removed_urls_total.extend(removed)
        candidate = finalize_answer(
            answer=candidate,
            required_mention=required_mention,
            required_url=required_url,
            max_chars=max_chars,
        )

        # If no audit requested, accept immediately
        if not payload.audit:
            answer = candidate
            break

        audit_model = (payload.audit_model or _get_audit_model()).strip()
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
            )
            fixed, removed2 = filter_answer_urls(fixed, allowed_urls)
            removed_urls_total.extend(removed2)
            fixed = finalize_answer(
                answer=fixed,
                required_mention=required_mention,
                required_url=required_url,
                max_chars=max_chars,
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

    # Build links out (include explicit links first, then allowed urls)
    links_out: List[str] = []
    seen: Set[str] = set()
    for u in (payload.links or []):
        if u and u not in seen:
            seen.add(u)
            links_out.append(u)
    for u in sorted(allowed_urls):
        if u and u not in seen:
            seen.add(u)
            links_out.append(u)

    debug: Optional[Dict[str, Any]] = None
    if payload.include_debug:
        debug = {
            "required_mention": required_mention,
            "required_url": required_url,
            "allowed_urls": sorted(allowed_urls),
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
