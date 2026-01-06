from __future__ import annotations

import logging
import os
from http import HTTPStatus
from typing import Any, Callable, Literal, Optional

import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

import rag_store
from rag_store import RAGChunk

# --- Import split modules (works both as package and flat) ---
from .rag_utils import (
    truthy_env,
    resolve_now_datetime,
    collect_allowed_urls,
    filter_answer_urls,
    select_required_context,
    build_chat_prompts,
    finalize_answer,
    get_output_style_default,
    get_max_chars_default,
)
from .rag_audit import run_answer_audit, AuditResult as AuditResultLite

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])
_session = requests.Session()


# -------------------------------------------------------------------
# Models (keep compatibility with existing scripts)
# -------------------------------------------------------------------

class IngestRequest(BaseModel):
    documents: list[str] = Field(..., description="Raw texts to ingest into the vector store.")


class IngestResponse(BaseModel):
    ingested: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

    # caller-provided metadata (NOT stored in RAG)
    datetime: str | None = Field(default=None, description="Optional ISO datetime string.")
    links: list[str] | None = Field(default=None, description="Optional URLs provided by the caller.")

    top_k: int = Field(5, ge=1, le=128)

    extra_context: str | None = Field(default=None, description="Short-lived context (e.g., live weather).")
    context: str | None = Field(default=None, description="Alias for extra_context")
    user_context: str | None = Field(default=None, description="Alias for extra_context")

    include_debug: bool = Field(default=False)

    output_style: Literal["default", "tweet_bot"] = Field(default="tweet_bot")
    max_chars: int = Field(default=512, ge=50, le=1024)

    audit: bool | None = Field(default=None)
    audit_rewrite: bool | None = Field(default=None)
    audit_model: str | None = Field(default=None)

    # strict loop controls
    max_attempts: int | None = Field(default=None, ge=1, le=20, description="Max regenerate attempts when audit fails.")


class ChunkOut(BaseModel):
    text: str
    distance: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditResult(BaseModel):
    model: str | None = None
    passed: bool
    issues: list[str] = Field(default_factory=list)
    fixed_answer: str | None = None
    original_answer: str | None = None
    raw: str | None = None


class QueryResponse(BaseModel):
    answer: str
    links: list[str] | None = None

    # debug
    context: list[str] | None = None
    chunks: list[ChunkOut] | None = None
    removed_urls: list[str] | None = None
    audit: AuditResult | None = None


# -------------------------------------------------------------------
# LLM / API helpers
# -------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


def _call_chat_with_model(model: str, sys_prompt: str, user_prompt: str) -> str:
    """
    Low-level call to Ollama's /api/chat with timeouts and friendly errors.
    """
    url = f"{DEFAULT_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    # Keep a hard timeout so backend doesn't hang forever
    try:
        r = _session.post(url, json=payload, timeout=(10, 90))
    except requests.Timeout as e:
        raise HTTPException(
            status_code=HTTPStatus.GATEWAY_TIMEOUT,
            detail=f"Ollama timeout: {e}",
        )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=f"Ollama connection error: {e}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=f"Ollama error {r.status_code}: {r.text[:500]}",
        )

    data = r.json()
    # Ollama returns {"message": {"content": "..."}}
    msg = (data or {}).get("message") or {}
    return (msg.get("content") or "").strip()


def _get_rag_model() -> str:
    return os.getenv("RAG_MODEL", DEFAULT_MODEL)


def _get_audit_model() -> str:
    return os.getenv("AUDIT_MODEL", _get_rag_model())


def _get_timezone() -> str:
    return os.getenv("TZ", "Asia/Tokyo")


def _format_now_block(now_label: str) -> str:
    # Keep stable format for prompting.
    # Include weekday and timezone label.
    return "\n".join(
        [
            "[NOW]",
            f"Current time: {now_label}",
            "[/NOW]",
        ]
    )


def _audit_context_text() -> str:
    # Small helper for consistent audit prompt framing
    return "\n".join(
        [
            "You are a strict validator.",
            "Return JSON only.",
            "Never hallucinate URLs.",
            "If required mention is missing, mark failed.",
        ]
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest):
    # Minimal ingestion: push to vector store
    if not payload.documents:
        return {"ingested": 0}
    n = rag_store.ingest_texts(payload.documents)
    return {"ingested": n}


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request):
    # Resolve time
    now_dt, now_label = resolve_now_datetime(payload.datetime, _get_timezone())
    now_block = _format_now_block(now_label)

    # Style defaults
    style = payload.output_style or get_output_style_default()
    max_chars = payload.max_chars or get_max_chars_default(style)

    # Short-lived extra context
    live_extra = payload.extra_context or payload.context or payload.user_context

    # Retrieve from vector store
    chunks: list[RAGChunk] = rag_store.query(payload.question, top_k=payload.top_k)
    context_texts = [c.text for c in chunks]

    # Collect allowed URLs from caller and from retrieved chunks
    allowed_urls = collect_allowed_urls(
        user_links=(payload.links or []),
        rag_chunks=chunks,
        extra_text=live_extra,
    )

    # Required mention/link (forces answers to reference RAG context)
    required_mention, required_url = select_required_context(chunks, allowed_urls)

    # Build prompts
    sys_prompt, user_prompt = build_chat_prompts(
        question=payload.question,
        now_block=now_block,
        context_texts=context_texts,
        live_extra=live_extra,
        required_mention=required_mention,
        required_url=required_url,
        output_style=style,
        max_chars=max_chars,
    )

    model = _get_rag_model()
    raw_answer = _call_chat_with_model(model, sys_prompt, user_prompt)

    # Post-process answer: normalize, enforce mention/link, trim to max chars
    answer = finalize_answer(
        raw_answer=raw_answer,
        output_style=style,
        max_chars=max_chars,
        required_mention=required_mention,
        required_url=required_url,
    )

    # Remove URLs not allowed (hard safety)
    answer, removed = filter_answer_urls(answer, allowed_urls=allowed_urls)

    # Final enforce (after URL filter)
    answer = finalize_answer(
        raw_answer=answer,
        output_style=style,
        max_chars=max_chars,
        required_mention=required_mention,
        required_url=required_url,
    )

    # ---- audit loop controls ----
    audit_enabled = payload.audit if payload.audit is not None else truthy_env("AUDIT_ENABLED", True)
    audit_rewrite = payload.audit_rewrite if payload.audit_rewrite is not None else truthy_env("AUDIT_REWRITE", True)
    audit_model = payload.audit_model or _get_audit_model()
    max_attempts = payload.max_attempts or int(os.getenv("AUDIT_MAX_ATTEMPTS", "6"))

    last_answer = answer
    last_removed = removed
    last_audit: AuditResult | None = None

    feedback = ""
    if audit_enabled:
        for attempt in range(max_attempts):
            # Validate current candidate
            audit_lite = run_answer_audit(
                question=payload.question,
                answer=last_answer,
                now_block=now_block,
                required_mention=required_mention,
                required_url=required_url,
                allowed_urls=sorted(allowed_urls),
                max_chars=max_chars,
                output_style=style,
                audit_context=_audit_context_text(),
                strict=True,
                rewrite=audit_rewrite,
                audit_model=audit_model,
                call_chat_with_model=lambda m, sp, up: _call_chat_with_model(m, sp, up),
            )

            last_audit = AuditResult(
                model=audit_model,
                passed=audit_lite.passed,
                issues=audit_lite.issues,
                fixed_answer=audit_lite.fixed_answer,
                raw=(audit_lite.raw if payload.include_debug else None),
            )

            if audit_lite.passed:
                break

            # If auditor gave fixed_answer, try it once (re-audit without rewrite)
            if audit_lite.fixed_answer:
                fixed = finalize_answer(
                    raw_answer=audit_lite.fixed_answer,
                    output_style=style,
                    max_chars=max_chars,
                    required_mention=required_mention,
                    required_url=required_url,
                )
                fixed, removed2 = filter_answer_urls(fixed, allowed_urls=allowed_urls)
                fixed = finalize_answer(
                    raw_answer=fixed,
                    output_style=style,
                    max_chars=max_chars,
                    required_mention=required_mention,
                    required_url=required_url,
                )

                audit2 = run_answer_audit(
                    question=payload.question,
                    answer=fixed,
                    now_block=now_block,
                    required_mention=required_mention,
                    required_url=required_url,
                    allowed_urls=sorted(allowed_urls),
                    max_chars=max_chars,
                    output_style=style,
                    audit_context=_audit_context_text(),
                    strict=True,
                    rewrite=False,
                    audit_model=audit_model,
                    call_chat_with_model=lambda m, sp, up: _call_chat_with_model(m, sp, up),
                )
                if audit2.passed:
                    last_answer = fixed
                    last_removed = last_removed + removed2
                    last_audit.original_answer = candidate
                    last_audit.passed = True
                    last_audit.fixed_answer = fixed
                    break

            # Rewrite via generation with feedback
            candidate = last_answer
            feedback = "\n".join(f"- {x}" for x in (audit_lite.issues or [])[:8]) or "- Fix all unmet requirements."

            if not audit_rewrite:
                break

            sys_prompt2, user_prompt2 = build_chat_prompts(
                question=payload.question,
                now_block=now_block,
                context_texts=context_texts,
                live_extra=live_extra,
                required_mention=required_mention,
                required_url=required_url,
                output_style=style,
                max_chars=max_chars,
                audit_feedback=feedback,
                previous_answer=candidate,
            )
            regenerated = _call_chat_with_model(model, sys_prompt2, user_prompt2)
            regenerated = finalize_answer(
                raw_answer=regenerated,
                output_style=style,
                max_chars=max_chars,
                required_mention=required_mention,
                required_url=required_url,
            )
            regenerated, removed2 = filter_answer_urls(regenerated, allowed_urls=allowed_urls)
            regenerated = finalize_answer(
                raw_answer=regenerated,
                output_style=style,
                max_chars=max_chars,
                required_mention=required_mention,
                required_url=required_url,
            )

            last_answer = regenerated
            last_removed = last_removed + removed2

        if last_audit and not last_audit.passed:
            # Keep as failed but still return something
            last_audit.original_answer = last_answer

        # feed issues back into next attempt
        feedback = "\n".join(f"- {x}" for x in (audit_lite.issues or [])[:8]) or "- Fix all unmet requirements."

    # ---- debug payload ----
    dbg_context = None
    dbg_chunks = None
    if payload.include_debug:
        dbg_context = context_texts[:8] + ([f"[Live context]\n{live_extra}"] if live_extra else [])
        dbg_chunks = [
            ChunkOut(text=c.text, distance=c.distance, metadata=(c.metadata or {}))
            for c in chunks[: payload.top_k]
        ]

    # links for UI
    links_out: list[str] = []
    seen = set()
    for u in req_links + sorted(allowed_urls):
        if u and u not in seen:
            links_out.append(u)
            seen.add(u)

    return QueryResponse(
        answer=last_answer,
        links=(links_out or None),
        context=dbg_context,
        chunks=dbg_chunks,
        removed_urls=(last_removed or None if payload.include_debug else None),
        audit=last_audit,
    )
