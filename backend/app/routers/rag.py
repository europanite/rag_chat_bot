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
try:
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
except Exception:  # pragma: no cover
    from rag_utils import (  # type: ignore
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
    from rag_audit import run_answer_audit, AuditResult as AuditResultLite  # type: ignore


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
    id: str | None = None
    text: str
    distance: float | None = None
    metadata: dict = Field(default_factory=dict)


class AuditResult(BaseModel):
    model: str | None = None
    passed: bool
    issues: list[str] = Field(default_factory=list)
    fixed_answer: str | None = None

    # debug
    original_answer: str | None = None
    raw: str | None = None


class StatusResponse(BaseModel):
    docs_dir: str
    json_files: int
    chunks_in_store: int
    files: list[str]


class ReindexResponse(BaseModel):
    reindexed: bool
    json_files: int
    chunks_in_store: int


class QueryResponse(BaseModel):
    answer: str
    links: list[str] | None = None
    context: list[str] | None = None
    chunks: list[ChunkOut] | None = None
    removed_urls: list[str] | None = None
    audit: AuditResult | None = None


# -------------------------------------------------------------------
# Ollama chat call
# -------------------------------------------------------------------

def _get_chat_url() -> str:
    return os.getenv("OLLAMA_CHAT_URL", "http://ollama:11434/api/chat")


def _get_chat_model() -> str:
    return os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")


def _call_chat_with_model(model: str, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    resp = _session.post(_get_chat_url(), json=payload, timeout=90)
    resp.raise_for_status()
    obj = resp.json()
    msg = obj.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise ValueError(f"Unexpected Ollama response: {obj}")
    return content


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    if not payload.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")
    ok = 0
    for doc in payload.documents:
        if isinstance(doc, str) and doc.strip():
            rag_store.add_document(doc)
            ok += 1
    if ok == 0:
        raise HTTPException(status_code=HTTPStatus.BAD_GATEWAY, detail="All ingests failed.")
    return IngestResponse(ingested=ok)


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    docs_dir = os.getenv("RAG_DOCS_DIR", "/data/json")
    files = rag_store.list_json_files(docs_dir)
    try:
        count = rag_store.get_collection_count()
    except Exception:
        count = 0
    return StatusResponse(docs_dir=docs_dir, json_files=len(files), chunks_in_store=count, files=files)


@router.post("/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    docs_dir = os.getenv("RAG_DOCS_DIR", "/data/json")
    files = rag_store.list_json_files(docs_dir)
    rag_store.reindex(docs_dir)
    try:
        count = rag_store.get_collection_count()
    except Exception:
        count = 0
    return ReindexResponse(reindexed=True, json_files=len(files), chunks_in_store=count)


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query_rag(payload: QueryRequest, http_request: Request) -> QueryResponse:
    # ---- resolve defaults ----
    style = payload.output_style or get_output_style_default()
    max_chars = int(payload.max_chars or get_max_chars_default(style))

    audit_enabled = payload.audit if payload.audit is not None else truthy_env("RAG_AUDIT_DEFAULT", False)
    audit_rewrite = payload.audit_rewrite if payload.audit_rewrite is not None else truthy_env("RAG_AUDIT_REWRITE_DEFAULT", True)
    audit_model = payload.audit_model or os.getenv("RAG_AUDIT_MODEL") or _get_chat_model()

    max_attempts = payload.max_attempts or int(os.getenv("RAG_MAX_ATTEMPTS", "6"))

    # ---- retrieve ----
    try:
        chunks: list[RAGChunk] = rag_store.query_similar_chunks(payload.question, top_k=payload.top_k)
    except Exception as exc:
        logger.exception("Vector search failed", exc_info=exc)
        raise HTTPException(status_code=HTTPStatus.BAD_GATEWAY, detail=f"Vector search failed: {exc}") from exc

    if not chunks:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="No relevant context found.")

    # ---- build context strings ----
    context_texts: list[str] = []
    for c in chunks:
        t = (c.text or "").strip()
        if t:
            context_texts.append(t)

    live_extra = (payload.extra_context or payload.context or payload.user_context or "").strip()

    # allowed urls: from context + request links
    req_links = [u.strip() for u in (payload.links or []) if isinstance(u, str) and u.strip()]
    allowed_urls = collect_allowed_urls("\n".join(context_texts), live_extra, extra_urls=req_links)

    # pick one required mention/url from top chunk
    req_mention, req_url = select_required_context(chunks)
    required_mention = req_mention or ""
    required_url = req_url or ""

    # NOW block (truth)
    now_dt, now_label = resolve_now_datetime(payload.datetime, tz_name="Asia/Tokyo")
    now_block = f"local_datetime: {now_label}\ntoday: {now_dt.date().isoformat()}"

    # ---- generation loop ----
    last_audit: Optional[AuditResult] = None
    last_removed: list[str] = []
    last_answer: str = ""

    feedback = ""  # we append audit issues into prompt for next attempt

    def _audit_context_text() -> str:
        ctx = "\n".join(context_texts[:8])
        if live_extra:
            ctx += "\n\n[Live context]\n" + live_extra
        return ctx

    for attempt in range(1, max_attempts + 1):
        extra_for_prompt = live_extra
        if feedback:
            extra_for_prompt = (extra_for_prompt + "\n\n[Audit feedback to fix]\n" + feedback).strip()

        sys_p, usr_p = build_chat_prompts(
            question=payload.question,
            now_block=now_block,
            chunks=chunks,
            extra_context=extra_for_prompt,
            output_style=style,
            required_mention=required_mention,
            required_url=required_url,
            allowed_urls=allowed_urls,
            max_chars=max_chars,
        )

        # generate
        try:
            raw = _call_chat_with_model(_get_chat_model(), sys_p, usr_p)
        except Exception as exc:
            logger.exception("Ollama chat failed", exc_info=exc)
            raise HTTPException(status_code=HTTPStatus.BAD_GATEWAY, detail=str(exc)) from exc

        candidate = finalize_answer(
            raw_answer=raw,
            output_style=style,
            max_chars=max_chars,
            required_mention=required_mention,
            required_url=required_url,
        )

        # filter urls
        candidate, removed = filter_answer_urls(candidate, allowed_urls=allowed_urls)
        candidate = finalize_answer(
            raw_answer=candidate,
            output_style=style,
            max_chars=max_chars,
            required_mention=required_mention,
            required_url=required_url,
        )

        last_answer = candidate
        last_removed = removed

        if not audit_enabled:
            last_audit = None
            break

        # audit (strict) + rewrite enabled
        audit_lite: AuditResultLite = run_answer_audit(
            question=payload.question,
            answer=candidate,
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
