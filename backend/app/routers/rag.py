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
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set,TypedDict
import requests
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import rag_store
from .rag_utils import *
from .rag_audit import AuditLite, run_answer_audit
RAG_MODEL = os.getenv("RAG_MODEL")
AUDIT_MODEL = os.getenv("AUDIT_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S"))
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])
# Reused HTTP session for Ollama calls (tests monkeypatch this).
_session = requests.Session()
# =========================
# LangGraph: generation loop
# =========================
class _GenState(TypedDict, total=False):
    # inputs
    question: str
    sys_prompt: str
    user_prompt: str
    required_mention: str
    required_url: str | None
    allowed_urls: set[str]
    max_chars: int
    now_dt: datetime | None
    now_block: str | None
    wants_future_events: bool
    strict_context: bool
    include_debug: bool
    audit_enabled: bool
    rewrite_enabled: bool
    attempts: int
    attempt: int
    # working
    candidate: str
    answer: str
    original_answer: str | None
    removed_urls_total: list[str]
    issues: list[str]
    last_audit: AuditResult | None
    # routing flags
    need_regen: bool
    need_audit: bool
    has_fixed: bool
def _node_generate(state: _GenState) -> _GenState:
    try:
        candidate = _call_ollama_chat(
            question=state["question"],
            system_prompt=state["sys_prompt"],
            user_prompt=state["user_prompt"],
        )
    except Exception as e:
        logger.exception("ollama chat failed")
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
    state["candidate"] = candidate
    if state.get("original_answer") is None:
        state["original_answer"] = candidate
    state["answer"] = candidate
    return state
def _node_postprocess(state: _GenState) -> _GenState:
    candidate = state.get("candidate") or ""
    candidate = finalize_answer(
        answer=candidate,
        required_mention=state["required_mention"],
        max_chars=state["max_chars"],
        now_dt=state.get("now_dt"),
    )
    candidate, removed = filter_answer_urls(candidate, state["allowed_urls"])
    state["removed_urls_total"].extend(removed)
    candidate = finalize_answer(
        answer=candidate,
        required_mention=state["required_mention"],
        max_chars=state["max_chars"],
        now_dt=state.get("now_dt"),
    )
    state["candidate"] = candidate
    state["answer"] = candidate
    return state
def _node_validate_format(state: _GenState) -> _GenState:
    candidate = (state.get("candidate") or "").strip()
    issues: list[str] = []
    if state.get("wants_future_events") and state.get("now_dt"):
        issues.extend(_temporal_issues_future_event_answer(candidate, now_dt=state["now_dt"]) or [])
    if "\n" in candidate or "\r" in candidate:
        issues.append("no line breaks (single paragraph)")
    if "http://" in candidate or "https://" in candidate:
        issues.append("no URLs in text")
    sents = split_sentences(candidate)
    if len(sents) != 3:
        issues.append("exactly 3 sentences")
    low = candidate.lower()
    if not re.search(r"\b(sunny|cloudy|windy|chilly|rainy)\b", low):
        issues.append("sentence 2 must include weather word")
    if not re.search(r"\b-?\d{1,2}\s*°\s*c\b", low):
        issues.append("sentence 2 must include temperature like 10°C")
    if not answer_mentions_required(candidate, state.get("required_mention") or ""):
        issues.append("sentence 3 must mention required mention")
    if not third_sentence_is_substantive(candidate, state.get("required_mention") or ""):
        issues.append("sentence 3 must include one concrete supported detail")
    state["issues"] = issues
    attempt = int(state.get("attempt") or 1)
    attempts = int(state["attempts"])
    rewrite_enabled = bool(state["rewrite_enabled"])
    state["need_regen"] = bool(issues) and rewrite_enabled and attempt < attempts
    state["need_audit"] = (not issues) and bool(state["audit_enabled"])
    state["has_fixed"] = False
    if state["need_regen"]:
        feedback = "; ".join(issues[:5])
        state["user_prompt"] = (
            state["user_prompt"]
            + "\n\n"
            + f"Format feedback: {feedback}\n"
            + "Rewrite to comply with rules. Do not add new facts.\n"
        )
        state["attempt"] = attempt + 1
    return state
def _node_audit(state: _GenState) -> _GenState:
    candidate = state.get("candidate") or ""
    audit_model = AUDIT_MODEL
    audit_lite: AuditLite = run_answer_audit(
        call_chat_with_model=lambda m, sp, up: _call_ollama_chat_with_model(model=m, system_prompt=sp, user_prompt=up),
        model=audit_model,
        answer=candidate,
        question=state["question"],
        now_block=state.get("now_block") or "",
        allowed_urls=state["allowed_urls"],
        required_url=state.get("required_url"),
        strict_context=bool(state.get("strict_context")),
        allow_rewrite=bool(state["rewrite_enabled"]),
        max_chars=int(state["max_chars"]),
        require_required_url_in_answer=False,
    )
    state["last_audit"] = AuditResult(
        model=audit_model,
        passed=audit_lite.passed,
        score=audit_lite.score,
        confidence=audit_lite.confidence,
        issues=audit_lite.issues,
        fixed_answer=audit_lite.fixed_answer,
        original_answer=state.get("original_answer") if state["rewrite_enabled"] else None,
        raw=audit_lite.raw if state.get("include_debug") else None,
    )
    # passed
    if audit_lite.passed:
        state["need_regen"] = False
        state["need_audit"] = False
        state["has_fixed"] = False
        state["answer"] = candidate
        return state
    # fixed_answer path
    if audit_lite.fixed_answer:
        state["candidate"] = audit_lite.fixed_answer
        state["has_fixed"] = True
        state["need_regen"] = False
        state["need_audit"] = False
        return state
    # regen with audit feedback
    attempt = int(state.get("attempt") or 1)
    attempts = int(state["attempts"])
    rewrite_enabled = bool(state["rewrite_enabled"])
    state["has_fixed"] = False
    state["need_audit"] = False
    state["need_regen"] = bool(rewrite_enabled) and attempt < attempts
    if state["need_regen"]:
        feedback = "; ".join((state["last_audit"].issues[:5] if state["last_audit"] else ["audit_failed"]))
        state["user_prompt"] = (
            state["user_prompt"]
            + "\n\n"
            + f"Audit feedback: {feedback}\n"
            + "Rewrite to comply with rules. Do not add new facts.\n"
        )
        state["attempt"] = attempt + 1
    return state
def _node_apply_fixed_then_reaudit(state: _GenState) -> _GenState:
    # apply fixed -> postprocess -> audit again (allow_rewrite=False)
    state = _node_postprocess(state)
    candidate = state.get("candidate") or ""
    audit_model = AUDIT_MODEL
    audit2: AuditLite = run_answer_audit(
        call_chat_with_model=lambda m, sp, up: _call_ollama_chat_with_model(model=m, system_prompt=sp, user_prompt=up),
        model=audit_model,
        answer=candidate,
        question=state["question"],
        now_block=state.get("now_block") or "",
        allowed_urls=state["allowed_urls"],
        required_url=state.get("required_url"),
        strict_context=bool(state.get("strict_context")),
        allow_rewrite=False,
        max_chars=int(state["max_chars"]),
        require_required_url_in_answer=False,
    )
    state["last_audit"] = AuditResult(
        model=audit_model,
        passed=audit2.passed,
        score=audit2.score,
        confidence=audit2.confidence,
        issues=audit2.issues,
        fixed_answer=None,
        original_answer=state.get("original_answer") if state["rewrite_enabled"] else None,
        raw=audit2.raw if state.get("include_debug") else None,
    )
    if audit2.passed:
        state["need_regen"] = False
        state["answer"] = candidate
        return state
    # if still fail, optionally regen with audit feedback
    attempt = int(state.get("attempt") or 1)
    attempts = int(state["attempts"])
    rewrite_enabled = bool(state["rewrite_enabled"])
    state["need_regen"] = bool(rewrite_enabled) and attempt < attempts
    if state["need_regen"]:
        feedback = "; ".join(state["last_audit"].issues[:5]) if state["last_audit"] else "audit_failed"
        state["user_prompt"] = (
            state["user_prompt"]
            + "\n\n"
            + f"Audit feedback: {feedback}\n"
            + "Rewrite to comply with rules. Do not add new facts.\n"
        )
        state["attempt"] = attempt + 1
    state["answer"] = candidate
    return state
@lru_cache(maxsize=1)
def _get_generation_graph():
    g = StateGraph(_GenState)
    g.add_node("generate", _node_generate)
    g.add_node("postprocess", _node_postprocess)
    g.add_node("validate", _node_validate_format)
    g.add_node("audit", _node_audit)
    g.add_node("apply_fixed", _node_apply_fixed_then_reaudit)
    g.set_entry_point("generate")
    g.add_edge("generate", "postprocess")
    g.add_edge("postprocess", "validate")
    def _route_after_validate(s: _GenState):
        if s.get("need_regen"):
            return "regen"
        if s.get("need_audit"):
            return "audit"
        return "end"
    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"regen": "generate", "audit": "audit", "end": END},
    )
    def _route_after_audit(s: _GenState):
        if s.get("has_fixed"):
            return "apply_fixed"
        if s.get("need_regen"):
            return "regen"
        return "end"
    g.add_conditional_edges(
        "audit",
        _route_after_audit,
        {"apply_fixed": "apply_fixed", "regen": "generate", "end": END},
    )
    # apply_fixed can either finish or regenerate (audit feedback)
    def _route_after_apply_fixed(s: _GenState):
        if s.get("need_regen"):
            return "regen"
        return "end"
    g.add_conditional_edges(
        "apply_fixed",
        _route_after_apply_fixed,
        {"regen": "generate", "end": END},
    )
    return g.compile()
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
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
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
def _call_openai_chat_with_model(*, model: str, system_prompt: str, user_prompt: str) -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK is not installed")
    api_key = (OPENAI_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = (OPENAI_BASE_URL or "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=_env_float("OPENAI_TEMPERATURE", 0.2),
        max_tokens=_env_int("OPENAI_MAX_TOKENS", 256),
    )
    message = response.choices[0].message if response.choices else None
    content = getattr(message, "content", "") or ""
    return content.strip()
def _call_ollama_chat_with_model(*, model: str, system_prompt: str, user_prompt: str) -> str:
    if LLM_PROVIDER == "openai":
        openai_model = model or OPENAI_CHAT_MODEL or RAG_MODEL or "gpt-4.1-mini"
        return _call_openai_chat_with_model(
            model=openai_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
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
    model = RAG_MODEL
    if LLM_PROVIDER == "openai":
        model = OPENAI_CHAT_MODEL or RAG_MODEL or "gpt-4.1-mini"
    return _call_ollama_chat_with_model(
        model=model,
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
# -----------------------------
# Phase 5 helpers: intent routing, blocked terms, debug logs
# -----------------------------
_JP_YMD_RE = re.compile(r"\b(20\d{2})年(\d{1,2})月(\d{1,2})日\b")
_JP_MD_RE = re.compile(r"\b(\d{1,2})月(\d{1,2})日\b")
def _normalize_term(t: str) -> str:
    if not t:
        return ""
    s = str(t).strip()
    if not s:
        return ""
    # common cleanup
    s = s.replace("#", "").strip()
    s = re.sub(r"\s+", " ", s)
    # ignore tiny tokens (noise)
    if len(s) < 2:
        return ""
    return s.lower()
def _infer_intent(question: str, topic_family: Optional[str]) -> str:
    tf = (topic_family or "").strip().lower()
    if tf:
        return tf
    q = (question or "").lower()
    # very light heuristics (keep stable)
    if any(k in q for k in ("event", "festival", "開催", "イベント", "schedule")):
        return "event"
    if any(k in q for k in ("restaurant", "cafe", "ramen", "食べ", "ランチ", "ディナー")):
        return "restaurant"
    if any(k in q for k in ("activity", "体験", "遊", "アクティビティ")):
        return "activity"
    return "general"
def _chunk_search_blob(chunk: Any) -> str:
    parts: list[str] = []
    txt = (getattr(chunk, "text", "") or "").strip()
    if txt:
        parts.append(txt)
    meta = getattr(chunk, "meta", None)
    if isinstance(meta, dict):
        for k in ("title", "place", "tags", "category", "source", "url"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
            elif isinstance(v, list):
                parts.extend([str(x) for x in v if str(x).strip()])
    return "\n".join(parts).lower()
def _chunk_is_blocked(chunk: Any, *, blocked_urls: Set[str], blocked_terms: Set[str]) -> bool:
    if blocked_urls:
        try:
            links = collect_source_links(chunks=[chunk], limit=16)
        except Exception:
            links = []
        for u in links:
            nu = normalize_url(u)
            if nu and nu in blocked_urls:
                return True
    if blocked_terms:
        blob = _chunk_search_blob(chunk)
        for t in blocked_terms:
            if t and t in blob:
                return True
    return False
def _chunk_id(chunk: Any) -> str:
    cid = getattr(chunk, "id", None)
    if isinstance(cid, str) and cid:
        return cid
    meta = getattr(chunk, "meta", None)
    if isinstance(meta, dict):
        for k in ("id", "doc_id", "slug"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    # fallback: stable hash of leading text
    txt = (getattr(chunk, "text", "") or "")[:80]
    return f"h{abs(hash(txt)) % 10_000_000:07d}"
def _extract_dates_from_text(text: str, *, now_dt: Optional[datetime]) -> list[date]:
    if not text:
        return []
    out: list[date] = []
    today = (now_dt.date() if now_dt else date.today())
    for m in _ISO_DATE_RE.finditer(text):
        try:
            d = date.fromisoformat(m.group(1))
            out.append(d)
        except Exception:
            pass
    for m in _MONTH_DAY_RE.finditer(text):
        mon_s = (m.group(1) or "").strip().lower().rstrip(".")
        dd = int(m.group(2))
        mm = _MONTHS.get(mon_s[:3], _MONTHS.get(mon_s))
        if not mm:
            continue
        yy = today.year
        try:
            d = date(yy, mm, dd)
            if d < today:
                d = date(yy + 1, mm, dd)
            out.append(d)
        except Exception:
            pass
    for m in _SLASH_DATE_RE.finditer(text):
        mm = int(m.group(1))
        dd = int(m.group(2))
        yy_s = m.group(3)
        yy = today.year
        if yy_s:
            try:
                yy = int(yy_s)
                if yy < 100:
                    yy += 2000
            except Exception:
                yy = today.year
        try:
            d = date(yy, mm, dd)
            if not yy_s and d < today:
                d = date(yy + 1, mm, dd)
            out.append(d)
        except Exception:
            pass
    for m in _JP_YMD_RE.finditer(text):
        try:
            yy = int(m.group(1)); mm = int(m.group(2)); dd = int(m.group(3))
            out.append(date(yy, mm, dd))
        except Exception:
            pass
    for m in _JP_MD_RE.finditer(text):
        try:
            mm = int(m.group(1)); dd = int(m.group(2))
            yy = today.year
            d = date(yy, mm, dd)
            if d < today:
                d = date(yy + 1, mm, dd)
            out.append(d)
        except Exception:
            pass
    # unique, sorted
    uniq = sorted({d for d in out})
    return uniq
def _extract_normalized_event_dates(chunks: list[Any], *, now_dt: Optional[datetime]) -> list[str]:
    dates: set[date] = set()
    for c in chunks:
        txt = (getattr(c, "text", "") or "")
        for d in _extract_dates_from_text(txt, now_dt=now_dt):
            dates.add(d)
        meta = getattr(c, "meta", None)
        if isinstance(meta, dict):
            for k in ("date", "start", "end", "datetime"):
                d2 = _safe_parse_date_like(meta.get(k), now_dt=now_dt)
                if d2:
                    dates.add(d2)
    return [d.isoformat() for d in sorted(dates)]
def _format_event_dates_block(date_isos: list[str]) -> str:
    if not date_isos:
        return "(No explicit event date found in context.)"
    lines = ["Event date hints (normalized from context):"]
    for di in date_isos[:10]:
        lines.append(f"- {di}")
    return "\n".join(lines)
def _write_rag_debug_log(*, payload: Any, intent: str, question: str, now_dt: Optional[datetime], now_block: str,
                         chunks: list[Any], required_url: Optional[str], required_mention: str,
                         answer: str, links_out: list[str], removed_urls: list[str],
                         blocked_urls: Set[str], blocked_terms: Set[str], audit: Any, extra: Optional[dict]) -> None:
    enabled = _env_bool("RAG_DEBUG_LOG", False)
    if not enabled:
        return
    try:
        base = (os.getenv("PUBLIC_DIR") or "/public").rstrip("/")
        out_dir = Path(base) / "snapshot"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = (now_dt or datetime.utcnow()).strftime("%Y%m%d_%H%M%S")
        safe_intent = re.sub(r"[^a-z0-9_-]+", "_", (intent or "general").lower())
        out_path = out_dir / f"rag_debug_{ts}_{safe_intent}.json"
        chunk_summ = []
        for c in chunks[:10]:
            meta = getattr(c, "meta", None)
            url = None
            if isinstance(meta, dict):
                url = meta.get("url") or meta.get("source_url")
            chunk_summ.append({
                "id": _chunk_id(c),
                "url": url,
                "text_head": (getattr(c, "text", "") or "")[:140],
            })
        audit_obj = None
        if audit is not None:
            try:
                audit_obj = {
                    "passed": bool(getattr(audit, "passed", False)),
                    "score": getattr(audit, "score", None),
                    "confidence": getattr(audit, "confidence", None),
                    "issues": getattr(audit, "issues", None),
                }
            except Exception:
                audit_obj = None
        data = {
            "ts_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "intent": intent,
            "question": question,
            "now_block": now_block,
            "blocked_urls": sorted(blocked_urls),
            "blocked_terms": sorted(blocked_terms),
            "required_url": required_url,
            "required_mention": required_mention,
            "selected_chunks": chunk_summ,
            "answer": answer,
            "links": links_out,
            "removed_urls": removed_urls,
            "audit": audit_obj,
            "extra": extra or {},
        }
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"debug log write failed: {e}")
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


    # Common date formats seen in upstream data
    # - YYYY/MM/DD or YYYY/M/D
    m = re.fullmatch(r"(\d{4})/(\d{1,2})/(\d{1,2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            return None

    # - YYYY.MM.DD or YYYY.M.D
    m = re.fullmatch(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            return None

    # - Japanese: YYYY年M月D日
    m = re.fullmatch(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
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
        # Fallback: our generator always includes HINTS: topic_kind=...
        m2 = re.search(r"\btopic_kind\s*=\s*([a-zA-Z_]+)\b", q)
        if not m2:
            return None
        raw = (m2.group(1) or "").strip()
        return raw.lower() or None
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
    # Selected topic (for clients to build blocked_terms history)
    required_mention: str = ""
    required_url: Optional[str] = None
    intent: Optional[str] = None
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
    now_block = _now_block(request, payload.datetime)
    # Intent routing (light, deterministic)
    topic_family = _extract_topic_family(question)
    intent = _infer_intent(question, topic_family)
    wants_future_events = _wants_future_events(question) and (intent in ("event", "general"))
    strict_context = bool(getattr(payload, "strict_context", True))
    if intent == "event":
        strict_context = True
    # Normalize blocked URLs + terms
    blocked_urls: Set[str] = set()
    for u in (payload.blocked_urls or []):
        nu = normalize_url(u)
        if nu:
            blocked_urls.add(nu)
    blocked_terms: Set[str] = set()
    for t in (getattr(payload, "blocked_terms", None) or []):
        nt = _normalize_term(t)
        if nt:
            blocked_terms.add(nt)
    # Retrieve context (oversample when avoiding repeats or using variety)
    try:
        try:
            v = float(getattr(payload, "variety", 0.0) or 0.0)
        except Exception:
            v = 0.0
        anchor_top_n = int(getattr(payload, "anchor_top_n", 8) or 8)
        anchor_top_n = max(2, min(50, anchor_top_n))
        retrieval_k = top_k
        if blocked_urls or blocked_terms or v > 0:
            retrieval_k = max(retrieval_k, anchor_top_n, top_k * 5)
        retrieval_k = max(1, min(50, retrieval_k))
        chunks = rag_store.query_similar_chunks(question, top_k=retrieval_k)
        # Prefer chunks that do NOT match blocked items, but keep blocked hits as fallback.
        if (blocked_urls or blocked_terms) and chunks:
            unblocked: list[Any] = []
            blocked_hits: list[Any] = []
            for c in chunks:
                if _chunk_is_blocked(c, blocked_urls=blocked_urls, blocked_terms=blocked_terms):
                    blocked_hits.append(c)
                else:
                    unblocked.append(c)
            if unblocked:
                chunks = unblocked + blocked_hits
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
    # Optional: diversify topic by reordering chunks before trimming.
    if v > 0 and chunks:
        s_seed = getattr(payload, "seed", None)
        if s_seed is None:
            s_seed = int(now_dt.strftime("%Y%m%d%H")) if now_dt else 0
        top_n = max(2, min(anchor_top_n, len(chunks)))
        chunks = reorder_chunks_for_variety(chunks=chunks, seed=int(s_seed), top_n=top_n, variety=v)
    # Trim to requested context size
    if len(chunks) > top_k:
        chunks = chunks[:top_k]
    context_texts = [(getattr(c, "text", "") or "").strip() for c in chunks]
    context_texts = [t for t in context_texts if t]
    # links from data.json metadata (preferred for UI)
    source_links = collect_source_links(chunks=chunks, limit=64)
    # allow-list URLs
    allowed_urls: Set[str] = collect_allowed_urls(
        user_links=payload.links,
        chunk_links=source_links,
        context_texts=context_texts,
        extra_text=payload.extra_context,
    )
    # Remove blocked URLs from the allow-list (prevents reusing the same URL)
    if blocked_urls:
        allowed_urls = {u for u in allowed_urls if u not in blocked_urls}
        source_links = [u for u in source_links if u not in blocked_urls]
    required_mention, required_url = select_required_context(
        chunks=chunks,
        allowed_urls=allowed_urls,
    )
    # Phase 5: event date normalization (helps avoid 'past date' mistakes)
    extra_context = payload.extra_context
    event_dates_iso: list[str] = []
    if intent == "event":
        event_dates_iso = _extract_normalized_event_dates(chunks, now_dt=now_dt)
        block = _format_event_dates_block(event_dates_iso)
        extra_context = (extra_context or "").rstrip() + "\n\n" + block
    # max_chars default
    try:
        max_chars = int(payload.max_chars or 256)
    except Exception:
        max_chars = 256
    max_chars = max(80, min(400, max_chars))
    sys_prompt, user_prompt = build_chat_prompts(
        question=question,
        now_block=now_block,
        context_texts=context_texts,
        extra_context=extra_context,
        required_mention=required_mention,
        required_url=required_url,
        allowed_urls=allowed_urls,
        max_chars=max_chars,
    )
    # Light intent-specific instruction
    if intent == "event":
        user_prompt = (
            user_prompt
            + "\n\nEvent mode: If an event date is present in context, mention it in ISO format. "
            + "Do not mention dates in the past relative to NOW."
        )
    # Generation (+ optional audit loop) via LangGraph
    audit_enabled = _env_bool("RAG_AUDIT", False)
    rewrite_enabled = _env_bool("RAG_AUDIT_REWRITE", False)
    attempts = _env_int("RAG_AUDIT_MAX_ATTEMPTS", 2)
    attempts = max(1, min(5, attempts))
    gen_graph = _get_generation_graph()
    out = gen_graph.invoke(
        {
            "question": question,
            "sys_prompt": sys_prompt,
            "user_prompt": user_prompt,
            "required_mention": required_mention,
            "required_url": required_url,
            "allowed_urls": allowed_urls,
            "max_chars": max_chars,
            "now_dt": now_dt,
            "now_block": now_block,
            "wants_future_events": wants_future_events,
            "strict_context": strict_context,
            "include_debug": bool(payload.include_debug),
            "audit_enabled": audit_enabled,
            "rewrite_enabled": rewrite_enabled,
            "attempts": attempts,
            "attempt": 1,
            "removed_urls_total": [],
            "issues": [],
        }
    )
    answer = (out.get("answer") or "").strip()
    final_issues: list[str] = []
    if not answer:
        final_issues.append("empty answer")
    if "\n" in answer or "\r" in answer:
        final_issues.append("no line breaks (single paragraph)")
    sents = split_sentences(answer)
    if len(sents) != 3:
        final_issues.append("exactly 3 sentences")
    if not answer_mentions_required(answer, required_mention):
        final_issues.append("sentence 3 must mention required mention")
    if not third_sentence_is_substantive(answer, required_mention):
        final_issues.append("sentence 3 must include one concrete supported detail")
    if final_issues:
        raise HTTPException(status_code=502, detail="Generated answer failed final quality gate: " + "; ".join(final_issues))
    removed_urls_total = out.get("removed_urls_total") or []
    last_audit = out.get("last_audit")
    # Build links out (keep ONLY ONE link for UI output)
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
    primary_link = required_url or (links_out[0] if links_out else "")
    links_out = [primary_link] if primary_link else []
    debug: Optional[Dict[str, Any]] = None
    if payload.include_debug:
        debug = {
            "intent": intent,
            "event_dates_iso": event_dates_iso,
            "required_mention": required_mention,
            "required_url": required_url,
            "allowed_urls": sorted(allowed_urls),
            "source_links": source_links,
            "top_k": top_k,
            "blocked_urls": sorted(blocked_urls),
            "blocked_terms": sorted(blocked_terms),
        }
    # Optional debug artifact log (for GitHub Pages snapshot)
    _write_rag_debug_log(
        payload=payload,
        intent=intent,
        question=question,
        now_dt=now_dt,
        now_block=now_block,
        chunks=chunks,
        required_url=required_url,
        required_mention=required_mention,
        answer=answer,
        links_out=links_out,
        removed_urls=sorted(set(removed_urls_total)),
        blocked_urls=blocked_urls,
        blocked_terms=blocked_terms,
        audit=last_audit,
        extra=debug,
    )
    return QueryResponse(
        answer=answer,
        context=context_texts,
        links=links_out,
        removed_urls=sorted(set(removed_urls_total)),
        audit=last_audit,
        debug=debug,
    )