"""
LLM-based audit for answers produced by the RAG router.

The audit model is asked to validate:
- The answer stays within provided context.
- The answer only uses allowed URLs.
- It respects formatting/length constraints (best-effort).

This module is designed to be called from routers/rag.py.
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional, Set

from pydantic import BaseModel, Field

from .rag_utils import strip_broken_schemes


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class AuditLite(BaseModel):
    passed: bool = False
    score: int = Field(default=0, ge=0, le=100)
    confidence: str = "low"  # low|medium|high
    issues: List[str] = Field(default_factory=list)
    fixed_answer: Optional[str] = None
    raw: Optional[str] = None


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    # Strip code fences
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    m = _JSON_OBJECT_RE.search(t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build_audit_prompts(
    *,
    answer: str,
    question: str,
    now_block: str,
    allowed_urls: Set[str],
    required_url: str,
    strict_context: bool,
    allow_rewrite: bool,
    max_chars: int,
) -> tuple[str, str]:
    allowed_list = "\n".join(f"- {u}" for u in sorted(allowed_urls)) if allowed_urls else "(none)"
    rewrite_rule = (
        "If you can rewrite into a compliant answer, set fixed_answer to the rewritten answer.\n"
        if allow_rewrite
        else "Do not rewrite. Set fixed_answer to null.\n"
    )
    system_prompt = (
        "You are a strict QA auditor for a RAG answer.\n"
        "Return ONLY valid JSON.\n"
        "Schema:\n"
        "{\n"
        '  "passed": boolean,\n'
        '  "score": integer (0-100),\n'
        '  "confidence": "low"|"medium"|"high",\n'
        '  "issues": string[],\n'
        '  "fixed_answer": string|null\n'
        "}\n"
    )

    user_prompt = (
        f"{now_block}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Answer:\n{answer.strip()}\n\n"
        f"Constraints:\n"
        f"- max_chars: {max_chars}\n"
        f"- required_url: {required_url}\n"
        f"- strict_context: {strict_context}\n"
        f"- allowed_urls:\n{allowed_list}\n\n"
        f"Rules:\n"
        f"- The answer must include required_url.\n"
        f"- The answer must not include any URL outside allowed_urls.\n"
        f"- The answer must not contain broken fragments like '(https://)'.\n"
        f"- If strict_context is true, it must not introduce unsupported facts.\n"
        f"- If the question requests upcoming events, the answer must not mention a date earlier than today (based on now_block datetime).\n"
        f"{rewrite_rule}\n"
        "Now produce JSON."
    )
    return system_prompt, user_prompt


def run_answer_audit(
    *,
    call_chat_with_model: Callable[[str, str, str], str],
    model: str,
    answer: str,
    question: str,
    now_block: str,
    allowed_urls: Set[str],
    required_url: str,
    strict_context: bool,
    allow_rewrite: bool,
    max_chars: int,
) -> AuditLite:
    """
    call_chat_with_model(model, system_prompt, user_prompt) -> str
    """
    # Defensive cleanup before auditing
    cleaned_answer = strip_broken_schemes(answer)

    sys_prompt, user_prompt = build_audit_prompts(
        answer=cleaned_answer,
        question=question,
        now_block=now_block,
        allowed_urls=allowed_urls,
        required_url=required_url,
        strict_context=strict_context,
        allow_rewrite=allow_rewrite,
        max_chars=max_chars,
    )

    raw = ""
    try:
        raw = call_chat_with_model(model, sys_prompt, user_prompt) or ""
    except Exception as e:
        return AuditLite(
            passed=False,
            score=0,
            confidence="low",
            issues=[f"audit_call_failed: {e}"],
            fixed_answer=None,
            raw=str(e),
        )

    data = _extract_json(raw)
    if not isinstance(data, dict):
        return AuditLite(
            passed=False,
            score=0,
            confidence="low",
            issues=["audit_parse_failed"],
            fixed_answer=None,
            raw=raw,
        )

    passed = bool(data.get("passed", False))
    score = int(data.get("score", 0) or 0)
    confidence = str(data.get("confidence", "low") or "low").lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    issues = data.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    issues = [str(x) for x in issues if str(x).strip()]

    fixed_answer = data.get("fixed_answer", None)
    if fixed_answer is not None:
        fixed_answer = str(fixed_answer).strip()
        if fixed_answer == "" or fixed_answer.lower() == "null":
            fixed_answer = None

    return AuditLite(
        passed=passed,
        score=max(0, min(100, score)),
        confidence=confidence,
        issues=issues,
        fixed_answer=fixed_answer,
        raw=raw,
    )
