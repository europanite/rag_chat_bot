from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class AuditResult(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    fixed_answer: Optional[str] = None

    # Debug fields (filled by caller when include_debug=True)
    original_answer: Optional[str] = None
    raw: Optional[str] = None


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text or "").strip()


def _parse_first_json_object(text: str) -> Optional[dict[str, Any]]:
    """
    Best-effort: find the first JSON object in free-form model output.
    Returns None if parsing fails.
    """
    s = _strip_code_fences(text)
    if not s:
        return None

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None


def _audit_prompt(
    question: str,
    answer: str,
    now_block: str,
    required_mention: str,
    required_url: str,
    allowed_urls: list[str],
    max_chars: int,
    output_style: str,
    audit_context: str,
    strict: bool,
) -> tuple[str, str]:
    sys = "\n".join(
        [
            "You are a strict JSON-only validator.",
            "Output ONLY a JSON object with keys: passed (bool), issues (array of strings), fixed_answer (string|null).",
            "Never include Markdown code fences.",
            "If you cannot fully fix, set fixed_answer to null.",
            "",
            audit_context or "",
        ]
    ).strip()

    # Keep the user prompt deterministic
    user = "\n".join(
        [
            now_block.strip(),
            "",
            f"[QUESTION]\n{question}\n[/QUESTION]",
            "",
            f"[ANSWER]\n{answer}\n[/ANSWER]",
            "",
            f"[REQUIREMENTS]",
            f"- Must mention: {required_mention}",
            f"- Must include URL exactly: {required_url}",
            f"- Only allowed URLs (if any): {', '.join(allowed_urls) if allowed_urls else '(none)'}",
            f"- Max chars: {max_chars}",
            f"- Output style: {output_style}",
            f"- Strict: {strict}",
            f"[/REQUIREMENTS]",
            "",
            "Return JSON only.",
        ]
    ).strip()

    return sys, user


def run_answer_audit(
    *,
    question: str,
    answer: str,
    now_block: str,
    required_mention: str,
    required_url: str,
    allowed_urls: list[str],
    max_chars: int,
    output_style: str,
    audit_context: str,
    strict: bool,
    rewrite: bool,
    audit_model: str,
    call_chat_with_model: Callable[[str, str, str], str],
) -> AuditResult:
    """
    Run audit against answer. If rewrite=True, auditor may attempt to produce fixed_answer.
    """
    sys, user = _audit_prompt(
        question=question,
        answer=answer,
        now_block=now_block,
        required_mention=required_mention,
        required_url=required_url,
        allowed_urls=allowed_urls,
        max_chars=max_chars,
        output_style=output_style,
        audit_context=audit_context,
        strict=strict,
    )

    raw = call_chat_with_model(audit_model, sys, user)
    obj = _parse_first_json_object(raw)

    issues: list[str] = []
    fixed: Optional[str] = None
    passed = False

    if obj:
        passed = bool(obj.get("passed", False))
        try:
            issues = [str(x) for x in (obj.get("issues") or [])]
        except Exception:
            issues = []
        if rewrite:
            fa = obj.get("fixed_answer")
            fixed = None if fa is None else str(fa)
    else:
        issues = ["Audit returned invalid JSON."]

    return AuditResult(passed=passed, issues=[str(x) for x in issues], fixed_answer=fixed, raw=raw)
