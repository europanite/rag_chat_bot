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


def build_audit_prompts(
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
) -> tuple[str, str]:
    strict_line = (
        "Be STRICT. If ANY requirement is violated, set passed=false and list concrete issues."
        if strict
        else "Be reasonable but accurate. If requirements are violated, set passed=false."
    )

    rewrite_line = (
        "If rewrite=true, ALSO produce a fixed_answer that fully satisfies all requirements."
        if rewrite
        else "Do NOT produce fixed_answer."
    )

    system_prompt = f"""You are an output auditor.

{strict_line}
Return ONLY valid JSON (no markdown).

Schema:
{{
  "passed": boolean,
  "issues": [string, ...],
  "fixed_answer": string|null
}}

Rules to check:
- The answer must mention: "{required_mention}"
- The answer must include this URL exactly: "{required_url}"
- All URLs must be from the allowed list (or be exactly the required_url).
- Max length: {max_chars} characters (count all characters).
- Output style: {output_style} (tweet_bot => single line, concise).
- Must not reference past dates as if they are current; use NOW as ground truth.
- If the answer is nonsensical, off-topic, or ignores RAG context, fail.

{rewrite_line}
"""

    allowed = "\n".join(f"- {u}" for u in allowed_urls) if allowed_urls else "(none)"
    user_prompt = f"""NOW:
{now_block}

QUESTION:
{question}

RAG CONTEXT (for grounding):
{audit_context}

ALLOWED URLs:
{allowed}

ANSWER TO AUDIT:
{answer}

rewrite={str(rewrite).lower()}
"""
    return system_prompt, user_prompt


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
    system_prompt, user_prompt = build_audit_prompts(
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
        rewrite=rewrite,
    )

    raw = call_chat_with_model(audit_model, system_prompt, user_prompt)
    obj = _parse_first_json_object(raw)

    if not obj:
        return AuditResult(passed=False, issues=["Auditor did not return valid JSON."], raw=raw)

    passed = bool(obj.get("passed", False))
    issues = obj.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]

    fixed = obj.get("fixed_answer", None)
    if fixed is not None and not isinstance(fixed, str):
        fixed = str(fixed)

    return AuditResult(passed=passed, issues=[str(x) for x in issues], fixed_answer=fixed, raw=raw)
