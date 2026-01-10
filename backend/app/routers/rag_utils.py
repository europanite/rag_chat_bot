from __future__ import annotations

import json
import re
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

# --- tweet hygiene ---
_GREET_RE = re.compile(r"^(hi|hello|good (morning|afternoon|evening))\b", re.I)
_META_PREFIX_RE = re.compile(r"^\s*here[’']?s a possible answer:\s*", re.I)
_META_LINE_RE = re.compile(
    r"^\s*(?:[\*\-]\s*)?(note:|i included\b|i also mentioned\b|the answer is within\b)\b",
    re.I,
)

def _strip_meta_preamble(text: str) -> str:
    a = (text or "").strip()
    a = _META_PREFIX_RE.sub("", a).strip()
    # unwrap one layer of quotes
    if len(a) >= 2 and a[0] == '"' and a[-1] == '"':
        a = a[1:-1].strip()
    # drop compliance chatter lines
    kept: List[str] = []
    for ln in a.splitlines():
        s = ln.strip()
        if not s:
            kept.append("")  # keep paragraph breaks
            continue
        if _META_LINE_RE.search(s):
            continue
        kept.append(ln)
    # collapse excessive blank lines
    out_lines: List[str] = []
    blank = 0
    for ln in kept:
        if ln.strip() == "":
            blank += 1
            if blank <= 1:
                out_lines.append("")
            continue
        blank = 0
        out_lines.append(ln.rstrip())
    return "\n".join(out_lines).strip()

def ensure_greeting_first(text: str, greeting: str = "Hello, everyone.") -> str:
    a = (text or "").strip()
    if not a:
        return greeting
    lines = a.splitlines()
    first = next((ln.strip() for ln in lines if ln.strip()), "")
    if first and _GREET_RE.match(first):
        return a
    return f"{greeting}\n{a}".strip()

# Matches URLs with a scheme. Keep it conservative to avoid capturing trailing punctuation.
_URL_RE = re.compile(r"https?://[^\s\)\]\}>\",']+")
# Matches "https://" or "http://" that is *not* followed by a normal URL character (broken scheme).
_BARE_SCHEME_RE = re.compile(r"(https?://)(?=($|[\s\)\]\}>\",']))")

# Common trailing punctuation we want to strip from URLs after regex extraction.
_URL_TRAIL_TRIM = ".,;:!?)>]\"'”’"


def truthy_env(val: Optional[str]) -> bool:
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def extract_urls_from_text(text: str) -> List[str]:
    if not text:
        return []
    urls = _URL_RE.findall(text)
    cleaned: List[str] = []
    for u in urls:
        cleaned.append(normalize_url(u))
    # Deduplicate preserving order
    seen = set()
    out = []
    for u in cleaned:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def normalize_url(url: str) -> str:
    """Normalize a URL for allow-list comparison."""
    if not url:
        return ""
    u = url.strip()
    # strip common trailing punctuation
    u = u.rstrip(_URL_TRAIL_TRIM)
    return u

def _links_from_meta_value(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # Might be JSON-encoded (because chroma metadata flattens)
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                obj = json.loads(s)
                return _links_from_meta_value(obj)
            except Exception:
                pass
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    if isinstance(val, list):
        out: List[str] = []
        for x in val:
            out.extend(_links_from_meta_value(x))
        return out
    if isinstance(val, dict):
        out: List[str] = []
        for kk in ("url", "href", "link", "permalink", "source_url", "sourceUrl"):
            vv = val.get(kk)
            if isinstance(vv, str) and vv.strip():
                out.append(vv.strip())
        return out
    return []

def collect_source_links(*, chunks: Sequence[object], limit: int = 64) -> List[str]:
    """
    Collect `metadata.links` from retrieved chunks (in rank order).
    """
    out: List[str] = []
    seen: Set[str] = set()
    for c in chunks:
        meta = getattr(c, "metadata", None) or {}
        for k in ("links", "link", "url", "permalink", "href", "source_url", "sourceUrl"):
            for raw in _links_from_meta_value(meta.get(k)):
                nu = normalize_url(raw)
                if not nu:
                    continue
                if not re.match(r"^https?://", nu, flags=re.I):
                    continue
                if nu in seen:
                    continue
                seen.add(nu)
                out.append(nu)
                if len(out) >= limit:
                    return out
    return out


def collect_allowed_urls(
    *,
    user_links: Optional[Sequence[str]] = None,
    chunk_links: Optional[Sequence[str]] = None,
    context_texts: Optional[Sequence[str]] = None,
    extra_text: Optional[str] = None,
    limit: int = 64,
) -> Set[str]:
    """
    Build an allow-list of URLs.

    user_links: explicitly requested links from the caller (highest priority).
    context_texts: retrieved RAG chunks.
    extra_text: live snapshot text (e.g., weather JSON).
    """
    urls: List[str] = []

    for u in (user_links or []):
        nu = normalize_url(u)
        if nu:
            urls.append(nu)

    for u in (chunk_links or []):
        nu = normalize_url(u)
        if nu:
            urls.append(nu)

    for t in (context_texts or []):
        urls.extend(extract_urls_from_text(t))

    if extra_text:
        urls.extend(extract_urls_from_text(extra_text))

    # Deduplicate but keep a stable order then return a set for fast membership.
    seen = set()
    ordered: List[str] = []
    for u in urls:
        if not u or u in seen:
            continue
        seen.add(u)
        ordered.append(u)
        if len(ordered) >= limit:
            break
    return set(ordered)


def filter_answer_urls(answer: str, allowed_urls: Set[str], *, keep_allowed: bool = True) -> Tuple[str, List[str]]:
    """
    Remove URLs from answer that are not in allowed_urls.
    Returns (filtered_answer, removed_urls).
    """
    if not answer:
        return answer, []
    allowed_norm = {normalize_url(u) for u in allowed_urls if u}
    removed: List[str] = []

    def _replace(match: re.Match) -> str:
        url = normalize_url(match.group(0))
        if keep_allowed and (url in allowed_norm):
            return url
        removed.append(url)
        return ""

    filtered = _URL_RE.sub(_replace, answer)

    # Clean doubled spaces created by removals
    filtered = re.sub(r"[ \t]{2,}", " ", filtered)
    filtered = re.sub(r"\n{3,}", "\n\n", filtered).strip()
    return filtered, removed


def strip_broken_schemes(text: str) -> str:
    """Remove broken standalone schemes like '(https://)'."""
    if not text:
        return text
    t = _BARE_SCHEME_RE.sub("", text)
    # also remove empty parentheses left behind: "( )", "()", "( )"
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\[\s*\]", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def select_required_context(
    *,
    chunks: Sequence[object],
    allowed_urls: Set[str],
) -> Tuple[str, str]:
    """
    Pick ONE required mention (string) and ONE required URL to enforce in the answer.

    We prefer:
    - a URL that appears in the top chunk's text or metadata
    - otherwise the first URL in allowed_urls
    """
    required_url = ""
    required_mention = ""

    # Best-effort extraction from first chunk.
    if chunks:
        first = chunks[0]
        meta = getattr(first, "metadata", None) or {}
        # mention priority: title/name/place
        for k in ("title", "name", "spot", "place", "event", "location"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                required_mention = v.strip()
                break

        # URL priority: metadata.links (or similar keys) -> text -> allowed_urls
        meta_urls: List[str] = []
        for k in ("links", "link", "url", "permalink", "href", "source_url", "sourceUrl"):
            meta_urls.extend(_links_from_meta_value(meta.get(k)))
        meta_urls = [normalize_url(u) for u in meta_urls if u]
        meta_urls = [u for u in meta_urls if re.match(r"^https?://", u, flags=re.I)]
        if meta_urls:
            required_url = meta_urls[0]

        text = getattr(first, "text", "") or ""
        urls_in_first = extract_urls_from_text(text)
        if urls_in_first:
            required_url = normalize_url(urls_in_first[0])

    if not required_url and allowed_urls:
        # stable pick: smallest string
        required_url = sorted(allowed_urls)[0]

    if not required_mention:
        # fallback mention: short snippet of first chunk
        if chunks:
            text = (getattr(chunks[0], "text", "") or "").strip()
            required_mention = (text.splitlines()[0] if text else "").strip()[:60]
        if not required_mention:
            required_mention = "the provided context"

    return required_mention, required_url


def build_chat_prompts(
    *,
    question: str,
    now_block: str,
    context_texts: Sequence[str],
    extra_context: Optional[str],
    required_mention: str,
    required_url: str,
    allowed_urls: Set[str],
    output_style: str = "tweet_bot",
    max_chars: int = 280,
) -> Tuple[str, str]:
    """
    Build (system_prompt, user_prompt).

    output_style:
      - "tweet_bot": concise social post; respects max_chars
      - "default": generic helpful answer
    """
    allowed_list = "\n".join(f"- {u}" for u in sorted(allowed_urls)) if allowed_urls else "(none)"

    style_rules = ""
    if output_style == "tweet_bot":
        style_rules = (
            f"- Output must be <= {max_chars} characters.\n"
            "- Output ONLY the post text (no preface, no quotes, no compliance notes).\n"
            "- The FIRST line must be a greeting (e.g., 'Hello, everyone.').\n"
            "- Do NOT include any URLs in the post text. Links are shown separately.\n"
            "- Keep it punchy and natural.\n"
            "- If you mention relative time words like 'tonight', they must match NOW.\n"
        )

    system_prompt = (
        "You are a careful assistant.\n"
        "You MUST answer using ONLY the provided context.\n"
        "Do NOT invent events, places, or dates.\n"
        "If you include any URL, it MUST be in the allow-list.\n"
        "Never output broken fragments like '(https://)'.\n"
    )

    # RAG context block
    rag_block = "\n\n".join(
        f"[{i+1}] {t.strip()}" for i, t in enumerate(context_texts) if t and t.strip()
    ).strip()

    user_prompt = (
        f"{now_block}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Required mention: {required_mention}\n"
        + (f"Required URL (for link section only): {required_url}\n\n" if output_style != "tweet_bot" else "\n")
        + (f"Allowed URLs:\n{allowed_list}\n\n" if output_style != "tweet_bot" else "")
        + f"RAG context:\n{rag_block}\n\n"
    )

    if extra_context and extra_context.strip():
        user_prompt += f"[Live context]\n{extra_context.strip()}\n\n"

    must_url = "" if output_style == "tweet_bot" else "- You MUST include the required URL.\n"
    url_rule = "- Do NOT include any URL in the post text.\n" if output_style == "tweet_bot" else "- Do NOT include any URL not in the allow-list.\n"
    user_prompt += (
        "Rules:\n"
        f"{style_rules}"
        f"- You MUST mention the required mention.\n"
        f"{must_url}"
        f"{url_rule}"
        "- If context is insufficient, say so briefly.\n\n"
        "Answer:"
    )
    return system_prompt, user_prompt


def finalize_answer(
    *,
    answer: str,
    required_mention: str,
    required_url: str,
    max_chars: int,
    output_style: str = "tweet_bot",
) -> str:
    """Post-process answer: strip broken schemes, enforce required mention/url, enforce length."""
    a = (answer or "").strip()
    a = strip_broken_schemes(a)

    if output_style == "tweet_bot":
        a = _strip_meta_preamble(a)
        a = ensure_greeting_first(a)

    if required_mention and required_mention.lower() not in a.lower():
        # Add a gentle mention; keep it short
        a = f"{a}\n\n({required_mention})".strip() if a else f"{required_mention}"

    # Hard cap (tweet_bot: no URL tail).

    if max_chars > 0 and len(a) > max_chars:
        a = a[:max_chars].rstrip()

    return a
