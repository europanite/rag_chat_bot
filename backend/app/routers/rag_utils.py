from __future__ import annotations

import json
import re
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple
from datetime import datetime
from typing import Optional, Tuple

import math
import random
from typing import List, Sequence

# --- tweet hygiene ---
_META_PREFIX_RE = re.compile(r"^\s*here['']?s a possible answer:\s*", re.I)
_META_LINE_RE = re.compile(
    r"^\s*(?:[\*\-]\s*)?(note:|i included\b|i also mentioned\b|the answer is within\b)\b",
    re.I,
)

_GENERIC_INSUFFICIENCY_RE = re.compile(
    r"\b(?:not enough|insufficient|unable to determine|cannot determine|can't determine|"
    r"cannot tell|can't tell|no relevant context|provided context)\b",
    re.I,
)

def _normalize_match_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _required_mention_candidates(required_mention: str) -> List[str]:
    mention = (required_mention or "").strip()
    if not mention:
        return []

    candidates: List[str] = [mention]

    # Titles in the store are often formatted like:
    #   "English Name (日本語名)"
    #   "Category: Core Name (...)"
    # Allow the answer to use a stable alias instead of demanding the full title string.
    paren_head = re.split(r"\s*[\(（]\s*", mention, maxsplit=1)[0].strip(" )）-–—:：,")
    if paren_head:
        candidates.append(paren_head)

    parts = [p.strip(" )）-–—:：,") for p in re.split(r"\s*[:：|-]\s*", mention) if p.strip(" )）-–—:：,")]
    if parts:
        candidates.append(parts[0])
        candidates.append(parts[-1])

    deduped: List[str] = []
    seen: Set[str] = set()
    for cand in candidates:
        norm = _normalize_match_text(cand)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(cand)
    return deduped

def answer_mentions_required(answer: str, required_mention: str) -> bool:
    mention = (required_mention or "").strip()
    if not mention or mention.lower() == "the provided context":
        return True
    norm_answer = _normalize_match_text(answer)
    for cand in _required_mention_candidates(mention):
        if _normalize_match_text(cand) in norm_answer:
            return True
    return False

def split_sentences(answer: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (answer or "").strip()) if s.strip()]

def third_sentence_is_substantive(answer: str, required_mention: str) -> bool:
    sents = split_sentences(answer)
    if len(sents) < 3:
        return False
    third = sents[2].strip()
    if not answer_mentions_required(third, required_mention):
        return False
    if _GENERIC_INSUFFICIENCY_RE.search(third):
        return False
    norm_third = _normalize_match_text(third)
    norm_mention = _normalize_match_text(required_mention)
    if norm_mention:
        norm_third = norm_third.replace(norm_mention, "").strip()
    return len(norm_third) >= 8 and len(norm_third.split()) >= 2


def _chunk_has_mention(c: object) -> bool:
    meta = getattr(c, "metadata", None) or {}
    for k in ("title", "name", "spot", "place", "event", "location"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return True
    return False

def reorder_chunks_for_variety(
    *,
    chunks: Sequence[object],
    seed: int,
    top_n: int = 8,
    variety: float = 0.35,
) -> List[object]:
    """
    Pick an 'anchor' chunk from top-N (weighted) and move it to index 0.
    - variety=0.0: legacy behavior (no reorder)
    - variety>0 : more diverse anchors
    """
    if not chunks or variety <= 0:
        return list(chunks or [])

    n = min(len(chunks), max(1, int(top_n)))
    if n <= 1:
        return list(chunks)

    # Prefer chunks that actually have a usable mention in metadata,
    # but still allow non-mention chunks so variety can work even when metadata is sparse.
    candidate_idxs = list(range(n))

    # Rank-based weights (robust to distance scale)
    # temp grows with variety => flatter distribution
    temp = 0.25 + 8.0 * float(variety)   # 0.35 -> ~3.05
    weights = [math.exp(-i / temp) for i in range(n)]

    # Boost chunks that have a usable mention in metadata (helps link/mention selection),
    # without making variety collapse to a single chunk when only one has metadata.
    cand_weights: list[float] = []
    for i in candidate_idxs:
        w = weights[i]
        if _chunk_has_mention(chunks[i]):
            w *= 1.6
        cand_weights.append(w)

    rng = random.Random(int(seed))
    anchor_idx = rng.choices(candidate_idxs, weights=cand_weights, k=1)[0]

    out = list(chunks)
    anchor = out.pop(anchor_idx)
    out.insert(0, anchor)
    return out

def _time_greeting(now_dt: Optional[datetime]) -> str:
    """Return a time-of-day greeting. Never returns 'Hello, everyone.'."""
    if not now_dt:
        return "Good day."
    h = int(now_dt.hour)
    if 5 <= h <= 10:
        return "Good morning."
    if 11 <= h <= 16:
        return "Good afternoon."
    if 17 <= h <= 21:
        return "Good evening."
    return "Good night."

def _greeting_kind(line: str) -> Optional[str]:
    s = (line or "").strip().lower()
    if s.startswith("good morning"):
        return "morning"
    if s.startswith("good afternoon"):
        return "afternoon"
    if s.startswith("good evening"):
        return "evening"
    if s.startswith("good night"):
        return "night"
    if s.startswith("hi") or s.startswith("hello"):
        return "hello"
    return None

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

def ensure_greeting_first(
    text: str,
    *,
    now_dt: Optional[datetime] = None,
    greeting: Optional[str] = None,
) -> str:
    """Ensure first line is a time-appropriate greeting (not 'Hello, everyone.')."""
    desired = (greeting or _time_greeting(now_dt)).strip()
    a = (text or "").strip()
    if not a:
        return desired
    lines = a.splitlines()

    idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    first = lines[idx].strip() if idx is not None else ""
    kind = _greeting_kind(first)
    desired_kind = _greeting_kind(desired)  # morning/afternoon/evening/night

    # If it already starts with the correct time greeting, keep it.
    if kind and kind != "hello" and desired_kind and kind == desired_kind:
        return a

    # If it starts with *any* greeting (hello/hi or wrong time), replace that first greeting line.
    if kind:
        if idx is not None:
            lines[idx] = desired
            return "\n".join(lines).strip()
        return desired

    # No greeting at all -> prepend.
    return f"{desired} {a}".strip()


# Matches URLs with a scheme. Keep it conservative to avoid capturing trailing punctuation.
_URL_RE = re.compile(r"https?://[^\s\)\]\}>\",']+")
# Matches "https://" or "http://" that is *not* followed by a normal URL character (broken scheme).
_BARE_SCHEME_RE = re.compile(r"(https?://)(?=($|[\s\)\]\}>\",']))")

# Common trailing punctuation we want to strip from URLs after regex extraction.
_URL_TRAIL_TRIM = ".,;:!?)>]\"'”'"


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


def filter_answer_urls(answer: str, allowed_urls: Set[str]) -> Tuple[str, List[str]]:
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
        if url in allowed_norm:
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
    Pick ONE required mention and ONE required URL to enforce in the answer.
    IMPORTANT:
    Prefer picking mention+url from the SAME chunk whenever possible.
    This prevents "mention says A but link points to B" when retrieved chunks contain mixed sources.
    """
    required_url: str = ""
    required_mention: str = ""

    def _pick_mention_for_chunk(c: object) -> str:
        meta = getattr(c, "metadata", None) or {}
        for k in ("title", "name", "spot", "place", "event", "location", "restaurant", "venue", "topic", "summary"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()[:60]
        text = (getattr(c, "text", "") or "").strip()
        if text:
            return (text.splitlines()[0] or "").strip()[:60]
        return ""

    def _pick_url_for_chunk(c: object) -> str:
        meta = getattr(c, "metadata", None) or {}
        meta_urls: List[str] = []
        for k in ("links", "link", "url", "permalink", "href", "source_url", "sourceUrl"):
            meta_urls.extend(_links_from_meta_value(meta.get(k)))
        meta_urls = [normalize_url(u) for u in meta_urls if u]
        meta_urls = [u for u in meta_urls if re.match(r"^https?://", u, flags=re.I)]
        text_urls = [normalize_url(u) for u in extract_urls_from_text(getattr(c, "text", "") or "")]
        for u in (meta_urls + text_urls):
            if not u:
                continue
            if (not allowed_urls) or (u in allowed_urls):
                return u
        return ""

    # 1) Prefer the earliest chunk that has a usable URL; take mention from the SAME chunk.
    for c in (chunks or []):
        u = _pick_url_for_chunk(c)
        if u:
            required_url = u
            required_mention = _pick_mention_for_chunk(c)
            break

    # 2) If no URL was found in any chunk, fall back to allow-list (rare).
    if not required_url and allowed_urls:
        required_url = sorted(allowed_urls)[0]

    # 3) Ensure mention exists (fallback to the top chunk).
    if not required_mention:
        if chunks:
            required_mention = _pick_mention_for_chunk(chunks[0])
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
    max_chars: int = 280,
) -> Tuple[str, str]:
    """
    Build (system_prompt, user_prompt).
    """
    allowed_list = "\n".join(f"- {u}" for u in sorted(allowed_urls)) if allowed_urls else "(none)"

    style_rules = (
        f"- Output must be <= {max_chars} characters.\n"
        "- Output ONLY the post text (no preface, no quotes, no compliance notes).\n"
        "- The FIRST sentence must be a time-appropriate greeting that matches NOW "
        "(Good morning/Good afternoon/Good evening/Good night).\n"
        "- Do NOT include any URLs in the post text. Links are shown separately.\n"
        "- Use EXACTLY 3 sentences in a single paragraph (no line breaks): (1) Greeting. (2) Weather. (3) Main.\n"
        "- The SECOND sentence must include one of: sunny/cloudy/windy/chilly/rainy AND a temperature (e.g., 'cloudy with 7°C.').\n"
        "- Keep it punchy and natural.\n"
        "- The THIRD sentence must focus ONLY on the required mention; do NOT introduce a second place/event name.\n"
        "- If you mention relative time words like 'tonight', they must match NOW.\n"
    )
    
    system_prompt = (
            "You are a careful assistant.\n"
            "You MUST answer using ONLY the provided context.\n"
            "Do NOT invent events, places, or dates.\n"
            "Rules:\n"
            f"{style_rules}"
            "- You MUST mention the required mention provided in the user prompt in the THIRD sentence.\n"
            "- The THIRD sentence must include one concrete supported detail about that mention.\n"
            "- Do NOT output a generic insufficiency/apology line such as 'not enough context' or 'provided context'.\n"
            "- Never output only a greeting or only greeting + weather.\n"
    )

    # RAG context block
    rag_block = "\n\n".join(
        f"[{i+1}] {t.strip()}" for i, t in enumerate(context_texts) if t and t.strip()
    ).strip()

    user_prompt = (
        f"{now_block}\n\n"
        f"{question.strip()}\n\n"
        f"Required mention: {required_mention}\n\n"
        f"RAG context:\n{rag_block}\n\n"
    )

    if extra_context and extra_context.strip():
        user_prompt += f"[Live context]\n{extra_context.strip()}\n\n"

    must_url = ""  
    url_rule = "- Do NOT include any URL in the post text.\n"

    user_prompt += (
       "Rules:\n"
        f"{style_rules}"
        f"- You MUST mention the required mention in sentence 3.\n"
        "- Sentence 3 must include one concrete supported detail about that mention.\n"
        "- Do NOT write a generic insufficiency/apology line.\n"
        "- Never output only a greeting or only greeting + weather.\n"
        f"{must_url}"
        f"{url_rule}"
        "Answer:"
    )
    return system_prompt, user_prompt


def finalize_answer(
    *,
    answer: str,
    required_mention: str,
    max_chars: int,
    now_dt: Optional[datetime] = None,
) -> str:
    """Post-process answer: strip broken schemes, enforce required mention/url, enforce length."""
    a = (answer or "").strip()
    a = strip_broken_schemes(a)

    a = _strip_meta_preamble(a)
    a = ensure_greeting_first(a, now_dt=now_dt)

    if max_chars > 0 and len(a) > max_chars:
        a = a[:max_chars].rstrip()

    return a
