#!/usr/bin/env python3
"""
Convert a directory of Markdown files into a single JSON array for RAG ingestion.

python3 md_dir_to_json.py ../data/md ./output.json --recursive --source local-notes --id-prefix tourism_

Expected output schema (flat fields; NO nested metadata):
[
  {
    "id": "unique_id",
    "source": "some-source",
    "lang": "en",
    "tags": ["tag1", "tag2"],
    "links": [{"label": "Event page", "url": "https://example.com"}, ...],
    "text": "..."
  },
  ...
]

Features
- IDs are UNIQUE and *stable* across runs (derived from file path + content hash).
- Parses a structured Markdown format if present:
    # title / # datetime / # place / # detail / # attention / # link / # tags
  (each section contains bullet lines starting with "- ").
- Extracts links from:
    - the "# link" section (if present)
    - Markdown links: [label](https://...)
    - Auto links: <https://...>
    - Bare URLs: https://...
  Links are de-duplicated (by URL) while preserving first-seen order.
- Falls back to using the entire Markdown content as "text" when the structure isn't found.

Usage
  python md_dir_to_json.py ./docs ./out.json --source local-notes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


SECTION_HEADERS = [
    "title",
    "datetime",
    "place",
    "detail",
    "attention",
    "link",
    "tags",
]


# --- language ---------------------------------------------------------------

def contains_japanese(text: str) -> bool:
    # Hiragana, Katakana, CJK Unified Ideographs (Kanji)
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text))


# --- IDs -------------------------------------------------------------------

def sanitize_stem(stem: str, max_len: int = 32) -> str:
    # Keep alnum, dash, underscore; replace others with underscore
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return (s[:max_len] if s else "doc")


def stable_id(rel_path: str, content: str, stem: str, prefix: str = "") -> str:
    h = hashlib.sha1()
    h.update(rel_path.encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(content.encode("utf-8", errors="ignore"))
    digest = h.hexdigest()[:12]
    base = f"{sanitize_stem(stem)}_{digest}"
    return f"{prefix}{base}"


# --- Markdown parsing -------------------------------------------------------

def split_sections(md: str) -> Dict[str, List[str]]:
    """
    Parse sections like:
      # title
      - xxx
      - yyy

    Returns:
      { "title": ["xxx"], "tags": ["a", "b"], ... }
    """
    # Normalize newlines
    md = md.replace("\r\n", "\n").replace("\r", "\n")

    # Find all headings of interest
    # Supports "# title" and "## title" etc.
    pat = re.compile(r"^(#{1,6})\s+([A-Za-z][A-Za-z0-9_-]*)\s*$", re.MULTILINE)

    hits: List[Tuple[str, int, int]] = []
    for m in pat.finditer(md):
        name = m.group(2).strip().lower()
        if name in SECTION_HEADERS:
            hits.append((name, m.start(), m.end()))

    if not hits:
        return {}

    # Add sentinel end
    hits_sorted = sorted(hits, key=lambda x: x[1])
    sections: Dict[str, List[str]] = {}
    for i, (name, start, end) in enumerate(hits_sorted):
        next_start = hits_sorted[i + 1][1] if i + 1 < len(hits_sorted) else len(md)
        body = md[end:next_start].strip("\n")

        # Collect bullet lines "- ..."
        items: List[str] = []
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:].strip())
        if items:
            sections[name] = items

    return sections


def build_text_from_sections(sections: Dict[str, List[str]]) -> str:
    # Render a compact plain-text block for embeddings.
    order = ["title", "datetime", "place", "detail", "attention", "link", "tags"]
    parts: List[str] = []

    for key in order:
        vals = sections.get(key)
        if not vals:
            continue

        if key == "title":
            parts.append(f"Title: {vals[0]}")
        elif key == "tags":
            parts.append("Tags: " + ", ".join(vals))
        else:
            # Join as bullet-like lines
            joined = "\n".join(f"- {v}" for v in vals)
            parts.append(f"{key.capitalize()}:\n{joined}")

    return "\n\n".join(parts).strip()


# --- Link extraction --------------------------------------------------------

_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_AUTOLINK_RE = re.compile(r"<(https?://[^>\s]+)>")
_BARE_URL_RE = re.compile(r"(?<!\()(?P<url>https?://[^\s<>\])}]+)")


def _strip_md_for_link_scan(md: str) -> str:
    """
    Remove fenced code blocks and inline code to avoid extracting example URLs.
    This is a heuristic; it won't be perfect, but it prevents many false positives.
    """
    s = md.replace("\r\n", "\n").replace("\r", "\n")
    s = _FENCED_CODE_BLOCK_RE.sub("", s)
    s = _INLINE_CODE_RE.sub("", s)
    return s


def _clean_url(url: str) -> str:
    # Trim common trailing punctuation that often appears in prose.
    return url.rstrip(").,;:!?'\"")


def _links_from_link_section(items: List[str]) -> List[Dict[str, str]]:
    """
    Turn '# link' section bullet items into link dicts.
    Supported patterns:
      - Label: https://...
      - https://...
    """
    out: List[Dict[str, str]] = []
    for raw in items:
        s = raw.strip()
        if not s:
            continue

        # Common "Label: URL" pattern
        m = re.match(r"^(?P<label>[^:]{1,80})\s*:\s*(?P<url>https?://\S+)\s*$", s)
        if m:
            out.append({"label": m.group("label").strip(), "url": _clean_url(m.group("url").strip())})
            continue

        # Otherwise: extract first URL from the line; keep the whole line as label only if it's not just the URL.
        m2 = re.search(r"https?://\S+", s)
        if m2:
            url = _clean_url(m2.group(0).strip())
            label = s
            if label == m2.group(0):
                label = ""
            out.append({"label": label, "url": url})
    return out


def extract_links(md: str, link_section_items: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Extract links from Markdown text and optional '# link' section items.
    De-duplicates by URL while preserving first-seen order.
    """
    seen = set()
    out: List[Dict[str, str]] = []

    def push(label: str, url: str) -> None:
        u = _clean_url(url.strip())
        if not u or u in seen:
            return
        seen.add(u)
        out.append({"label": (label or "").strip(), "url": u})

    # 1) '# link' section has priority (if present)
    if link_section_items:
        for d in _links_from_link_section(link_section_items):
            push(d.get("label", ""), d.get("url", ""))

    # 2) Scan the (cleaned) markdown body
    scan = _strip_md_for_link_scan(md)

    for m in _MD_LINK_RE.finditer(scan):
        push(m.group(1), m.group(2))

    for m in _AUTOLINK_RE.finditer(scan):
        push("", m.group(1))

    # Bare URLs (best-effort)
    for m in _BARE_URL_RE.finditer(scan):
        push("", m.group("url"))

    return out


# --- file iteration ---------------------------------------------------------

def iter_markdown_files(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = sorted([p for p in input_dir.rglob("*.md") if p.is_file()])
    else:
        files = sorted([p for p in input_dir.glob("*.md") if p.is_file()])
    return files


# --- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=str, help="Directory containing .md files")
    ap.add_argument("output_json", type=str, help="Output JSON file path")
    ap.add_argument("--source", type=str, default="markdown", help="Value for record['source']")
    ap.add_argument("--recursive", action="store_true", help="Search .md files recursively")
    ap.add_argument("--id-prefix", type=str, default="", help="Optional prefix for ids, e.g. 'tourism_'")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir not found or not a directory: {input_dir}")

    md_files = iter_markdown_files(input_dir, args.recursive)
    if not md_files:
        raise SystemExit(f"No .md files found in: {input_dir}")

    records = []
    seen_ids = set()

    for path in md_files:
        md = path.read_text(encoding="utf-8", errors="replace")
        rel_path = str(path.relative_to(input_dir))

        sections = split_sections(md)
        if sections:
            text = build_text_from_sections(sections)
            tags = sections.get("tags", [])
            title = sections.get("title", [path.stem])[0]
            link_section_items = sections.get("link", [])
        else:
            # Fallback: embed full markdown as-is
            text = md.strip()
            tags = []
            title = path.stem
            link_section_items = None

        links = extract_links(md, link_section_items=link_section_items)

        lang = "ja" if contains_japanese(md) else "en"

        doc_id = stable_id(rel_path, md, path.stem, prefix=args.id_prefix)
        # Guarantee uniqueness even if a (very unlikely) hash collision happens
        if doc_id in seen_ids:
            suffix = 2
            while f"{doc_id}_{suffix}" in seen_ids:
                suffix += 1
            doc_id = f"{doc_id}_{suffix}"
        seen_ids.add(doc_id)

        record = {
            "id": doc_id,
            "source": args.source,
            "lang": lang,
            "tags": tags,
            "links": links,
            "text": text if text else title,
        }
        records.append(record)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(records)} records -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
