#!/usr/bin/env python3
"""
Convert a directory of Markdown files into a single JSON array for RAG ingestion.

python3 md_dir_to_json.py ../data/md ./output.json --recursive --source local-notes --id-prefix tourism_

Expected output schema (based on the user's sample):
[
  {
    "id": "unique_id",
    "source": "some-source",
    "metadata": { "lang": "en", "tags": ["tag1", "tag2"] },
    "text": "..."
  },
  ...
]

Features
- IDs are UNIQUE and *stable* across runs (derived from file path + content hash).
- Parses a structured Markdown format if present:
    # title / # datetime / # place / # detail / # attention / # link / # tags
  (each section contains bullet lines starting with "- ").
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
from typing import Dict, List, Tuple


SECTION_HEADERS = [
    "title",
    "datetime",
    "place",
    "detail",
    "attention",
    "link",
    "tags",
]


def contains_japanese(text: str) -> bool:
    # Hiragana, Katakana, CJK Unified Ideographs (Kanji)
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text))


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


def iter_markdown_files(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = sorted([p for p in input_dir.rglob("*.md") if p.is_file()])
    else:
        files = sorted([p for p in input_dir.glob("*.md") if p.is_file()])
    return files


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
        else:
            # Fallback: embed full markdown as-is
            text = md.strip()
            tags = []
            title = path.stem

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
            "metadata": {
                "lang": lang,
                "tags": tags,
            },
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
