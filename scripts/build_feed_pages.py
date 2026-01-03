#!/usr/bin/env python3
"""
Build static, paginated feed JSON files for "infinite scroll" on GitHub Pages.

Key behavior:
- Reads many snapshot files: frontend/app/public/feed/feed_*.json
  (each snapshot is usually a single post object).
- Emits:
  - frontend/app/public/feed/page_000.json, page_001.json, ...
  - rewrites frontend/app/public/latest.json to a *pointer*:
      {"feed_url":"./feed/page_000.json","updated_at":"..."}
- IMPORTANT FIX (for images):
  For snapshot files named feed_*.json that represent ONE post, the filename stem is canonical.
  We:
    - set item.id = <stem> (so it starts with "feed_" and matches the image filename)
    - if image/image_url is missing but ./image/<stem>.png exists, we inject image_url.

This fixes cases where the post JSON has id=generated_at and no image_url, so the web UI
doesn't know which image file to load.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _item_fingerprint(it: Dict[str, Any]) -> str:
    date = str(it.get("date", "")).strip()
    text = str(it.get("text", "")).strip()
    return _sha1(f"{date}\n{text}")


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        x = s.strip()
        if not x:
            return None
        if x.endswith("Z"):
            x = x[:-1] + "+00:00"
        return datetime.fromisoformat(x)
    except Exception:
        return None


def _ensure_id(it: Dict[str, Any]) -> None:
    if isinstance(it.get("id"), str) and it["id"].strip():
        it["id"] = it["id"].strip()
        return
    date = str(it.get("date", "")).strip()
    text = str(it.get("text", "")).strip()
    it["id"] = f"auto_{_sha1(date + '|' + text)[:12]}"


def _ensure_permalink(it: Dict[str, Any]) -> None:
    if isinstance(it.get("permalink"), str) and it["permalink"].strip():
        it["permalink"] = it["permalink"].strip()
        return
    pid = str(it.get("id", "")).strip()
    if pid:
        it["permalink"] = f"./?post={pid}"


def _coerce_items(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept:
      - {"items":[...]}
      - [{...}, {...}]
      - {...} (single object entry)
    Return: list of item dicts.
    """
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        out: List[Dict[str, Any]] = []
        for it in obj["items"]:
            if isinstance(it, dict):
                out.append(it)
        return out

    if isinstance(obj, list):
        out = []
        for it in obj:
            if isinstance(it, dict):
                out.append(it)
        return out

    if isinstance(obj, dict):
        # Single entry object
        if isinstance(obj.get("date"), str) and isinstance(obj.get("text"), str):
            return [obj]

    return []


def _snapshot_is_single(obj: Any) -> bool:
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        return len([x for x in obj["items"] if isinstance(x, dict)]) == 1
    if isinstance(obj, list):
        return len([x for x in obj if isinstance(x, dict)]) == 1
    if isinstance(obj, dict) and isinstance(obj.get("date"), str) and isinstance(obj.get("text"), str):
        return True
    return False


def _apply_snapshot_canonicalization(
    it: Dict[str, Any],
    *,
    src_stem: str,
    public_dir: Path,
) -> None:
    """
    For feed_*.json snapshots (single-post), make them compatible with web image rules:
    - canonical id = filename stem (starts with 'feed_')
    - add image_url if missing and the image file exists
    """
    if not src_stem.startswith("feed_"):
        return

    # Canonicalize id to stem
    old_id = str(it.get("id", "")).strip()
    if old_id and old_id != src_stem and "legacy_id" not in it:
        it["legacy_id"] = old_id
    it["id"] = src_stem
    it["permalink"] = f"./?post={src_stem}"

    # Inject image_url if missing and file exists
    has_image = isinstance(it.get("image"), str) and it["image"].strip()
    has_image_url = isinstance(it.get("image_url"), str) and it["image_url"].strip()
    if not (has_image or has_image_url):
        img_path = public_dir / "image" / f"{src_stem}.png"
        if img_path.exists():
            it["image_url"] = f"./image/{src_stem}.png"


def _iter_sources(public_dir: Path) -> Iterable[Tuple[Path, str]]:
    feed_dir = public_dir / "feed"

    # Primary: snapshot files
    if feed_dir.exists():
        for p in sorted(feed_dir.glob("feed_*.json"), reverse=True):
            yield p, "snapshot"

    # Legacy / fallback sources
    for p in [
        public_dir / "output.json",
        public_dir / "feed.json",
        public_dir / "feed_latest.json",
        (public_dir / "feed") / "output.json",
        (public_dir / "feed") / "feed.json",
        (public_dir / "feed") / "feed_latest.json",
    ]:
        if p.exists():
            yield p, "legacy"


def main() -> int:
    public_dir = Path(os.environ.get("FEED_PATH", "frontend/app/public"))
    latest_path = Path(os.environ.get("LATEST_PATH", str(public_dir / "latest.json")))

    feed_dir = public_dir / "feed"
    feed_dir.mkdir(parents=True, exist_ok=True)

    page_size = int(os.environ.get("PAGE_SIZE", "30"))
    max_items = int(os.environ.get("MAX_ITEMS", "500"))

    items: List[Dict[str, Any]] = []
    seen_fp: set[str] = set()

    for src, kind in _iter_sources(public_dir):
        try:
            obj = _load_json(src)
        except Exception:
            continue

        src_stem = src.stem
        single = _snapshot_is_single(obj) and kind == "snapshot"

        for it in _coerce_items(obj):
            if not (isinstance(it.get("date"), str) and isinstance(it.get("text"), str)):
                continue

            # For single snapshot feed_*.json, canonicalize id + image_url using filename stem
            if single:
                _apply_snapshot_canonicalization(it, src_stem=src_stem, public_dir=public_dir)

            # Ensure minimum fields
            _ensure_id(it)
            _ensure_permalink(it)

            fp = _item_fingerprint(it)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            items.append(it)

    # Sort newest-first by generated_at if possible; else keep stable by id
    def sort_key(it: Dict[str, Any]) -> Tuple[int, str]:
        ga = str(it.get("generated_at", "")).strip()
        dt = _parse_iso(ga)
        if dt is None:
            return (0, str(it.get("id", "")))
        # larger is newer
        return (int(dt.timestamp()), str(it.get("id", "")))

    items.sort(key=sort_key, reverse=True)

    if max_items > 0:
        items = items[:max_items]

    # Write pages
    page_paths: List[Path] = []
    total_pages = (len(items) + page_size - 1) // page_size if items else 1

    for idx in range(total_pages):
        start = idx * page_size
        end = start + page_size
        page_items = items[start:end]

        page_obj: Dict[str, Any] = {"items": page_items}
        if idx + 1 < total_pages:
            page_obj["next_url"] = f"./page_{idx+1:03d}.json"

        out_path = feed_dir / f"page_{idx:03d}.json"
        _dump_json(out_path, page_obj)
        page_paths.append(out_path)

    # Clean old page files beyond current count
    for p in feed_dir.glob("page_*.json"):
        if p not in page_paths:
            try:
                p.unlink()
            except Exception:
                pass

    # Rewrite latest.json as a pointer to page_000.json
    updated_at = ""
    if items:
        ga = str(items[0].get("generated_at", "")).strip()
        updated_at = ga or str(items[0].get("date", "")).strip()
    if not updated_at:
        updated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    pointer = {"feed_url": "./feed/page_000.json", "updated_at": updated_at}
    _dump_json(latest_path, pointer)

    print(f"Wrote {len(page_paths)} pages, {len(items)} items. latest -> {pointer['feed_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
