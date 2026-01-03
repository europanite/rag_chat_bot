#!/usr/bin/env python3
"""
Build static, paginated feed JSON files for "infinite scroll" on a static site (no backend).

- Reads individual feed snapshots: frontend/app/public/feed/feed_*.json
- Flattens/merges items into one timeline (newest first)
- Writes pages: frontend/app/public/feed/page_000.json, page_001.json, ...
- Updates frontend/app/public/latest.json to point to ./feed/page_000.json

Env:
  FEED_PATH        Base public dir (default: frontend/app/public)
  FEED_DIR         Feed dir (overrides FEED_PATH + "/feed")
  LATEST_PATH      Path to latest.json (default: FEED_PATH + "/latest.json")
  FEED_PAGE_SIZE   Page size (default: 20)

You can also pass CLI args to override.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def _item_key(item: Dict[str, Any]) -> str:
    """
    Sort key: prefer generated_at (ISO), else date (YYYY-MM-DD), else id.
    Newest first (descending).
    """
    ga = _safe_str(item.get("generated_at") or "")
    dt = _safe_str(item.get("date") or "")
    _id = _safe_str(item.get("id") or "")
    # Lexicographic works for ISO/YYYY-MM-DD.
    return ga or dt or _id


def _ensure_id(item: Dict[str, Any]) -> None:
    if isinstance(item.get("id"), str) and item["id"].strip():
        return
    base = (_safe_str(item.get("generated_at")) + "|" + _safe_str(item.get("date")) + "|" + _safe_str(item.get("text"))).encode("utf-8")
    h = hashlib.sha256(base).hexdigest()[:16]
    # Keep it stable-ish but readable
    item["id"] = f"auto_{h}"


def _ensure_permalink(item: Dict[str, Any]) -> None:
    if isinstance(item.get("permalink"), str) and item["permalink"].strip():
        return
    _ensure_id(item)
    _id = _safe_str(item.get("id"))
    if not _id:
        return
    item["permalink"] = f"./?post={urllib.parse.quote(_id, safe='')}"

def _extract_items(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept:
      - {"items": [...]}
      - a single item object
    """
    if isinstance(obj, dict):
        items = obj.get("items")
        if isinstance(items, list):
            out: List[Dict[str, Any]] = []
            for it in items:
                if isinstance(it, dict):
                    out.append(it)
            return out
        # single item
        if "date" in obj and ("text" in obj or "place" in obj):
            return [obj]  # type: ignore[list-item]
    return []


@dataclass
class BuildConfig:
    feed_dir: Path
    latest_path: Path
    page_size: int
    pattern: str = "feed_*.json"


def resolve_config(args: argparse.Namespace) -> BuildConfig:
    feed_path = os.getenv("FEED_PATH", "frontend/app/public")
    feed_dir_env = os.getenv("FEED_DIR", "")
    latest_path_env = os.getenv("LATEST_PATH", "")

    feed_dir = Path(args.feed_dir or feed_dir_env or (Path(feed_path) / "feed")).resolve()
    latest_path = Path(args.latest_path or latest_path_env or (Path(feed_path) / "latest.json")).resolve()

    page_size = int(args.page_size or os.getenv("FEED_PAGE_SIZE", "20"))
    if page_size <= 0:
        raise SystemExit("page_size must be >= 1")

    return BuildConfig(feed_dir=feed_dir, latest_path=latest_path, page_size=page_size)


def build_pages(cfg: BuildConfig) -> Tuple[int, int]:
    if not cfg.feed_dir.exists():
        raise SystemExit(f"Feed dir not found: {cfg.feed_dir}")

    src_files = sorted(
        [p for p in cfg.feed_dir.glob(cfg.pattern) if p.is_file() and p.suffix.lower() == ".json" and not p.name.startswith("page_")],
        key=lambda p: p.name,
    )

    items: List[Dict[str, Any]] = []
    for p in src_files:
        try:
            obj = _load_json(p)
        except Exception:
            continue
        for it in _extract_items(obj):
            if isinstance(it, dict):
                _ensure_id(it)
                _ensure_permalink(it)
                items.append(it)

    # Sort newest first
    items.sort(key=_item_key, reverse=True)

    # De-dupe by id (keep first/newest)
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for it in items:
        _id = _safe_str(it.get("id"))
        if not _id or _id in seen:
            continue
        seen.add(_id)
        deduped.append(it)

    total = len(deduped)
    if total == 0:
        # Still ensure a page_000.json exists for the frontend, even if empty.
        page0 = cfg.feed_dir / "page_000.json"
        _dump_json(page0, {"items": [], "page": 0, "page_size": cfg.page_size, "total_items": 0, "next_url": None})
        return 1, 0

    num_pages = (total + cfg.page_size - 1) // cfg.page_size

    # Write pages
    for i in range(num_pages):
        start = i * cfg.page_size
        end = min(start + cfg.page_size, total)
        page_items = deduped[start:end]

        updated_at = _safe_str(page_items[0].get("generated_at") or page_items[0].get("date") or "")
        place = page_items[0].get("place")

        next_url = f"page_{i+1:03d}.json" if i + 1 < num_pages else None
        page_obj: Dict[str, Any] = {
            "updated_at": updated_at or None,
            "place": place if isinstance(place, str) else None,
            "page": i,
            "page_size": cfg.page_size,
            "total_items": total,
            "next_url": next_url,
            "items": page_items,
        }
        out_path = cfg.feed_dir / f"page_{i:03d}.json"
        _dump_json(out_path, page_obj)

    # Remove stale page files beyond last page
    for p in cfg.feed_dir.glob("page_*.json"):
        m = p.stem.split("_")[-1]
        try:
            idx = int(m)
        except Exception:
            continue
        if idx >= num_pages:
            try:
                p.unlink()
            except Exception:
                pass

    return num_pages, total


def update_latest(latest_path: Path) -> None:
    try:
        obj = _load_json(latest_path)
        if not isinstance(obj, dict):
            obj = {}
    except Exception:
        obj = {}

    obj["feed_url"] = "./feed/page_000.json"
    obj["feed_file"] = "page_000.json"
    # Keep existing keys (place, updated_at, etc.) if present; frontend doesn't require them here.
    _dump_json(latest_path, obj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed-dir", dest="feed_dir", default="", help="Feed dir path (default: FEED_DIR or FEED_PATH/feed)")
    ap.add_argument("--latest-path", dest="latest_path", default="", help="Path to latest.json (default: LATEST_PATH or FEED_PATH/latest.json)")
    ap.add_argument("--page-size", dest="page_size", default="", help="Items per page (default: FEED_PAGE_SIZE or 20)")
    args = ap.parse_args()

    cfg = resolve_config(args)
    cfg.feed_dir.mkdir(parents=True, exist_ok=True)

    num_pages, total = build_pages(cfg)

    # Ensure latest.json exists
    cfg.latest_path.parent.mkdir(parents=True, exist_ok=True)
    # update_latest(cfg.latest_path)

    print(f"Built {num_pages} page(s) from {total} item(s)")
    print(f"Feed dir: {cfg.feed_dir}")
    print(f"Updated latest: {cfg.latest_path}")


if __name__ == "__main__":
    main()
