#!/usr/bin/env python3
"""
Build static, paginated feed JSON files for "infinite scroll" UIs.

Input: a directory containing multiple feed snapshots like:

  feed/feed_20260102_123456_JST.json
  feed/feed_20260103_205844_JST.json
  ...

Each snapshot can be either:
  - a single item object {date,text,place,...}
  - a feed object {items:[...]}
  - a list of items [...]

Output:
  - feed/page_000.json, feed/page_001.json, ...
  - latest.json will be rewritten as a POINTER that points to the first page
    (e.g. {"feed_url":"./feed/page_000.json","updated_at":"...Z"})

This keeps the deployed app small and enables pagination while still keeping
per-run snapshot files for traceability.

Backward compatibility:
  - If you previously stored history in feed.json or output.json, this script
    will also read those (both in <public>/ and <public>/feed/) so older history
    doesn't "disappear" after migrating to feed_*.json snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _item_key(item: Dict[str, Any]) -> str:
    # Prefer explicit id if present; otherwise use a hashable signature.
    if _safe_str(item.get("id")):
        return _safe_str(item.get("id"))

    sig = "|".join(
        [
            _safe_str(item.get("generated_at")),
            _safe_str(item.get("date")),
            _safe_str(item.get("place")),
            _safe_str(item.get("text")),
        ]
    ).strip()
    if not sig:
        sig = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


def _ensure_id(item: Dict[str, Any]) -> None:
    if not _safe_str(item.get("id")):
        item["id"] = _item_key(item)


def _ensure_permalink(item: Dict[str, Any]) -> None:
    # Use stable id (or computed key) as permalink token.
    _ensure_id(item)
    if not _safe_str(item.get("permalink")):
        item["permalink"] = f"./?post={_safe_str(item.get('id'))}"


def _extract_items(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept multiple historical formats:
      - {"items":[...]}
      - [...]
      - {"date":..., "text":...} (single item)
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        if isinstance(obj.get("items"), list):
            return [x for x in obj["items"] if isinstance(x, dict)]
        # single item object
        if any(k in obj for k in ("date", "text", "place", "generated_at")):
            return [obj]  # type: ignore[list-item]
    return []


@dataclass
class BuildConfig:
    feed_path: Path
    feed_dir: Path
    latest_path: Path
    page_size: int
    pattern: str = "feed_*.json"


def resolve_config(args: argparse.Namespace) -> BuildConfig:
    feed_path = os.getenv("FEED_PATH", "frontend/app/public")
    feed_dir = os.getenv("FEED_DIR", str(Path(feed_path) / "feed"))
    latest_path = os.getenv("LATEST_PATH", str(Path(feed_path) / "latest.json"))
    page_size = int(os.getenv("PAGE_SIZE", str(args.page_size)))

    return BuildConfig(
        feed_path=Path(feed_path).resolve(),
        feed_dir=Path(feed_dir).resolve(),
        latest_path=Path(latest_path).resolve(),
        page_size=page_size,
    )


def _iter_source_files(cfg: BuildConfig) -> List[Path]:
    """Return JSON files that can contain historical feed items.

    We support both:
    - Snapshot files: feed_*.json
    - Legacy aggregated feeds (kept for backward compatibility):
      - <feed_dir>/feed.json, <feed_dir>/output.json
      - <feed_path>/feed.json, <feed_path>/output.json

    This prevents older history from disappearing when you switch formats.
    """
    files: List[Path] = []

    # 1) Legacy aggregated feeds (optional)
    legacy_paths = {
        cfg.feed_dir / "feed.json",
        cfg.feed_dir / "output.json",
        cfg.feed_path / "feed.json",
        cfg.feed_path / "output.json",
    }
    for p in sorted(legacy_paths, key=lambda x: x.as_posix()):
        if p.is_file() and p.suffix.lower() == ".json":
            files.append(p)

    # 2) Snapshot files (newer runs) - ignore generated pagination pages
    snaps = [
        p
        for p in cfg.feed_dir.glob(cfg.pattern)
        if p.is_file() and p.suffix.lower() == ".json" and not p.name.startswith("page_")
    ]
    snaps_sorted = sorted(snaps, key=lambda p: p.name)
    files.extend(snaps_sorted)

    # De-dupe identical paths while keeping order
    seen: set[str] = set()
    out: List[Path] = []
    for p in files:
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def build_pages(cfg: BuildConfig) -> Tuple[int, int]:
    cfg.feed_dir.mkdir(parents=True, exist_ok=True)

    src_files = _iter_source_files(cfg)
    if not src_files:
        # Ensure at least page_000.json exists for the app.
        page0 = cfg.feed_dir / "page_000.json"
        _dump_json(page0, {"items": [], "next_url": None})
        return 1, 0

    # Load items from ALL sources (legacy + snapshots)
    items: List[Dict[str, Any]] = []
    for p in src_files:
        try:
            obj = _load_json(p)
        except Exception as e:
            print(f"WARN: skipping invalid json: {p} err={e}")
            continue

        for item in _extract_items(obj):
            # Normalize: always set id + permalink.
            _ensure_id(item)
            _ensure_permalink(item)
            items.append(item)

    # De-dupe by id/key (keep newest first)
    # Sort newest -> oldest by a timestamp heuristic.
    def _sort_key(it: Dict[str, Any]) -> str:
        # Prefer generated_at; else date; else empty.
        return _safe_str(it.get("generated_at")) or _safe_str(it.get("date")) or ""

    items_sorted = sorted(items, key=_sort_key, reverse=True)

    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for it in items_sorted:
        k = _item_key(it)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(it)

    total = len(deduped)
    if total == 0:
        page0 = cfg.feed_dir / "page_000.json"
        _dump_json(page0, {"items": [], "next_url": None})
        return 1, 0

    # Write page_000, page_001, ...
    num_pages = (total + cfg.page_size - 1) // cfg.page_size

    for page_idx in range(num_pages):
        start = page_idx * cfg.page_size
        end = min(start + cfg.page_size, total)
        page_items = deduped[start:end]

        page_name = f"page_{page_idx:03d}.json"
        page_path = cfg.feed_dir / page_name

        next_url: Optional[str] = None
        if page_idx + 1 < num_pages:
            next_url = f"page_{page_idx + 1:03d}.json"

        _dump_json(page_path, {"items": page_items, "next_url": next_url})

    # Remove stale pages beyond current count
    for p in sorted(cfg.feed_dir.glob("page_*.json")):
        try:
            idx = int(p.stem.split("_", 1)[1])
        except Exception:
            continue
        if idx >= num_pages:
            try:
                p.unlink()
            except Exception:
                pass

    # Write pointer file for the app (HomeScreen follows feed_url/feed_path/feed_file).
    # This ensures the app can fetch ./latest.json and still show the whole history.
    page0 = cfg.feed_dir / "page_000.json"
    try:
        rel = page0.relative_to(cfg.feed_path).as_posix()
        feed_url = f"./{rel}"
    except Exception:
        # Fallback: common layout (<public>/feed/page_000.json)
        feed_url = "./feed/page_000.json"

    now_utc = datetime.utcnow().isoformat() + "Z"
    _dump_json(cfg.latest_path, {"feed_url": feed_url, "updated_at": now_utc})

    return num_pages, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-size", type=int, default=20, help="Items per page (default 20)")
    args = ap.parse_args()

    cfg = resolve_config(args)
    num_pages, total = build_pages(cfg)
    print(f"OK: wrote {num_pages} pages with total={total} items. latest={cfg.latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
