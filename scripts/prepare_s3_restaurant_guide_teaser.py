#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

try:
    from artifact_paths import artifact_feed_dir, artifact_latest_path, artifact_public_dir
except ImportError:
    from scripts.artifact_paths import artifact_feed_dir, artifact_latest_path, artifact_public_dir


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def normalize_rel_url(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if re.match(r"^[a-z][a-z0-9+.-]*://", s) or s.startswith("//"):
        return ""
    while s.startswith("./"):
        s = s[2:]
    return s.lstrip("/")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pick one guide at random from an existing S3-synced index.json and write a teaser feed entry"
        )
    )
    p.add_argument("--articles-index", required=True, help="Path to a generated guide index JSON")
    p.add_argument("--place", default="Yokosuka")
    p.add_argument("--avatar-image", default="image/avatar/spot.png")
    p.add_argument("--kind", default="guide")
    p.add_argument("--category", default="guide")
    p.add_argument("--link-title", default="Open this article")
    p.add_argument("--selection-mode", default="random_s3_guide_index")
    p.add_argument("--public-dir", default="")
    p.add_argument("--latest-path", default="")
    p.add_argument("--recent-exclude-limit", type=int, default=20)
    p.add_argument("--random-seed", default="")
    p.add_argument("--output-state", default="")
    return p


def recent_keys(feed_dir: Path, max_files: int) -> Set[str]:
    keys: Set[str] = set()
    if max_files <= 0 or not feed_dir.exists():
        return keys

    for path in sorted(feed_dir.glob("feed_*.json"), key=lambda p: p.name, reverse=True)[:max_files]:
        try:
            obj = load_json(path)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for raw in (
            obj.get("guide_id"),
            obj.get("guide_slug"),
            obj.get("article_slug"),
            obj.get("slug"),
            obj.get("permalink"),
            obj.get("title"),
        ):
            if isinstance(raw, str) and raw.strip():
                keys.add(raw.strip())
    return keys


def guide_keys(item: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for raw in (
        item.get("id"),
        item.get("slug"),
        item.get("permalink"),
        item.get("title"),
    ):
        if isinstance(raw, str) and raw.strip():
            out.add(raw.strip())
    return out


def choose_random_item(items: List[Dict[str, Any]], excluded: Set[str], seed: str) -> Dict[str, Any]:
    candidates = [it for it in items if not (guide_keys(it) & excluded)]
    pool = candidates or items
    if not pool:
        raise SystemExit("Guide index does not contain any selectable items")
    rng = random.Random(seed or None)
    return rng.choice(pool)


def main() -> int:
    args = build_parser().parse_args()

    public_dir = Path(args.public_dir) if str(args.public_dir).strip() else artifact_public_dir()
    latest_path = Path(args.latest_path) if str(args.latest_path).strip() else artifact_latest_path(public_dir)
    feed_dir = artifact_feed_dir(public_dir)
    articles_index_path = Path(args.articles_index)

    if not articles_index_path.is_file():
        raise SystemExit(f"Guide index JSON was not found: {articles_index_path}")

    payload = load_json(articles_index_path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Guide index JSON is not an object: {articles_index_path}")

    items_raw = payload.get("items")
    if not isinstance(items_raw, list):
        raise SystemExit(f"Guide index JSON does not contain an items[] array: {articles_index_path}")

    items = [it for it in items_raw if isinstance(it, dict) and str(it.get("title") or "").strip()]
    if not items:
        raise SystemExit(f"Guide index JSON contains no valid guide items: {articles_index_path}")

    excluded = recent_keys(feed_dir, args.recent_exclude_limit)
    chosen = choose_random_item(items, excluded, args.random_seed)

    now_local = local_stamp()
    feed_stem = f"feed_{now_local}"
    generated_at = utc_now_iso_z()

    guide_title = str(chosen.get("title") or "").strip()
    guide_short_title = str(chosen.get("short_title") or "").strip() or guide_title
    guide_description = str(chosen.get("description") or "").strip()
    guide_summary = str(chosen.get("summary") or "").strip() or guide_title
    guide_short_text = str(chosen.get("short_text") or "").strip()
    guide_place = str(chosen.get("place") or args.place).strip() or args.place
    guide_permalink = str(chosen.get("permalink") or "").strip()
    guide_date = str(chosen.get("date") or chosen.get("published_at") or "").strip() or utc_today()
    hero_image = str(chosen.get("hero_image") or "").strip()
    guide_text = guide_short_text or guide_description or guide_summary
    links = [{"title": args.link_title, "url": guide_permalink}] if guide_permalink else []

    item: Dict[str, Any] = {
        "id": feed_stem,
        "kind": args.kind,
        "date": guide_date,
        "generated_at": generated_at,
        "title": guide_title,
        "text": guide_text,
        "place": guide_place,
        "category": args.category,
        "summary": guide_summary,
        "description": guide_description,
        "permalink": guide_permalink,
        "links": links,
        "image": hero_image,
        "image_url": hero_image,
        "avatar_image": args.avatar_image,
        "short_title": guide_short_title,
        "short_text": guide_text,
        "guide_id": str(chosen.get("id") or "").strip(),
        "guide_slug": str(chosen.get("slug") or "").strip(),
        "guide_section": str(chosen.get("section") or payload.get("section") or "").strip(),
        "selection_mode": args.selection_mode,
        "recent_exclude_limit": args.recent_exclude_limit,
    }

    feed_path = feed_dir / f"{feed_stem}.json"
    write_json(feed_path, item)
    write_json(latest_path, item)

    state = {
        "feed_stem": feed_stem,
        "feed_path": str(feed_path),
        "latest_path": str(latest_path),
        "image_rel": normalize_rel_url(hero_image),
        "guide_id": item["guide_id"],
        "guide_slug": item["guide_slug"],
        "guide_section": item["guide_section"],
        "guide_permalink": guide_permalink,
        "category": args.category,
        "candidate_count": len(items),
        "excluded_recent_key_count": len(excluded),
    }

    if args.output_state:
        out = Path(args.output_state)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
