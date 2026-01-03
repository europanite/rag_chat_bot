#!/usr/bin/env python3
"""
Generate an illustration image for the latest post and patch feed JSON(s) to reference it.

- Reads:  LATEST_PATH (default: frontend/app/public/latest.json)
- Writes: frontend/app/public/image/<feed_stem>.png
- Patches:
  - latest.json (entry fields)
  - feed snapshot file (feed/feed_<...>.json) in BOTH shapes:
      * {"items":[...]} (legacy)
      * {date,text,...} (current single-object)
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from diffusers import AutoPipelineForText2Image


MODEL_ID = os.environ.get("SD_MODEL_ID", "stabilityai/sd-turbo").strip() or "stabilityai/sd-turbo"
DEVICE = "cpu"  # GitHub Actions runner is typically CPU for this job


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_str(x: Any) -> str:
    return str(x) if x is not None else ""


def build_prompt(text: str, place: str) -> str:
    t = " ".join(text.split()).strip()[:240]
    p = place.strip()
    if p:
        return f"cinematic illustration, {p}, based on this short story: {t}"
    return f"cinematic illustration, based on this short story: {t}"


def _match_item(item: dict, *, date: str, text: str, generated_at: str) -> bool:
    if not isinstance(item, dict):
        return False
    same_dt = safe_str(item.get("date")).strip() == date and safe_str(item.get("text")).strip() == text
    same_ga = bool(generated_at) and safe_str(item.get("generated_at")).strip() == generated_at
    return same_dt or same_ga


def patch_feed_file(
    feed_path: Path,
    *,
    date: str,
    text: str,
    generated_at: str,
    feed_stem: str,
    rel_image_url: str,
    image_prompt: str,
    image_generated_at: str,
) -> bool:
    """
    Patch a feed JSON file that may be:
      - {"items":[...]} (legacy)
      - [{...}, {...}]  (rare)
      - {...}           (current snapshot single-object)
    """
    if not feed_path.exists():
        return False

    obj = load_json(feed_path)
    changed = False

    def apply_patch(it: dict) -> None:
        nonlocal changed
        if not _match_item(it, date=date, text=text, generated_at=generated_at):
            return
        # Canonicalize to feed-stem ID so web can apply stem-match rule.
        old_id = safe_str(it.get("id")).strip()
        if old_id and old_id != feed_stem and "legacy_id" not in it:
            it["legacy_id"] = old_id
        it["id"] = feed_stem
        it["permalink"] = f"./?post={feed_stem}"

        it["image_url"] = rel_image_url
        it["image_prompt"] = image_prompt
        it["image_model"] = MODEL_ID
        it["image_generated_at"] = image_generated_at
        changed = True

    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        for it in obj["items"]:
            if isinstance(it, dict):
                apply_patch(it)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                apply_patch(it)
    elif isinstance(obj, dict):
        # Current snapshot format: the object itself is the item
        apply_patch(obj)

    if changed:
        dump_json(feed_path, obj)
    return changed


def main() -> int:
    public_dir = Path(os.environ.get("FEED_PATH", "frontend/app/public"))
    latest_path = Path(os.environ.get("LATEST_PATH", str(public_dir / "latest.json")))

    if not latest_path.exists():
        print(f"ERROR: latest.json not found: {latest_path}")
        return 2

    latest = load_json(latest_path)
    if not isinstance(latest, dict):
        print("ERROR: latest.json is not an object")
        return 2

    date = safe_str(latest.get("date")).strip()
    text = safe_str(latest.get("text")).strip()
    place = safe_str(latest.get("place")).strip()
    generated_at = safe_str(latest.get("generated_at")).strip()

    if not date or not text:
        print("ERROR: latest.json missing date/text")
        return 2

    feed_dir = public_dir / "feed"
    feeds = sorted(feed_dir.glob("feed_*.json"), reverse=True)
    if not feeds:
        print(f"ERROR: No feed snapshots found in {feed_dir}")
        return 2

    # We name the image by the newest snapshot filename stem.
    feed_stem = feeds[0].stem

    out_dir = public_dir / "image"
    out_path = out_dir / f"{feed_stem}.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = random.randint(0, 2**31 - 1)
    prompt = build_prompt(text, place)

    print(f"MODEL_ID={MODEL_ID}")
    print(f"seed={seed}")
    print(f"feed_stem={feed_stem}")
    print(f"out_path={out_path}")
    print(f"prompt={prompt}")

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe = pipe.to(DEVICE)

    # SD-Turbo is fast; keep steps modest on CPU
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0, generator=generator).images[0]
    image.save(out_path)

    rel_image_url = f"./image/{out_path.name}"
    now_iso = now_iso_utc()

    # Patch latest.json entry
    old_latest_id = safe_str(latest.get("id")).strip()
    if old_latest_id and old_latest_id != feed_stem and "legacy_id" not in latest:
        latest["legacy_id"] = old_latest_id
    latest["id"] = feed_stem
    latest["permalink"] = f"./?post={feed_stem}"
    latest["image_url"] = rel_image_url
    latest["image_prompt"] = prompt
    latest["image_model"] = MODEL_ID
    latest["image_generated_at"] = now_iso
    dump_json(latest_path, latest)

    # Patch the newest snapshot feed file (current single-object or legacy items list)
    patched = patch_feed_file(
        feeds[0],
        date=date,
        text=text,
        generated_at=generated_at,
        feed_stem=feed_stem,
        rel_image_url=rel_image_url,
        image_prompt=prompt,
        image_generated_at=now_iso,
    )
    print(f"patched_snapshot={patched} path={feeds[0]}")

    # (Optional) Patch legacy aggregations if they exist
    for legacy in [
        public_dir / "feed.json",
        public_dir / "output.json",
        feed_dir / "feed.json",
        feed_dir / "output.json",
        feed_dir / "feed_latest.json",
    ]:
        if legacy.exists():
            ok = patch_feed_file(
                legacy,
                date=date,
                text=text,
                generated_at=generated_at,
                feed_stem=feed_stem,
                rel_image_url=rel_image_url,
                image_prompt=prompt,
                image_generated_at=now_iso,
            )
            print(f"patched_legacy={ok} path={legacy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
