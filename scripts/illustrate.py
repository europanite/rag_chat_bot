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
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from artifact_paths import (
        artifact_feed_dir,
        artifact_image_dir,
        artifact_latest_path,
        artifact_public_dir,
        resolve_public_avatar_url,
        resolve_public_image_url,
    )
except ImportError:
    from scripts.artifact_paths import (
        artifact_feed_dir,
        artifact_image_dir,
        artifact_latest_path,
        artifact_public_dir,
        resolve_public_avatar_url,
        resolve_public_image_url,
    )

import torch
from diffusers import AutoPipelineForText2Image


MODEL_ID = os.environ.get("SDXL_MODEL_ID", "stabilityai/sdxl-turbo").strip()
LORA_PATH = os.environ.get("LORA_PATH", "").strip()
LORA_SCALE = float(os.environ.get("LORA_SCALE", "0.8"))
PROMPT_TMPL = (os.environ.get("PROMPT") or "").strip()
NEGATIVE_TMPL = (os.environ.get("NEGATIVE") or "").strip()
STEPS = int(os.environ.get("STEPS", "6"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "2.0"))
SEED_OVERRIDE = (os.environ.get("SEED") or "").strip()
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))
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

def _render(t: str, *, place: str, core: str) -> str:
    s = (t or "").strip()
    if not s:
        return ""
    p = place or "Yokosuka, Japan"
    return s.replace("{place}", p).replace("{core}", core)

def build_prompt(text: str, place: str) -> tuple[str, str]:
    core = " ".join((text or "").split()).strip()[:240]
    p = (place or "").strip()
    if PROMPT_TMPL:
        prompt = _render(PROMPT_TMPL, place=p, core=core)
    else:
        prompt = f"cinematic illustration, {p}, based on this short story: {core}" if p else f"cinematic illustration, based on this short story: {core}"
    negative = _render(NEGATIVE_TMPL, place=p, core=core) if NEGATIVE_TMPL else ""
    return prompt, negative

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
    image_negative: str = "",
    image_model: str = MODEL_ID,
    image_generated_at: str,
    image_lora: str = "",
    image_lora_scale: float = 0.0,
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
        it["image"] = rel_image_url
        it["image_url"] = rel_image_url
        avatar_image = safe_str(it.get("avatar_image")).strip()
        if avatar_image:
            it["avatar_url"] = resolve_public_avatar_url(avatar_image)
        it["image_prompt"] = image_prompt
        it["image_negative"] = image_negative
        it["image_model"] = image_model
        if image_lora:
            it["image_lora"] = image_lora
            it["image_lora_scale"] = float(image_lora_scale)
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

def resolve_fixed_image_path(value: str, public_dir: Path) -> Path:
    """Resolve a fixed image path from latest.json (value may be absolute, repo-relative, or FEED_PATH-relative)."""
    v = (value or "").strip()
    p0 = Path(v)
    candidates = [
        p0,
        public_dir / v,
        public_dir / "image" / v,
        public_dir / "image" / "fixed" / v,
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    msg = " | ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"fixed image not found. tried: {msg}")

def main() -> int:
    public_dir = artifact_public_dir()
    latest_path = artifact_latest_path(public_dir)

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

    feed_dir = artifact_feed_dir(public_dir)
    feeds = sorted(feed_dir.glob("feed_*.json"), reverse=True)
    if not feeds:
        print(f"ERROR: No feed snapshots found in {feed_dir}")
        return 2

    # We name the image by the newest snapshot filename stem.
    feed_stem = feeds[0].stem

    out_dir = artifact_image_dir(public_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed = safe_str(latest.get("image_fixed")).strip()
    lora_tag = ""
    image_model = MODEL_ID

    if fixed:
        try:
            src = resolve_fixed_image_path(fixed, public_dir)
        except Exception as e:
            print(f"ERROR: {e}")
            return 2
        out_path = out_dir / f"{feed_stem}{(src.suffix or '.png')}"
        print(f"MODEL_ID={MODEL_ID}")
        print("mode=fixed_image")
        print(f"fixed_image={src}")
        print(f"feed_stem={feed_stem}")
        print(f"out_path={out_path}")
        shutil.copyfile(src, out_path)
        prompt = f"[fixed_image] {src.as_posix()}"
        negative = ""
        image_model = "fixed"
    else:
        out_path = out_dir / f"{feed_stem}.png"

        seed = int(SEED_OVERRIDE) if SEED_OVERRIDE.isdigit() else random.randint(0, 2**31 - 1)
        seed += SEED_OFFSET
        prompt,negative = build_prompt(text, place)

        print(f"MODEL_ID={MODEL_ID}")
        print("mode=text2img")
        print(f"seed={seed}")
        print(f"feed_stem={feed_stem}")
        print(f"out_path={out_path}")
        print(f"prompt={prompt}")
        print(f"negative={negative}")

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        pipe = pipe.to(DEVICE)

        image_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative,
            "num_inference_steps": int(STEPS),
            "guidance_scale": float(GUIDANCE_SCALE),
            "generator": generator,
        }

        if LORA_PATH:
            p = Path(LORA_PATH)
            if not p.exists():
                print(f"ERROR: LORA_PATH not found: {p}")
                return 2
            try:
                pipe.load_lora_weights(str(p))
                lora_tag = p.name
                image_kwargs["cross_attention_kwargs"] = {"scale": float(LORA_SCALE)}
            except Exception as e:
                print(f"ERROR: failed to load LoRA: {p} ({e})")
                return 2

        image = pipe(**image_kwargs).images[0]
        image.save(out_path)


    rel_image_url = resolve_public_image_url(f"image/{out_path.name}")
    now_iso = now_iso_utc()

    # Patch latest.json entry
    old_latest_id = safe_str(latest.get("id")).strip()
    if old_latest_id and old_latest_id != feed_stem and "legacy_id" not in latest:
        latest["legacy_id"] = old_latest_id
    latest["id"] = feed_stem
    latest["permalink"] = f"./?post={feed_stem}"
    latest["image"] = rel_image_url
    latest["image_url"] = rel_image_url
    avatar_image = safe_str(latest.get("avatar_image")).strip()
    if avatar_image:
        latest["avatar_url"] = resolve_public_avatar_url(avatar_image)
    latest["image_prompt"] = prompt
    latest["image_negative"] = negative
    latest["image_model"] = image_model
    if lora_tag:
        latest["image_lora"] = lora_tag
        latest["image_lora_scale"] = float(LORA_SCALE)
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
        image_negative=negative,
        image_model=image_model,
        image_generated_at=now_iso,
        image_lora=lora_tag,
        image_lora_scale=float(LORA_SCALE),
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
                    image_negative=negative,
                    image_model=image_model,
                    image_generated_at=now_iso,
                    image_lora=lora_tag,
                    image_lora_scale=LORA_SCALE,
            )
            print(f"patched_legacy={ok} path={legacy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
