from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else str(value)


def _clean_rel(value: str) -> str:
    rel = str(value or "").strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    return rel.lstrip("/")


def _join_url(base: str, rel: str) -> str:
    base = str(base or "").strip().rstrip("/")
    rel = _clean_rel(rel)
    if not rel:
        return base
    if not base:
        return f"./{rel}"
    return f"{base}/{rel}"


def artifact_public_dir(public_dir: str | Path | None = None) -> Path:
    if public_dir is not None and str(public_dir).strip():
        return Path(public_dir)
    configured = _env("ARTIFACT_PUBLIC_DIR", "")
    if configured:
        return Path(configured)
    return Path(_env("FEED_PATH", "frontend/app/public"))


def artifact_feed_dir(public_dir: str | Path | None = None) -> Path:
    configured = _env("ARTIFACT_FEED_DIR", "")
    if configured:
        return Path(configured)
    return artifact_public_dir(public_dir) / "feed"


def artifact_image_dir(public_dir: str | Path | None = None) -> Path:
    configured = _env("ARTIFACT_IMAGE_DIR", "")
    if configured:
        return Path(configured)
    return artifact_public_dir(public_dir) / "image"


def artifact_snapshot_dir(
    public_dir: str | Path | None = None,
    latest_path: str | Path | None = None,
) -> Path:
    configured = _env("ARTIFACT_SNAPSHOT_DIR", "")
    if configured:
        return Path(configured)
    if latest_path is not None and str(latest_path).strip():
        return Path(latest_path).parent / "snapshot"
    return artifact_public_dir(public_dir) / "snapshot"


def artifact_latest_path(public_dir: str | Path | None = None) -> Path:
    configured = _env("ARTIFACT_LATEST_PATH", "")
    if configured:
        return Path(configured)
    return artifact_public_dir(public_dir) / "latest.json"


def computed_feed_snapshot_path(
    now_local: str,
    public_dir: str | Path | None = None,
) -> Path:
    return artifact_feed_dir(public_dir) / f"feed_{now_local}.json"


def resolve_public_feed_url(rel_path: str) -> str:
    rel = _clean_rel(rel_path)
    if not rel:
        return ""
    feed_base = _env("PUBLIC_FEED_BASE_URL", "")
    if feed_base:
        return _join_url(feed_base, rel)
    asset_base = _env("PUBLIC_ASSET_BASE_URL", "")
    if asset_base:
        return _join_url(asset_base, rel)
    return f"./{rel}"


def resolve_public_image_url(rel_path: str) -> str:
    rel = _clean_rel(rel_path)
    if not rel:
        return ""
    image_base = _env("PUBLIC_IMAGE_BASE_URL", "")
    if image_base:
        return _join_url(image_base, rel)
    asset_base = _env("PUBLIC_ASSET_BASE_URL", "")
    if asset_base:
        return _join_url(asset_base, rel)
    return f"./{rel}"


def resolve_public_avatar_url(rel_path: str) -> str:
    return resolve_public_image_url(rel_path)
diff --git a/scripts/build_feed_pages.py b/scripts/build_feed_pages.py
index 2cde0c1..9121650 100644
--- a/scripts/build_feed_pages.py
+++ b/scripts/build_feed_pages.py
@@ -1,8 +1,30 @@
 from __future__ import annotations
 
 import json
+import importlib.util
