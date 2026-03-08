#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import mimetypes
import os
import random
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

USER_AGENT = "rag_chat_bot-gha/drive-random-post"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".svg"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".json"}
FOLDER_PATTERNS = [
    re.compile(r"https?://drive\.google\.com/drive/(?:u/\d+/)?folders/([A-Za-z0-9_-]+)"),
    re.compile(r"https?://drive\.google\.com/drive/mobile/folders/([A-Za-z0-9_-]+)"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]+)"),
]
FILE_PATTERNS = [
    re.compile(r"https?://drive\.google\.com/file/d/([A-Za-z0-9_-]+)"),
    re.compile(r"https?://docs\.google\.com/document/d/([A-Za-z0-9_-]+)"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]+)"),
]
RESOURCEKEY_RE = re.compile(r"[?&]resourcekey=([^&#]+)")
DOC_URL_RE = re.compile(r"https?://docs\.google\.com/document/d/([A-Za-z0-9_-]+)")
FILE_URL_RE = re.compile(r"https?://drive\.google\.com/file/d/([A-Za-z0-9_-]+)")


@dataclass
class FolderRef:
    folder_id: str
    resource_key: str = ""


@dataclass
class DriveEntry:
    name: str
    href: str
    file_id: str
    resource_key: str = ""
    kind: str = "unknown"  # image | text | doc | unknown


class AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._current_href = ""
        self._current_text: list[str] = []
        self.anchors: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        self._current_href = attr_map.get("href", "")
        self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a":
            return
        if self._current_href:
            text = html.unescape(" ".join(self._current_text)).strip()
            href = html.unescape(self._current_href).strip()
            self.anchors.append((href, text))
        self._current_href = ""
        self._current_text = []


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def request_text(url: str, *, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def request_bytes(url: str, *, timeout: int = 60) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), (resp.headers.get_content_type() or "application/octet-stream")


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    safe = safe.strip("._")
    return safe or "file"


def remove_suffixes(name: str) -> str:
    p = Path(name or "")
    stem = p.stem if p.suffix else (name or "")
    stem = stem.lower().strip()
    for suffix in (".caption", "_caption", "-caption", " caption", ".text", "_text", "-text", " text"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.strip("._ -")


def infer_ext(name: str, mime_type: str, default: str = ".bin") -> str:
    ext = Path(name or "").suffix.lower()
    if ext:
        return ext
    guess = mimetypes.guess_extension(mime_type or "")
    return guess or default


def extract_resource_key(source: str) -> str:
    m = RESOURCEKEY_RE.search(source or "")
    return urllib.parse.unquote(m.group(1)) if m else ""


def extract_folder_ref(source: str) -> FolderRef:
    s = (source or "").strip()
    if not s:
        raise SystemExit("Google Drive public folder URL is empty")
    for pat in FOLDER_PATTERNS:
        m = pat.search(s)
        if m:
            return FolderRef(folder_id=m.group(1), resource_key=extract_resource_key(s))
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s):
        return FolderRef(folder_id=s)
    raise SystemExit(f"Could not extract Google Drive folder ID from: {s}")


def extract_file_id(source: str) -> str:
    s = (source or "").strip()
    for pat in FILE_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s):
        return s
    raise SystemExit(f"Could not extract Google Drive file ID from: {s}")


def classify_entry(name: str, href: str) -> str:
    lower = (name or "").lower().strip()
    ext = Path(lower).suffix
    if ext in IMAGE_EXTS:
        return "image"
    if ext in TEXT_EXTS:
        return "text"
    if DOC_URL_RE.search(href):
        return "doc"
    return "unknown"


def normalize_href(href: str) -> str:
    if href.startswith("//"):
        return f"https:{href}"
    if href.startswith("/"):
        return f"https://drive.google.com{href}"
    return href


def list_public_folder_entries(folder: FolderRef) -> list[DriveEntry]:
    params = {"id": folder.folder_id}
    if folder.resource_key:
        params["resourcekey"] = folder.resource_key
    q = urllib.parse.urlencode(params)
    urls = [
        f"https://drive.google.com/embeddedfolderview?{q}#list",
        f"https://drive.google.com/drive/folders/{folder.folder_id}?{q}",
        f"https://drive.google.com/drive/mobile/folders/{folder.folder_id}?{q}",
    ]

    seen: set[tuple[str, str]] = set()
    out: list[DriveEntry] = []
    last_html = ""

    for url in urls:
        try:
            body = request_text(url)
        except Exception as exc:
            eprint(f"WARN: failed to fetch folder listing from {url}: {exc}")
            continue
        last_html = body
        parser = AnchorCollector()
        parser.feed(body)
        for href, text in parser.anchors:
            href2 = normalize_href(href)
            if not href2.startswith("http://") and not href2.startswith("https://"):
                continue
            if "drive.google.com" not in href2 and "docs.google.com" not in href2:
                continue
            if "/folders/" in href2:
                continue
            try:
                file_id = extract_file_id(href2)
            except SystemExit:
                continue
            name = text.strip() or Path(urllib.parse.urlparse(href2).path).name.strip()
            if not name:
                continue
            resource_key = extract_resource_key(href2) or folder.resource_key
            key = (file_id, name)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                DriveEntry(
                    name=name,
                    href=href2,
                    file_id=file_id,
                    resource_key=resource_key,
                    kind=classify_entry(name, href2),
                )
            )
        if out:
            break

    if not out:
        snippet = re.sub(r"\s+", " ", last_html[:500])
        raise SystemExit(
            "Could not enumerate files from the public Google Drive folder. "
            "Confirm the folder is shared as 'Anyone with the link' and that the URL is correct. "
            f"Fetched snippet: {snippet!r}"
        )
    return out


def choose_pairs(entries: Sequence[DriveEntry]) -> list[tuple[DriveEntry, DriveEntry]]:
    images: dict[str, list[DriveEntry]] = {}
    captions: dict[str, list[DriveEntry]] = {}

    for entry in entries:
        base = remove_suffixes(entry.name)
        if not base:
            continue
        if entry.kind == "image":
            images.setdefault(base, []).append(entry)
        elif entry.kind in {"text", "doc"}:
            captions.setdefault(base, []).append(entry)

    pairs: list[tuple[DriveEntry, DriveEntry]] = []
    for base in sorted(set(images) & set(captions)):
        img = sorted(images[base], key=lambda x: (Path(x.name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}, x.name))[0]
        cap = sorted(captions[base], key=lambda x: (x.kind != "doc", x.name))[0]
        pairs.append((img, cap))
    return pairs


def recent_source_image_ids(feed_dir: Path, limit: int) -> set[str]:
    if limit <= 0 or not feed_dir.exists():
        return set()
    ids: set[str] = set()
    files = sorted(feed_dir.glob("feed_*.json"), reverse=True)[:limit]
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = str(obj.get("source_image_id") or "").strip()
        if sid:
            ids.add(sid)
    return ids


def choose_random_pair(
    pairs: Sequence[tuple[DriveEntry, DriveEntry]],
    *,
    excluded_image_ids: set[str],
    random_seed: str,
) -> tuple[DriveEntry, DriveEntry]:
    if not pairs:
        raise SystemExit("No image+caption pairs were found in the public Google Drive folder")

    rng = random.Random()
    if random_seed:
        rng.seed(random_seed)
    else:
        rng.seed()

    fresh = [p for p in pairs if p[0].file_id not in excluded_image_ids]
    target = fresh if fresh else list(pairs)
    return rng.choice(target)


def build_download_url(entry: DriveEntry) -> str:
    params = {"export": "download", "id": entry.file_id}
    if entry.resource_key:
        params["resourcekey"] = entry.resource_key
    return "https://drive.google.com/uc?" + urllib.parse.urlencode(params)


def build_doc_export_url(entry: DriveEntry) -> str:
    params = {"format": "txt"}
    if entry.resource_key:
        params["resourcekey"] = entry.resource_key
    return f"https://docs.google.com/document/d/{entry.file_id}/export?" + urllib.parse.urlencode(params)


def download_entry_bytes(entry: DriveEntry) -> tuple[bytes, str]:
    url = build_download_url(entry)
    data, content_type = request_bytes(url)
    return data, content_type


def download_caption_text(entry: DriveEntry) -> str:
    url = build_doc_export_url(entry) if entry.kind == "doc" else build_download_url(entry)
    data, _ = request_bytes(url)
    return data.decode("utf-8", errors="replace")


def load_generate_talk_module(repo_root: Path):
    path = repo_root / "scripts" / "generate_talk.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("generate_talk", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def fallback_utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fallback_utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def fallback_local_stamp(_: str) -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def fetch_weather_snapshot_safe(gen: Any, *, lat: float, lon: float, tz_name: str, place: str) -> tuple[str, Dict[str, Any]]:
    if gen and hasattr(gen, "fetch_weather_snapshot"):
        try:
            return gen.fetch_weather_snapshot(lat=lat, lon=lon, tz_name=tz_name, place=place)
        except Exception as exc:
            eprint(f"WARN: fetch_weather_snapshot failed; continuing without weather: {exc}")
    return json.dumps({}, ensure_ascii=False, indent=2) + "\n", {}


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pick a random image+caption pair from a public Google Drive folder and write a feed entry")
    p.add_argument("--folder-url", required=True, help="Public Google Drive folder URL or folder ID")
    p.add_argument("--kind", default="spot")
    p.add_argument("--avatar-image", default="image/avatar/spot.png")
    p.add_argument("--place", default="Yokosuka")
    p.add_argument("--lat", default="35.2810")
    p.add_argument("--lon", default="139.6722")
    p.add_argument("--tz-name", default="Asia/Tokyo")
    p.add_argument("--public-dir", default="frontend/app/public")
    p.add_argument("--latest-path", default="frontend/app/public/latest.json")
    p.add_argument("--output-state", default="")
    p.add_argument("--recent-exclude-limit", type=int, default=20)
    p.add_argument("--random-seed", default="")
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    public_dir = Path(args.public_dir)
    latest_path = Path(args.latest_path)

    folder = extract_folder_ref(args.folder_url)
    entries = list_public_folder_entries(folder)
    pairs = choose_pairs(entries)
    excluded_ids = recent_source_image_ids(public_dir / "feed", args.recent_exclude_limit)
    image_entry, caption_entry = choose_random_pair(
        pairs,
        excluded_image_ids=excluded_ids,
        random_seed=args.random_seed,
    )

    image_bytes, image_mime = download_entry_bytes(image_entry)
    image_ext = infer_ext(image_entry.name, image_mime, default=".bin")
    if image_ext.lower() not in IMAGE_EXTS:
        raise SystemExit(
            f"Selected file does not look like an image after download: name={image_entry.name!r} mime={image_mime!r}"
        )

    caption_text = download_caption_text(caption_entry).strip()
    if not caption_text:
        raise SystemExit(f"Caption text is empty for selected caption file: {caption_entry.name}")

    gen = load_generate_talk_module(repo_root)
    utc_now_iso_z = getattr(gen, "utc_now_iso_z", fallback_utc_now_iso_z)
    utc_date = getattr(gen, "utc_date", fallback_utc_date)
    local_stamp = getattr(gen, "local_stamp", fallback_local_stamp)

    now_iso = utc_now_iso_z()
    today = utc_date()
    now_local = local_stamp(args.tz_name)
    feed_stem = f"feed_{now_local}"

    image_filename = f"{feed_stem}{image_ext.lower()}"
    image_path = public_dir / "image" / image_filename
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)

    snap_json_raw, snap_obj = fetch_weather_snapshot_safe(
        gen,
        lat=float(args.lat),
        lon=float(args.lon),
        tz_name=args.tz_name,
        place=args.place,
    )

    item = {
        "id": feed_stem,
        "date": today,
        "generated_at": now_iso,
        "text": caption_text,
        "place": args.place,
        "weather": snap_obj,
        "links": [],
        "kind": args.kind,
        "avatar_image": args.avatar_image,
        "image_url": f"./image/{image_filename}",
        "source_folder_id": folder.folder_id,
        "source_folder_url": args.folder_url,
        "source_image_name": image_entry.name,
        "source_image_id": image_entry.file_id,
        "source_caption_name": caption_entry.name,
        "source_caption_id": caption_entry.file_id,
        "selection_mode": "random_public_folder_pair",
        "recent_exclude_limit": args.recent_exclude_limit,
    }

    feed_path = public_dir / "feed" / f"{feed_stem}.json"
    write_json(feed_path, item)
    write_json(latest_path, item)

    snap_path = latest_path.parent / "snapshot" / f"snapshot_{now_local}.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(snap_json_raw, encoding="utf-8")

    state = {
        "feed_stem": feed_stem,
        "feed_path": str(feed_path),
        "latest_path": str(latest_path),
        "image_path": str(image_path),
        "image_rel": f"image/{image_filename}",
        "source_folder_id": folder.folder_id,
        "source_image_name": image_entry.name,
        "source_image_id": image_entry.file_id,
        "source_caption_name": caption_entry.name,
        "source_caption_id": caption_entry.file_id,
        "candidate_pair_count": len(pairs),
        "excluded_recent_image_id_count": len(excluded_ids),
    }
    if args.output_state:
        out = Path(args.output_state)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
