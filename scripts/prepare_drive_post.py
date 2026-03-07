#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import mimetypes
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

DRIVE_FILE_PATTERNS = [
    re.compile(r"https?://drive\.google\.com/file/d/([A-Za-z0-9_-]+)"),
    re.compile(r"https?://docs\.google\.com/document/d/([A-Za-z0-9_-]+)"),
    re.compile(r"https?://docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]+)"),
    re.compile(r"https?://docs\.google\.com/presentation/d/([A-Za-z0-9_-]+)"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]+)"),
]
DOC_URL_RE = re.compile(r"https?://docs\.google\.com/document/d/([A-Za-z0-9_-]+)")
DRIVE_URL_RE = re.compile(r"https?://(?:drive|docs)\.google\.com/", re.IGNORECASE)
TEXT_LIKE_MIMES = {
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/json",
    "application/vnd.google-apps.document",
}
IMAGE_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tif",
    "image/svg+xml": ".svg",
}


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    safe = safe.strip("._")
    return safe or "file"


def remove_suffixes(name: str) -> str:
    stem = Path(name).stem.lower().strip()
    for suffix in (".caption", "_caption", "-caption", " caption", "_text", "-text", " text"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.strip("._ -")


def infer_ext(name: str, mime_type: str, default: str = ".bin") -> str:
    ext = Path(name or "").suffix.lower()
    if ext:
        return ext
    if mime_type in IMAGE_EXT_BY_MIME:
        return IMAGE_EXT_BY_MIME[mime_type]
    guess = mimetypes.guess_extension(mime_type or "")
    return guess or default


def direct_fetch_bytes(url: str) -> Tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "rag_chat_bot-gha"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        content_type = resp.headers.get_content_type() or "application/octet-stream"
        return data, content_type


def load_generate_talk_module(repo_root: Path):
    path = repo_root / "scripts" / "generate_talk.py"
    spec = importlib.util.spec_from_file_location("generate_talk", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class DriveAccessor:
    def __init__(self, *, service_account_json: str = "", service_account_json_path: str = "") -> None:
        self.service = None
        self._auth_enabled = False
        self._build_service(service_account_json=service_account_json, service_account_json_path=service_account_json_path)

    @property
    def auth_enabled(self) -> bool:
        return self._auth_enabled

    def _build_service(self, *, service_account_json: str, service_account_json_path: str) -> None:
        raw = (service_account_json or "").strip()
        path = (service_account_json_path or "").strip()
        if not raw and path:
            raw = Path(path).read_text(encoding="utf-8")
        if not raw:
            return

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except Exception as e:
            raise SystemExit(
                "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON is set, but google-api-python-client/google-auth is not installed"
            ) from e

        info = json.loads(raw)
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        self.service = build("drive", "v3", credentials=creds, cache_discovery=False)
        self._auth_enabled = True

    def is_drive_like(self, source: str) -> bool:
        s = (source or "").strip()
        if not s:
            return False
        if DRIVE_URL_RE.search(s):
            return True
        return bool(re.fullmatch(r"[A-Za-z0-9_-]{10,}", s))

    def extract_file_id(self, source: str) -> str:
        s = (source or "").strip()
        if not s:
            raise SystemExit("Google Drive source is empty")
        for pat in DRIVE_FILE_PATTERNS:
            m = pat.search(s)
            if m:
                return m.group(1)
        if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s):
            return s
        raise SystemExit(f"Could not extract Google Drive file ID from: {s}")

    def _public_download_url(self, source: str, file_id: str, *, force_text_export: bool = False) -> str:
        s = (source or "").strip()
        if force_text_export or DOC_URL_RE.search(s):
            return f"https://docs.google.com/document/d/{file_id}/export?format=txt"
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    def get_metadata(self, source: str) -> Dict[str, Any]:
        file_id = self.extract_file_id(source)
        if not self.auth_enabled:
            return {"id": file_id, "name": file_id, "mimeType": "application/octet-stream", "parents": []}
        meta = (
            self.service.files()
            .get(
                fileId=file_id,
                fields="id,name,mimeType,parents,modifiedTime,webViewLink",
                supportsAllDrives=True,
            )
            .execute()
        )
        return dict(meta or {})

    def download_binary(self, source: str) -> Tuple[bytes, Dict[str, Any]]:
        s = (source or "").strip()
        if s.startswith("http://") or s.startswith("https://"):
            if not self.is_drive_like(s):
                data, content_type = direct_fetch_bytes(s)
                name = sanitize_filename(Path(urllib.parse.urlparse(s).path).name or "download")
                return data, {"id": "", "name": name, "mimeType": content_type, "parents": []}

        file_id = self.extract_file_id(s)
        meta = self.get_metadata(s)

        if self.auth_enabled:
            from googleapiclient.http import MediaIoBaseDownload

            mime_type = str(meta.get("mimeType") or "")
            if mime_type.startswith("application/vnd.google-apps"):
                raise SystemExit(f"Image source must be a binary image file, but got Google Workspace type: {mime_type}")

            request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buf.getvalue(), meta

        data, content_type = direct_fetch_bytes(self._public_download_url(s, file_id))
        meta["mimeType"] = content_type or meta.get("mimeType") or "application/octet-stream"
        return data, meta

    def download_text(self, source: str) -> Tuple[str, Dict[str, Any]]:
        s = (source or "").strip()
        if not s:
            raise SystemExit("Caption source is empty")

        if s.startswith("http://") or s.startswith("https://"):
            if not self.is_drive_like(s):
                data, content_type = direct_fetch_bytes(s)
                name = sanitize_filename(Path(urllib.parse.urlparse(s).path).name or "caption.txt")
                return data.decode("utf-8", errors="replace"), {
                    "id": "",
                    "name": name,
                    "mimeType": content_type,
                    "parents": [],
                }

        file_id = self.extract_file_id(s)
        meta = self.get_metadata(s)

        if self.auth_enabled:
            from googleapiclient.http import MediaIoBaseDownload

            mime_type = str(meta.get("mimeType") or "")
            if mime_type == "application/vnd.google-apps.document":
                request = self.service.files().export_media(fileId=file_id, mimeType="text/plain")
            elif mime_type in TEXT_LIKE_MIMES or mime_type.startswith("text/"):
                request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
            else:
                raise SystemExit(f"Caption file must be text or Google Docs, but got: {mime_type}")

            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buf.getvalue().decode("utf-8", errors="replace"), meta

        download_url = self._public_download_url(s, file_id, force_text_export=bool(DOC_URL_RE.search(s)))
        data, content_type = direct_fetch_bytes(download_url)
        meta["mimeType"] = content_type or meta.get("mimeType") or "text/plain"
        return data.decode("utf-8", errors="replace"), meta

    def find_sidecar_caption(self, image_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.auth_enabled:
            return None

        parents = image_meta.get("parents") or []
        if not parents:
            return None

        image_id = str(image_meta.get("id") or "")
        image_name = str(image_meta.get("name") or "")
        base = remove_suffixes(image_name)
        if not base:
            return None

        parent = parents[0]
        q = (
            f"'{parent}' in parents and trashed = false and "
            f"mimeType != 'application/vnd.google-apps.folder'"
        )
        resp = (
            self.service.files()
            .list(
                q=q,
                fields="files(id,name,mimeType,modifiedTime)",
                pageSize=200,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = list((resp or {}).get("files") or [])

        def score(meta: Dict[str, Any]) -> int:
            fid = str(meta.get("id") or "")
            name = str(meta.get("name") or "")
            mime_type = str(meta.get("mimeType") or "")
            if fid == image_id:
                return -1
            if not (mime_type.startswith("text/") or mime_type in TEXT_LIKE_MIMES):
                return -1

            lower = name.lower().strip()
            stem_norm = remove_suffixes(name)
            exact_names = {
                f"{base}.txt",
                f"{base}.md",
                f"{base}.markdown",
                f"{base}",
            }
            caption_names = {
                f"{base}.caption.txt",
                f"{base}_caption.txt",
                f"{base}-caption.txt",
                f"{base} caption.txt",
            }
            if lower in exact_names:
                return 100
            if lower in caption_names:
                return 95
            if stem_norm == base:
                return 90
            if base in stem_norm:
                return 80
            return -1

        ranked = sorted(
            (f for f in files if score(f) >= 0),
            key=lambda x: (score(x), str(x.get("modifiedTime") or "")),
            reverse=True,
        )
        return ranked[0] if ranked else None


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download an image + caption from Google Drive and write a feed entry")
    p.add_argument("--image", required=True, help="Google Drive image file ID/URL or any direct image URL")
    p.add_argument("--caption", default="", help="Google Drive caption file ID/URL or any direct text URL")
    p.add_argument("--kind", default="spot")
    p.add_argument("--avatar-image", default="avatar/spot.png")
    p.add_argument("--place", default="Yokosuka")
    p.add_argument("--lat", default="35.2810")
    p.add_argument("--lon", default="139.6722")
    p.add_argument("--tz-name", default="Asia/Tokyo")
    p.add_argument("--public-dir", default="frontend/app/public")
    p.add_argument("--latest-path", default="frontend/app/public/latest.json")
    p.add_argument("--output-state", default="")
    p.add_argument("--service-account-json-path", default="")
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    public_dir = Path(args.public_dir)
    latest_path = Path(args.latest_path)
    service_account_json = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "")
    caption_text_override = os.environ.get("DRIVE_POST_CAPTION_TEXT", "")

    drive = DriveAccessor(
        service_account_json=service_account_json,
        service_account_json_path=args.service_account_json_path or os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON_PATH", ""),
    )
    gen = load_generate_talk_module(repo_root)

    now_iso = gen.utc_now_iso_z()
    today = gen.utc_date()
    now_local = gen.local_stamp(args.tz_name)
    feed_stem = f"feed_{now_local}"

    image_bytes, image_meta = drive.download_binary(args.image)
    image_mime = str(image_meta.get("mimeType") or "application/octet-stream")
    image_name = str(image_meta.get("name") or "download")
    image_ext = infer_ext(image_name, image_mime, default=".bin")
    if image_ext.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".svg"}:
        raise SystemExit(f"Downloaded file does not look like an image: name={image_name!r} mimeType={image_mime!r}")

    caption_source = ""
    caption_meta: Dict[str, Any] = {}
    if caption_text_override.strip():
        caption_text = caption_text_override.strip()
        caption_source = "env:DRIVE_POST_CAPTION_TEXT"
    elif args.caption.strip():
        caption_text, caption_meta = drive.download_text(args.caption)
        caption_text = caption_text.strip()
        caption_source = str(caption_meta.get("name") or args.caption)
    else:
        sidecar = drive.find_sidecar_caption(image_meta)
        if not sidecar:
            raise SystemExit(
                "Caption was not provided and no sidecar caption file with the same basename was found in Google Drive. "
                "Pass --caption / workflow input caption_file, or set caption_text, or share a matching .txt/.md/Google Doc."
            )
        sid = str(sidecar.get("id") or "")
        caption_text, caption_meta = drive.download_text(sid)
        caption_text = caption_text.strip()
        caption_source = str(caption_meta.get("name") or sid)

    if not caption_text.strip():
        raise SystemExit("Caption text is empty")

    image_filename = f"{feed_stem}{image_ext.lower()}"
    image_path = public_dir / "image" / image_filename
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)

    snap_json_raw, snap_obj = gen.fetch_weather_snapshot(
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
        "source_image_name": image_name,
        "source_image_id": str(image_meta.get("id") or ""),
        "source_caption_name": str(caption_meta.get("name") or caption_source),
        "source_caption_id": str(caption_meta.get("id") or ""),
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
        "caption_source": caption_source,
        "source_image_name": image_name,
        "source_image_id": str(image_meta.get("id") or ""),
        "source_caption_name": str(caption_meta.get("name") or caption_source),
        "source_caption_id": str(caption_meta.get("id") or ""),
    }
    if args.output_state:
        out = Path(args.output_state)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
