#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
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

try:
    from artifact_paths import (
        artifact_feed_dir,
        artifact_image_dir,
        artifact_latest_path,
        artifact_public_dir,
        resolve_public_image_url,
    )
except ImportError:
    from scripts.artifact_paths import (
        artifact_feed_dir,
        artifact_image_dir,
        artifact_latest_path,
        artifact_public_dir,
        resolve_public_image_url,
    )

USER_AGENT = "rag_chat_bot-gha/drive-random-article"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".svg"}
TEXT_EXTS = {".md", ".markdown", ".txt"}
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
    kind: str = "unknown"  # folder | image | text | doc | unknown


@dataclass
class ImageRef:
    source: str
    alt: str = ""


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


def request_text(url: str, *, timeout: int = 120) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def request_bytes(url: str, *, timeout: int = 120) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), (resp.headers.get_content_type() or "application/octet-stream")


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    safe = safe.strip("._")
    return safe or "file"


def slugify(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "article"


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
    if "/folders/" in href:
        return "folder"
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


def list_public_folder_entries(folder: FolderRef, *, include_folders: bool) -> list[DriveEntry]:
    params = {"id": folder.folder_id}
    if folder.resource_key:
        params["resourcekey"] = folder.resource_key
    q = urllib.parse.urlencode(params)
    urls = [
        f"https://drive.google.com/embeddedfolderview?{q}#list",
        f"https://drive.google.com/drive/folders/{folder.folder_id}?{q}",
        f"https://drive.google.com/drive/mobile/folders/{folder.folder_id}?{q}",
    ]

    seen: set[tuple[str, str, str]] = set()
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
            if not href2.startswith(("http://", "https://")):
                continue
            if "drive.google.com" not in href2 and "docs.google.com" not in href2:
                continue
            is_folder = "/folders/" in href2
            if is_folder and not include_folders:
                continue
            file_id = ""
            if is_folder:
                try:
                    file_id = extract_folder_ref(href2).folder_id
                except SystemExit:
                    continue
            else:
                try:
                    file_id = extract_file_id(href2)
                except SystemExit:
                    continue
            name = text.strip() or Path(urllib.parse.urlparse(href2).path).name.strip()
            if not name:
                continue
            kind = classify_entry(name, href2)
            resource_key = extract_resource_key(href2) or folder.resource_key
            key = (kind, file_id, name)
            if key in seen:
                continue
            seen.add(key)
            out.append(DriveEntry(name=name, href=href2, file_id=file_id, resource_key=resource_key, kind=kind))
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


def recent_source_folder_ids(feed_dir: Path, limit: int) -> set[str]:
    if limit <= 0 or not feed_dir.exists():
        return set()
    out: set[str] = set()
    files = sorted(feed_dir.glob("feed_*.json"), reverse=True)[:limit]
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = str(obj.get("source_article_folder_id") or "").strip()
        if sid:
            out.add(sid)
    return out


def choose_random_article_folder(
    folders: Sequence[DriveEntry],
    *,
    excluded_folder_ids: set[str],
    random_seed: str,
) -> DriveEntry:
    if not folders:
        raise SystemExit("No child folders were found under the articles root folder")
    rng = random.Random()
    rng.seed(random_seed or None)
    fresh = [it for it in folders if it.file_id not in excluded_folder_ids]
    target = fresh if fresh else list(folders)
    return rng.choice(target)


def choose_article_doc(entries: Sequence[DriveEntry]) -> DriveEntry:
    candidates = [it for it in entries if it.kind in {"text", "doc"}]
    if not candidates:
        raise SystemExit("Selected article folder does not contain any .md/.txt/Google Docs article source")

    def sort_key(it: DriveEntry) -> tuple[int, int, str]:
        name = it.name.lower().strip()
        preferred = 0 if name in {"article.md", "index.md", "README.md".lower(), "post.md"} else 1
        text_priority = 0 if name.endswith(".md") or name.endswith(".markdown") else 1
        return preferred, text_priority, name

    return sorted(candidates, key=sort_key)[0]


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
    return request_bytes(url)


def download_text(entry: DriveEntry) -> str:
    url = build_doc_export_url(entry) if entry.kind == "doc" else build_download_url(entry)
    data, _ = request_bytes(url)
    return data.decode("utf-8-sig", errors="replace")


def parse_ref(value: str) -> ImageRef:
    raw = (value or "").strip()
    if not raw:
        return ImageRef(source="", alt="")
    if "|" in raw:
        src, alt = raw.split("|", 1)
        return ImageRef(source=src.strip(), alt=alt.strip())
    return ImageRef(source=raw, alt="")


def parse_link(value: str) -> Dict[str, str]:
    raw = (value or "").strip()
    if not raw:
        return {"title": "", "url": ""}
    if "|" in raw:
        title, url = raw.split("|", 1)
        return {"title": title.strip(), "url": url.strip()}
    return {"title": raw, "url": raw}


def parse_article_markdown(text: str) -> Dict[str, Any]:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")

    header_lines: list[str] = []
    body_start = 0
    for idx, line in enumerate(lines):
        if line.strip() == "---":
            body_start = idx + 1
            break
        header_lines.append(line)
    else:
        body_start = 0
        header_lines = []

    meta: Dict[str, Any] = {"photos": [], "links": []}
    for line in header_lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if not value:
            continue
        if key == "photo":
            meta["photos"].append(parse_ref(value))
        elif key == "link":
            link = parse_link(value)
            if link["url"]:
                meta["links"].append(link)
        elif key == "hero-image":
            meta["hero_image"] = parse_ref(value)
        else:
            meta[key.replace("-", "_")] = value

    body = "\n".join(lines[body_start:]).strip()
    if not body:
        body = raw.strip()

    title = str(meta.get("title") or "").strip()
    if not title:
        m = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
        if m:
            title = m.group(1).strip()
    if not title:
        raise SystemExit("Article markdown is missing Title: and no H1 was found in the body")

    summary = str(meta.get("summary") or "").strip()
    if not summary:
        summary = extract_summary(body)

    short_title = str(meta.get("short_title") or title).strip()
    short_text = str(meta.get("short_text") or summary).strip()

    return {
        "title": title,
        "slug": str(meta.get("slug") or slugify(title)).strip() or slugify(title),
        "category": str(meta.get("category") or "").strip(),
        "place": str(meta.get("place") or "").strip(),
        "published_at": str(meta.get("published_at") or "").strip(),
        "summary": summary,
        "short_title": short_title,
        "short_text": short_text,
        "hero_image": meta.get("hero_image") if isinstance(meta.get("hero_image"), ImageRef) else None,
        "photos": list(meta.get("photos") or []),
        "links": list(meta.get("links") or []),
        "body_md": body,
    }


def extract_summary(body_md: str, limit: int = 220) -> str:
    text = body_md
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>+\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].strip()
    return clipped + "..."


def find_entry_by_name(entries: Sequence[DriveEntry], source: str) -> Optional[DriveEntry]:
    wanted = Path((source or "").strip()).name.lower()
    if not wanted:
        return None
    for entry in entries:
        if entry.kind not in {"image", "text", "doc"}:
            continue
        if Path(entry.name).name.lower() == wanted:
            return entry
    return None


def looks_like_url(value: str) -> bool:
    s = (value or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def drive_entry_from_url(source: str) -> Optional[DriveEntry]:
    try:
        file_id = extract_file_id(source)
    except SystemExit:
        return None
    name = Path(urllib.parse.urlparse(source).path).name or file_id
    return DriveEntry(name=name, href=source, file_id=file_id, resource_key=extract_resource_key(source), kind="image")


def resolve_ref_to_source(ref: ImageRef, folder_entries: Sequence[DriveEntry]) -> tuple[str, Optional[DriveEntry]]:
    if not ref or not ref.source:
        return "", None
    src = ref.source.strip()
    if looks_like_url(src):
        entry = drive_entry_from_url(src)
        return src, entry
    entry = find_entry_by_name(folder_entries, src)
    if entry is None:
        raise SystemExit(f"Referenced file {src!r} was not found in the selected article folder")
    return src, entry


def download_ref(ref: ImageRef, folder_entries: Sequence[DriveEntry]) -> tuple[bytes, str, str]:
    src, entry = resolve_ref_to_source(ref, folder_entries)
    if entry is not None:
        data, content_type = download_entry_bytes(entry)
        filename = entry.name or Path(src).name or entry.file_id
        return data, content_type, filename
    data, content_type = request_bytes(src)
    filename = Path(urllib.parse.urlparse(src).path).name or "image"
    return data, content_type, filename


def ensure_iso(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(microsecond=0).isoformat()
    except Exception:
        return raw


def utc_date_from_iso(value: str) -> str:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


def local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_article_image(public_dir: Path, slug: str, filename: str, data: bytes, mime_type: str) -> str:
    ext = infer_ext(filename, mime_type, default=".bin")
    if ext.lower() not in IMAGE_EXTS:
        raise SystemExit(f"Downloaded asset is not an image: {filename!r} mime={mime_type!r}")
    base = sanitize_filename(Path(filename).stem)
    final_name = f"{base}{ext.lower()}"
    rel = Path("image") / "articles" / slug / final_name
    out_path = public_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return resolve_public_image_url(str(rel).replace("\\", "/"))


def md_inline_to_html(text: str) -> str:
    s = html.escape(text or "")
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", "", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def markdown_to_html(md: str) -> str:
    lines = (md or "").splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    in_list = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{md_inline_to_html(' '.join(paragraph).strip())}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            close_list()
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            close_list()
            out.append(f"<h3>{md_inline_to_html(stripped[4:])}</h3>")
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            close_list()
            out.append(f"<h2>{md_inline_to_html(stripped[3:])}</h2>")
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            close_list()
            out.append(f"<h1>{md_inline_to_html(stripped[2:])}</h1>")
            continue
        if re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            if not in_list:
                out.append("<ul>")
                in_list = True
            item = re.sub(r"^[-*]\s+", "", stripped)
            out.append(f"<li>{md_inline_to_html(item)}</li>")
            continue
        paragraph.append(stripped)

    flush_paragraph()
    close_list()
    return "\n".join(out)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_article_html(article: Dict[str, Any]) -> str:
    title = html.escape(str(article.get("title") or "GOODDAY YOKOSUKA"))
    summary = html.escape(str(article.get("summary") or ""))
    hero = str(article.get("hero_image") or "").strip()
    body_html = markdown_to_html(str(article.get("body_md") or ""))

    photo_html = []
    for photo in article.get("photos") or []:
        url = html.escape(str(photo.get("url") or ""))
        alt = html.escape(str(photo.get("alt") or ""))
        if url:
            photo_html.append(f'<figure><img src="{url}" alt="{alt}" /><figcaption>{alt}</figcaption></figure>')

    links_html = []
    for link in article.get("links") or []:
        label = html.escape(str(link.get("title") or link.get("url") or ""))
        url = html.escape(str(link.get("url") or ""))
        if url:
            links_html.append(f'<li><a href="{url}">{label}</a></li>')

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>{title} | GOODDAY YOKOSUKA</title>
  <meta name=\"description\" content=\"{summary}\" />
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f4ff; color: #111827; }}
    .wrap {{ max-width: 860px; margin: 0 auto; padding: 24px 16px 72px; }}
    .card {{ background: #fff; border: 2px solid #000; border-radius: 20px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,.08); }}
    h1, h2, h3 {{ line-height: 1.25; }}
    p, li {{ line-height: 1.8; font-size: 16px; }}
    .eyebrow {{ display: inline-block; font-size: 12px; background: #ede9fe; border-radius: 999px; padding: 6px 10px; margin-bottom: 12px; }}
    .meta {{ color: #4b5563; font-size: 14px; margin-bottom: 16px; }}
    img {{ display: block; max-width: 100%; height: auto; border-radius: 16px; border: 1px solid #ddd; }}
    figure {{ margin: 20px 0; }}
    figcaption {{ color: #6b7280; font-size: 13px; margin-top: 8px; }}
    a {{ color: #0B57D0; }}
  </style>
</head>
<body>
  <main class=\"wrap\">
    <div class=\"card\">
      <div class=\"eyebrow\">GOODDAY YOKOSUKA / long-form guide</div>
      <h1>{title}</h1>
      <div class=\"meta\">{html.escape(str(article.get('place') or 'Yokosuka'))} · {html.escape(str(article.get('published_at') or ''))}</div>
      {f'<img src="{html.escape(hero)}" alt="{title}" />' if hero else ''}
      {f'<p>{summary}</p>' if summary else ''}
      {body_html}
      {''.join(photo_html)}
      {f'<h2>Links</h2><ul>{"".join(links_html)}</ul>' if links_html else ''}
    </div>
  </main>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pick a random child folder from a public Google Drive root, build a long-form article page, and emit a teaser feed entry")
    p.add_argument("--root-folder-url", required=True, help="Public Google Drive root folder URL or folder ID")
    p.add_argument("--place", default="Yokosuka")
    p.add_argument("--avatar-image", default="image/avatar/spot.png")
    p.add_argument("--kind", default="article_feature")
    p.add_argument("--public-dir", default="")
    p.add_argument("--latest-path", default="")
    p.add_argument("--recent-exclude-limit", type=int, default=20)
    p.add_argument("--random-seed", default="")
    p.add_argument("--output-state", default="")
    return p


def main() -> int:
    args = build_parser().parse_args()

    public_dir = Path(args.public_dir) if str(args.public_dir).strip() else artifact_public_dir()
    latest_path = Path(args.latest_path) if str(args.latest_path).strip() else artifact_latest_path(public_dir)
    feed_dir = artifact_feed_dir(public_dir)
    image_dir = artifact_image_dir(public_dir)
    feed_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    root_folder = extract_folder_ref(args.root_folder_url)
    root_entries = list_public_folder_entries(root_folder, include_folders=True)
    child_folders = [it for it in root_entries if it.kind == "folder"]
    excluded = recent_source_folder_ids(feed_dir, args.recent_exclude_limit)
    chosen_folder = choose_random_article_folder(child_folders, excluded_folder_ids=excluded, random_seed=args.random_seed)
    article_folder = extract_folder_ref(chosen_folder.href)
    article_entries = list_public_folder_entries(article_folder, include_folders=False)

    article_doc = choose_article_doc(article_entries)
    article_text = download_text(article_doc)
    article = parse_article_markdown(article_text)
    slug = slugify(str(article.get("slug") or article.get("title") or chosen_folder.name))
    published_at = ensure_iso(str(article.get("published_at") or ""))
    published_date = utc_date_from_iso(published_at)

    hero_url = ""
    photos: list[dict[str, str]] = []

    hero_ref = article.get("hero_image")
