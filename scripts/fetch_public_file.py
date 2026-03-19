#!/usr/bin/env python3
"""Download a public file from Google Drive or a direct URL.

Supported inputs:
- https://drive.google.com/file/d/<FILE_ID>/view?...
- https://drive.google.com/open?id=<FILE_ID>
- https://drive.google.com/uc?id=<FILE_ID>&export=download
- a non-Drive direct URL

This downloader follows the Google Drive confirm page used for large files.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from html import unescape
from http.cookiejar import CookieJar
from pathlib import Path

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def extract_file_id(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)

    match = re.search(r"/file/d/([^/]+)", parsed.path)
    if match:
        return match.group(1)

    if "id" in query and query["id"]:
        return query["id"][0]

    return None


def normalize_url(url: str) -> str:
    if "drive.google.com" not in url:
        return url

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    resource_key = (query.get("resourcekey") or [None])[0]
    file_id = extract_file_id(url)
    if not file_id:
        raise SystemExit(f"Could not extract Google Drive file id from URL: {url}")

    params = {"export": "download", "id": file_id}
    if resource_key:
        params["resourcekey"] = resource_key
    return "https://drive.google.com/uc?" + urllib.parse.urlencode(params)


def build_opener() -> urllib.request.OpenerDirector:
    jar = CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def http_get(opener: urllib.request.OpenerDirector, url: str) -> tuple[bytes, dict[str, str]]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener.open(req, timeout=300) as resp:
        body = resp.read()
        headers = {k.lower(): v for k, v in resp.headers.items()}
    return body, headers


def looks_like_drive_confirm_page(body: bytes, headers: dict[str, str], current_url: str) -> bool:
    if "drive.google.com" not in current_url:
        return False
    content_type = headers.get("content-type", "")
    if "text/html" not in content_type and b"download_warning" not in body:
        return False
    html = body.decode("utf-8", errors="ignore")
    return (
        "download_warning" in html
        or 'id="download-form"' in html
        or ("Google Drive" in html and "download" in html)
    )


def maybe_follow_drive_confirm(
    opener: urllib.request.OpenerDirector,
    body: bytes,
    headers: dict[str, str],
    current_url: str,
) -> tuple[bytes, dict[str, str]]:
    if not looks_like_drive_confirm_page(body, headers, current_url):
        return body, headers

    html = body.decode("utf-8", errors="ignore")
    candidates: list[str] = []

    for match in re.finditer(r'href="([^"]+)"', html):
        href = unescape(match.group(1))
        if "confirm=" in href and ("/uc?" in href or "export=download" in href):
            candidates.append(urllib.parse.urljoin("https://drive.google.com", href))

    form_match = re.search(r'<form id="download-form"[^>]*action="([^"]+)"', html)
    if form_match:
        action = unescape(form_match.group(1)).replace("&amp;", "&")
        hidden_inputs = dict(
            re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', html)
        )
        if hidden_inputs:
            qs = urllib.parse.urlencode(hidden_inputs)
            sep = "&" if "?" in action else "?"
            candidates.append(action + sep + qs)
        else:
            candidates.append(action)

    for candidate in candidates:
        body2, headers2 = http_get(opener, candidate)
        content_type2 = headers2.get("content-type", "")
        content_disp2 = headers2.get("content-disposition", "")
        if "attachment" in content_disp2.lower() or "text/html" not in content_type2:
            return body2, headers2

    return body, headers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Public Google Drive URL or direct URL")
    parser.add_argument("--out", required=True, help="Output path for the downloaded file")
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1,
        help="Fail if the downloaded file is smaller than this many bytes (default: 1)",
    )
    args = parser.parse_args()

    final_url = normalize_url(args.url.strip())
    opener = build_opener()
    body, headers = http_get(opener, final_url)
    body, headers = maybe_follow_drive_confirm(opener, body, headers, final_url)

    if len(body) < args.min_bytes:
        raise SystemExit(
            f"Downloaded file is too small: {len(body)} bytes < required {args.min_bytes}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(body)

    print(f"Saved file to: {out_path}")
    print(f"Source URL: {final_url}")
    print(f"Bytes: {len(body)}")
    print(f"Content-Type: {headers.get('content-type', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
