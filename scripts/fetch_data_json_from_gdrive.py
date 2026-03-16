#!/usr/bin/env python3
"""Download a public Google Drive JSON file to a local path.

Supported inputs:
- https://drive.google.com/file/d/<FILE_ID>/view?...
- https://drive.google.com/open?id=<FILE_ID>
- https://drive.google.com/uc?id=<FILE_ID>&export=download
- a non-Drive direct JSON URL

The downloader follows the Google Drive confirm page when Drive shows the
"download anyway" interstitial for large files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from html import unescape
from http.cookiejar import CookieJar
from pathlib import Path

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"


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

    params = {
        "export": "download",
        "id": file_id,
    }
    if resource_key:
        params["resourcekey"] = resource_key
    return "https://drive.google.com/uc?" + urllib.parse.urlencode(params)


def build_opener() -> urllib.request.OpenerDirector:
    jar = CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def http_get(opener: urllib.request.OpenerDirector, url: str) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener.open(req, timeout=120) as resp:
        body = resp.read()
        content_type = resp.headers.get("Content-Type", "")
    return body, content_type


def maybe_follow_drive_confirm(
    opener: urllib.request.OpenerDirector,
    body: bytes,
    content_type: str,
    current_url: str,
) -> tuple[bytes, str]:
    if "drive.google.com" not in current_url:
        return body, content_type

    html = body.decode("utf-8", errors="ignore")

    # If JSON already arrived, keep it.
    try:
        json.loads(html)
        return body, content_type
    except Exception:
        pass

    # Google Drive sometimes returns an HTML confirm page before the real file.
    if "text/html" not in content_type and "download_warning" not in html:
        return body, content_type

    candidates: list[str] = []

    for match in re.finditer(r'href="([^"]+)"', html):
        href = unescape(match.group(1))
        if "confirm=" in href and ("/uc?" in href or "export=download" in href):
            candidates.append(urllib.parse.urljoin("https://drive.google.com", href))

    form_match = re.search(r'<form id="download-form"[^>]*action="([^"]+)"', html)
    if form_match:
        action = unescape(form_match.group(1)).replace("&amp;", "&")
        hidden_inputs = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', html))
        if hidden_inputs:
            qs = urllib.parse.urlencode(hidden_inputs)
            sep = "&" if "?" in action else "?"
            candidates.append(action + sep + qs)
        else:
            candidates.append(action)

    for candidate in candidates:
        body2, content_type2 = http_get(opener, candidate)
        try:
            json.loads(body2.decode("utf-8"))
            return body2, content_type2
        except Exception:
            continue

    return body, content_type


def validate_json_bytes(data: bytes) -> None:
    try:
        json.loads(data.decode("utf-8"))
    except Exception as exc:
        preview = data[:500].decode("utf-8", errors="replace")
        raise SystemExit(
            "Downloaded content is not valid UTF-8 JSON. "
            f"Preview: {preview!r}. Error: {exc}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Public Google Drive URL or direct JSON URL")
    parser.add_argument("--out", required=True, help="Output path for the downloaded JSON file")
    args = parser.parse_args()

    final_url = normalize_url(args.url.strip())
    opener = build_opener()
    body, content_type = http_get(opener, final_url)
    body, content_type = maybe_follow_drive_confirm(opener, body, content_type, final_url)
    validate_json_bytes(body)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(body)

    print(f"Saved JSON to: {out_path}")
    print(f"Source URL: {final_url}")
    print(f"Bytes: {len(body)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
