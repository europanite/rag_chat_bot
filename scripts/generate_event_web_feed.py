#!/usr/bin/env python3
"""Generate a fixed feed entry for the nearest future Cocoyoko event.

This script intentionally does not use the internal RAG pipeline.
It starts from the public Cocoyoko event index and follows detail links.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests

INDEX_URL = "https://www.cocoyoko.net/event/"
PLACE = "Yokosuka"
AVATAR_IMAGE = "image/avatar/event.png"
TIMEOUT = 60
USER_AGENT = "goodday-yokosuka-event-bot/1.0 (+https://goodday-yokosuka.com/)"

DATE_TOKEN_RE = re.compile(r"(\d{4}:\d{2}:\d{2}:\d{2}:\d{2}:\d{2})\|(\d{4}:\d{2}:\d{2}:\d{2}:\d{2}:\d{2})\|\d+")
JP_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
TITLE_RE = re.compile(r"<h1[^>]*>(.*?)</h1>|<h2[^>]*>(.*?)</h2>", re.IGNORECASE | re.DOTALL)
INFO_SECTION_RE = re.compile(r"##\s*information(.*?)(?:##\s*access|##\s*about|##\s*spot|$)", re.IGNORECASE | re.DOTALL)
ABOUT_SECTION_RE = re.compile(r"##\s*about(.*?)(?:##\s*spot|##\s*access|$)", re.IGNORECASE | re.DOTALL)
URL_RE = re.compile(r'https?://[^\s)>\'"\\]+')

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = dict(attrs)
        href = attr_map.get("href")
        if href:
            self.links.append(href)


@dataclass
class Event:
    title: str
    detail_url: str
    official_url: str
    venue: str
    summary: str
    description: str
    start_at: datetime
    end_at: datetime
    date_label: str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args()

def decode_html_response(response: requests.Response) -> str:
    content_type = (response.headers.get("Content-Type") or "").lower()

    if "charset=shift_jis" in content_type or "charset=cp932" in content_type:
        return response.content.decode("cp932", errors="replace")
    if "charset=utf-8" in content_type:
        return response.content.decode("utf-8", errors="replace")

    for enc in ("cp932", "shift_jis", response.apparent_encoding, response.encoding, "utf-8"):
        if not enc:
            continue
        try:
            text = response.content.decode(enc, errors="strict")
        except Exception:
            continue
        lowered = text.lower()
        if "<html" in lowered or "<!doctype html" in lowered or "<body" in lowered:
            return text

    return response.content.decode("utf-8", errors="replace")

def fetch(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    return decode_html_response(response)

def strip_tags(text: str) -> str:
    return WHITESPACE_RE.sub(" ", unescape(TAG_RE.sub(" ", text))).strip()


def parse_index_links(index_html: str, base_url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(index_html)
    urls: list[str] = []
    seen: set[str] = set()
    for href in parser.links:
        absolute = urljoin(base_url, href)
        if not absolute.startswith("https://www.cocoyoko.net/event/"):
            continue
        if absolute.rstrip("/") == base_url.rstrip("/"):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
    return urls


def parse_datetime_token(token: str) -> datetime:
    return datetime.strptime(token, "%Y:%m:%d:%H:%M:%S")


def parse_date_tokens(text: str) -> tuple[datetime, datetime] | None:
    match = DATE_TOKEN_RE.search(text)
    if not match:
        return None
    start_at = parse_datetime_token(match.group(1))
    end_at = parse_datetime_token(match.group(2))
    return start_at, end_at


def parse_jp_dates(text: str) -> tuple[datetime, datetime] | None:
    matches = list(JP_DATE_RE.finditer(text))
    if not matches:
        return None
    first = matches[0]
    start_at = datetime(int(first.group(1)), int(first.group(2)), int(first.group(3)))
    if len(matches) >= 2:
        last = matches[1]
        end_at = datetime(int(last.group(1)), int(last.group(2)), int(last.group(3)), 23, 59, 59)
    else:
        end_at = datetime(int(first.group(1)), int(first.group(2)), int(first.group(3)), 23, 59, 59)
    return start_at, end_at


def extract_title(html: str) -> str:
    match = TITLE_RE.search(html)
    if not match:
        raise ValueError("Could not find event title")
    title = match.group(1) or match.group(2) or ""
    title = strip_tags(title)
    if not title:
        raise ValueError("Event title is empty")
    return title


def extract_section(pattern: re.Pattern[str], html: str) -> str:
    match = pattern.search(html)
    if not match:
        return ""
    return strip_tags(match.group(1))


def extract_official_url(html: str, detail_url: str) -> str:
    for line in strip_tags(html).splitlines():
        if "公式サイト" in line:
            found = URL_RE.search(line)
            if found:
                return found.group(0)
    parser = LinkParser()
    parser.feed(html)
    for href in parser.links:
        absolute = urljoin(detail_url, href)
        if absolute.startswith("https://www.cocoyoko.net/"):
            continue
        if absolute.startswith("http://") or absolute.startswith("https://"):
            return absolute
    return detail_url


def extract_venue(html: str) -> str:
    cleaned = strip_tags(html)
    venue_patterns = [
        re.compile(r"開催場所[:：]?\s*(.+?)(?:公式サイト|お問合わせ|主催者情報|アクセス|$)"),
        re.compile(r"会場[:：]?\s*(.+?)(?:公式サイト|お問合わせ|主催者情報|アクセス|$)"),
    ]
    for pattern in venue_patterns:
        match = pattern.search(cleaned)
        if match:
            value = WHITESPACE_RE.sub(" ", match.group(1)).strip(" ・|｜:：")
            if value:
                return value
    return PLACE


def first_sentences(text: str, limit: int = 2) -> str:
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[。.!?])\s+", text) if chunk.strip()]
    return " ".join(chunks[:limit]).strip()


def build_summary(title: str, date_label: str, venue: str, description: str) -> str:
    lead = f"{title} is an upcoming event in {PLACE} scheduled for {date_label}"
    if venue and venue != PLACE:
        lead += f" at {venue}"
    lead += "."

    detail = description.strip()
    if not detail:
        return lead + " Please check the official page for the latest details."

    detail = detail.replace("。", ". ")
    detail = WHITESPACE_RE.sub(" ", detail).strip()
    detail = first_sentences(detail, limit=2)
    if not detail.endswith((".", "!", "?")):
        detail += "."
    return f"{lead} {detail}"


def format_date_range(start_at: datetime, end_at: datetime) -> str:
    start = f"{MONTH_NAMES[start_at.month]} {start_at.day}, {start_at.year}"
    end = f"{MONTH_NAMES[end_at.month]} {end_at.day}, {end_at.year}"
    if start_at.date() == end_at.date():
        return start
    return f"{start} to {end}"


def parse_event(detail_url: str, html: str) -> Event | None:
    title = extract_title(html)

    date_pair = parse_date_tokens(html)
    if date_pair is None:
        date_pair = parse_jp_dates(strip_tags(html))
    if date_pair is None:
        return None
    start_at, end_at = date_pair

    venue = extract_venue(html)
    info = extract_section(INFO_SECTION_RE, html)
    about = extract_section(ABOUT_SECTION_RE, html)
    description = about or info or title
    summary = build_summary(title, format_date_range(start_at, end_at), venue, description)
    official_url = extract_official_url(html, detail_url)

    return Event(
        title=title,
        detail_url=detail_url,
        official_url=official_url,
        venue=venue,
        summary=summary,
        description=description,
        start_at=start_at,
        end_at=end_at,
        date_label=format_date_range(start_at, end_at),
    )


def select_future_event(events: Iterable[Event]) -> Event:
    now = datetime.now()
    future_events = [event for event in events if event.end_at >= now]
    if not future_events:
        raise SystemExit("No future Cocoyoko events were found")
    future_events.sort(key=lambda event: (event.start_at, event.end_at, event.title))
    return future_events[0]


def build_entry(event: Event) -> dict[str, object]:
    text = event.summary
    if not text.endswith("."):
        text += "."
    text += " Check the official page for the latest schedule and participation details."
    return {
        "kind": "event",
        "place": PLACE,
        "title": event.title,
        "text": text,
        "summary": event.summary,
        "description": event.description,
        "avatar_image": AVATAR_IMAGE,
        "links": [event.detail_url, event.official_url] if event.official_url != event.detail_url else [event.detail_url],
        "event_start": event.start_at.isoformat(),
        "event_end": event.end_at.isoformat(),
    }


def main() -> int:
    args = parse_args()
    index_html = fetch(INDEX_URL)
    detail_urls = parse_index_links(index_html, INDEX_URL)
    if not detail_urls:
        raise SystemExit("Found 0 event detail links on index page")

    events: list[Event] = []
    for detail_url in detail_urls:
        try:
            detail_html = fetch(detail_url)
            event = parse_event(detail_url, detail_html)
            if event is not None:
                events.append(event)
        except requests.RequestException:
            continue
        except ValueError:
            continue

    selected = select_future_event(events)
    payload = build_entry(selected)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Selected future event: {selected.title}")
    print(f"Wrote payload to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
