from __future__ import annotations

import json
import os
import re
import sys
import random
from dataclasses import dataclass
from datetime import datetime
from time import sleep
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup

import requests

try:
    from artifact_paths import (
        artifact_latest_path,
        artifact_public_dir,
        artifact_snapshot_dir,
        computed_feed_snapshot_path,
        resolve_public_avatar_url,
    )
except ImportError:
    from scripts.artifact_paths import (
        artifact_latest_path,
        artifact_public_dir,
        artifact_snapshot_dir,
        computed_feed_snapshot_path,
        resolve_public_avatar_url,
    )


INDEX_URL = os.getenv("COCOYOKO_EVENT_INDEX_URL", "https://www.cocoyoko.net/event/").strip()
TZ_NAME = os.getenv("TZ_NAME", "Asia/Tokyo").strip() or "Asia/Tokyo"
PLACE = os.getenv("PLACE", "Yokosuka").strip() or "Yokosuka"
AVATAR_IMAGE = os.getenv("AVATAR_IMAGE", "image/avatar/event.png").strip()
MAX_CHARS = max(120, int(os.getenv("MAX_CHARS", "220")))
DEBUG = os.getenv("DEBUG", "0").strip() == "1"
TIMEOUT = max(30, int(os.getenv("OLLAMA_TIMEOUT_S")))
OLLAMA_MAX_RETRIES = max(1, int(os.getenv("OLLAMA_MAX_RETRIES", "2")))
MAX_DETAIL_LINKS = max(1, int(os.getenv("COCOYOKO_MAX_DETAIL_LINKS", "7")))
EVENT_RANDOM_POOL = max(1, int(os.getenv("COCOYOKO_EVENT_RANDOM_POOL", "3")))
EVENT_RANDOMIZE = os.getenv("COCOYOKO_EVENT_RANDOMIZE", "1").strip() != "0"
UA = "goodday-yokosuka-event-bot/1.0 (+https://goodday-yokosuka.com/)"

JP_DATE_STAMP_RE = re.compile(
    r"(\d{4}):(\d{2}):(\d{2}):\d{2}:\d{2}:\d{2}\|(\d{4}):(\d{2}):(\d{2}):\d{2}:\d{2}:\d{2}\|\d+"
)
TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
WS_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+")


@dataclass
class Event:
    title: str
    detail_url: str
    official_url: str
    start_at: datetime
    end_at: datetime
    date_label: str
    place: str
    description: str


class EventFeedError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def debug(msg: str) -> None:
    if DEBUG:
        log(msg)


def unique_in_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


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
    response = requests.get(
        url,
        headers={"User-Agent": UA, "Accept-Language": "ja,en-US;q=0.8,en;q=0.6"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return decode_html_response(response)


def strip_tags(text: str) -> str:
    text = SCRIPT_STYLE_RE.sub(" ", text)
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"(?i)</?(p|div|li|section|article|h1|h2|h3|h4|tr|td|th|dt|dd|ul|ol)\b[^>]*>", "\n", text)
    text = TAG_RE.sub(" ", text)
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def parse_date_stamp(text: str) -> Optional[tuple[datetime, datetime]]:
    m = JP_DATE_STAMP_RE.search(text)
    if not m:
        return None
    tz = ZoneInfo(TZ_NAME)
    start_at = datetime(
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        0,
        0,
        0,
        tzinfo=tz,
    )
    end_at = datetime(
        int(m.group(4)),
        int(m.group(5)),
        int(m.group(6)),
        23,
        59,
        59,
        tzinfo=tz,
    )
    return start_at, end_at


def parse_index_links(index_html: str, base_url: str) -> List[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', index_html, re.IGNORECASE)
    urls: List[str] = []
    base_netloc = urlparse(base_url).netloc.lower()
    for href in hrefs:
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.netloc.lower() != base_netloc:
            continue
        if not parsed.path.startswith("/event/"):
            continue
        if not parsed.path.endswith(".html"):
            continue
        if absolute.rstrip("/") == base_url.rstrip("/"):
            continue
        urls.append(absolute)
    return unique_in_order(urls)


def extract_candidate_text(raw_html: str) -> str:
    text = strip_tags(raw_html)
    lines = [WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines[:220])


def parse_json_from_llm_output(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    block = re.search(r"\{.*\}", text, re.DOTALL)
    if not block:
        raise EventFeedError("LLM did not return JSON")
    return json.loads(block.group(0))


def call_openai(messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    if not api_key:
        raise EventFeedError("OPENAI_API_KEY is not set")
    last_exc: Optional[Exception] = None
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.1},
                    "format": "json",
                },
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            msg = data.get("message", {})
            content = msg.get("content")
            if not content:
                raise EventFeedError("Ollama returned no content")
            return content
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            log(f"WARN Ollama timeout on attempt {attempt}/{OLLAMA_MAX_RETRIES}")
            if attempt < OLLAMA_MAX_RETRIES:
                sleep(min(2 * attempt, 5))
                continue
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            break

    raise EventFeedError(f"Ollama chat failed after {OLLAMA_MAX_RETRIES} attempt(s): {last_exc}")




def validate_runtime() -> None:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower() or "ollama"
    if provider != "ollama":
        return

    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").strip().rstrip("/")
    if not base_url:
        raise EventFeedError("OLLAMA_BASE_URL is not set")

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=TIMEOUT)
        response.raise_for_status()
    except Exception as exc:
        raise EventFeedError(f"Ollama is unreachable at {base_url}: {exc}") from exc


def call_ollama(messages: List[Dict[str, str]]) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").strip().rstrip("/")
    model = (
        os.getenv("RAG_MODEL", "").strip()
        or os.getenv("AUDIT_MODEL", "").strip()
        or os.getenv("OPENAI_CHAT_MODEL", "").strip()
        or "llama3.1:8b"
    )
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1},
            "format": "json",
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    msg = data.get("message", {})
    content = msg.get("content")
    if not content:
        raise EventFeedError("Ollama returned no content")
    return content

def html_to_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text("\n", strip=True)

def interpret_event_page(detail_url: str, raw_html: str) -> Dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower() or "ollama"

    page_text = html_to_text(raw_html)
    page_text = page_text[:12000]

    system = (
        "You extract structured event facts from Japanese tourism web pages.\n"
        "Return only JSON.\n"
        "Do not invent facts.\n"
        "Use empty string when unknown.\n"
    )

    user = f"""Extract the actual event information from this event detail page.

Return JSON with exactly these keys:
title
venue
date_label
description_ja
official_url

detail_url:
{detail_url}

page_text:
{page_text}
"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if provider == "ollama":
        content = call_ollama(messages)
    else:
        content = call_openai(messages)

    parsed = parse_json_from_llm_output(content) or {}

    return {
        "title": str(parsed.get("title", "") or "").strip(),
        "venue": str(parsed.get("venue", "") or "").strip(),
        "date_label": str(parsed.get("date_label", "") or "").strip(),
        "description_ja": str(parsed.get("description_ja", "") or "").strip(),
        "official_url": str(parsed.get("official_url", "") or "").strip(),
    }

def fallback_official_url(raw_html: str, detail_url: str) -> str:
    for href in re.findall(r'href=["\']([^"\']+)["\']', raw_html, re.IGNORECASE):
        abs_url = urljoin(detail_url, unescape(href))
        if not abs_url.startswith("http"):
            continue
        if "cocoyoko.net/event/" in abs_url:
            continue
        if "maps.google" in abs_url:
            continue
        return abs_url
    return detail_url


def parse_event_detail(url: str) -> Event:
    raw_html = fetch(url)

    date_range = parse_date_stamp(raw_html)
    if not date_range:
        raise EventFeedError(f"date stamp not found: {url}")
    start_at, end_at = date_range

    interpreted = interpret_event_page(url, raw_html)
    title = interpreted["title"]
    place = interpreted["venue"]
    date_label = interpreted["date_label"]
    description = interpreted["description_ja"]
    official_url = interpreted["official_url"] or fallback_official_url(raw_html, url)

    return Event(
        title=title,
        detail_url=url,
        official_url=official_url,
        start_at=start_at,
        end_at=end_at,
        date_label=date_label,
        place=place,
        description=description,
    )

def english_date_phrase(event: Event) -> str:
    s = event.start_at
    e = event.end_at
    if s.date() == e.date():
        return f"{s.strftime('%B')} {s.day}, {s.year}"
    if s.year == e.year and s.month == e.month:
        return f"{s.strftime('%B')} {s.day}-{e.day}, {s.year}"
    return f"{s.strftime('%B')} {s.day}, {s.year} to {e.strftime('%B')} {e.day}, {e.year}"


def summarize_event_for_post(event: Event) -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower() or "ollama"

    system = (
        "Write a short but informative English event post in 3 sentences.\n"
        "Sentence 1: event name, date, and location.\n"
        "Sentence 2: 1–2 concrete things visitors can do or experience.\n"
        "Use factual and specific wording.\n"
        f"Keep it under {MAX_CHARS}.\n"
        "Use the factual event title, date, venue, and 1 clear attraction.\n"
        "Do not use vague phrases like 'Check the official page' unless needed.\n"
        "Do not repeat the title twice.\n"
    )
    user = f"""Write one short English event post for this event.

JSON schema:
{{"text": "..."}}

Facts:
title: {event.title}
date: {english_date_phrase(event)}
venue: {event.place}
description_ja: {event.description}
official_url: {event.official_url}
"""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if provider == "ollama":
        content = call_ollama(messages)
    else:
        content = call_openai(messages)
    parsed = parse_json_from_llm_output(content)
    text = str(parsed.get("text", "") or "").strip()
    if not text:
        raise EventFeedError("LLM returned empty post text")
    return text[:MAX_CHARS].rstrip()


def build_entry(event: Event, text: str, now_dt: datetime) -> Dict[str, Any]:
    generated_at = now_dt.astimezone(ZoneInfo("UTC")).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    obj: Dict[str, Any] = {
        "date": now_dt.strftime("%Y-%m-%d"),
        "generated_at": generated_at,
        "text": text,
        "place": PLACE,
        "links": [event.official_url or event.detail_url],
        "required_mention": event.title,
        "kind": "event",
        "event": {
            "title": event.title,
            "detail_url": event.detail_url,
            "official_url": event.official_url,
            "start_at": event.start_at.isoformat(),
            "end_at": event.end_at.isoformat(),
            "date_label": event.date_label,
            "place": event.place,
            "description": event.description,
        },
    }
    if AVATAR_IMAGE:
        obj["avatar_image"] = AVATAR_IMAGE
        obj["avatar_url"] = resolve_public_avatar_url(AVATAR_IMAGE)
    return obj

def select_event(events: List[Event], now_dt: datetime) -> Event:
    future_events = [e for e in events if e.end_at >= now_dt]
    if not future_events:
        raise SystemExit("No future Cocoyoko events were found")

    future_events.sort(key=lambda e: (e.start_at, e.title))

    if not EVENT_RANDOMIZE:
        return future_events[0]

    pool = future_events[:EVENT_RANDOM_POOL]
    seed_key = now_dt.strftime("%Y-%m-%d")
    rng = random.Random(seed_key)
    return rng.choice(pool)

def write_outputs(feed_path: Path, latest_path: Path, entry: Dict[str, Any], stamp: str) -> None:
    feed_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    feed_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    snapshot_obj = {
        "generated_at": entry["generated_at"],
        "event": entry["event"],
    }
    snapshot_path = artifact_snapshot_dir(latest_path=latest_path) / f"snapshot_{stamp}.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    tz = ZoneInfo(TZ_NAME)
    now_dt = datetime.now(tz)

    validate_runtime()

    public_dir = artifact_public_dir()
    latest_path = artifact_latest_path(public_dir)
    stamp = now_dt.strftime("%Y%m%d_%H%M%S_%Z")
    feed_path = computed_feed_snapshot_path(stamp, public_dir)

    index_html = fetch(INDEX_URL)
    links = parse_index_links(index_html, INDEX_URL)
    debug(f"Found {len(links)} event detail links")
    links = links[:MAX_DETAIL_LINKS]
    log(f"Trying up to {len(links)} event detail links")

    events: List[Event] = []
    failures: List[str] = []
    for url in links:
        try:
            event = parse_event_detail(url)
            events.append(event)
            debug(f"Parsed event: {event.title} | {event.start_at.date()} | {event.place}")
        except Exception as exc:
            msg = f"{url}: {exc}"
            failures.append(msg)
            log(f"WARN parse failed for {msg}")

    if not events:
        detail = failures[0] if failures else "unknown error"
        raise SystemExit(
            f"No parseable Cocoyoko event pages were found; first_error={detail}"
        )

    chosen = select_event(events, now_dt)
    post_text = summarize_event_for_post(chosen)
    entry = build_entry(chosen, post_text, now_dt)
    write_outputs(feed_path, latest_path, entry, stamp)

    if DEBUG:
        log(json.dumps(entry, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
