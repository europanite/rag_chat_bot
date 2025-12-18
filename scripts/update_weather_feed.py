#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
  # Keep it simple: UTC ISO string; UI can display as-is.
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
  if not path.exists():
    return None
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return None


def _atomic_write(path: Path, obj: Any) -> None:
  """Write JSON atomically.

  Use a *unique* temp file in the same directory.
  This avoids collisions with stale *.tmp files that may be left behind
  (e.g., if a previous run crashed or created root-owned temp files).
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  data = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"

  fd, tmp_name = tempfile.mkstemp(
    prefix=path.name + ".",
    suffix=".tmp",
    dir=str(path.parent),
  )
  tmp = Path(tmp_name)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      f.write(data)
      f.flush()
      os.fsync(f.fileno())
    tmp.replace(path)
  finally:
    # If replace() succeeded, tmp path no longer exists.
    try:
      if tmp.exists():
        tmp.unlink()
    except Exception:
      pass


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--feed", default="public/weather_feed.json")
  ap.add_argument("--latest", default="public/latest.json")
  ap.add_argument("--date", required=True, help="YYYY-MM-DD")
  ap.add_argument("--text", required=True)
  ap.add_argument("--place", default="")
  ap.add_argument("--limit", type=int, default=365)
  args = ap.parse_args()

  feed_path = Path(args.feed)
  latest_path = Path(args.latest)

  feed = _load_json(feed_path) or {"items": []}
  items: List[Dict[str, Any]] = list(feed.get("items") or [])

  entry: Dict[str, Any] = {
    "id": args.date,
    "date": args.date,
    "text": args.text.strip(),
  }
  if args.place:
    entry["place"] = args.place

  # Upsert by id/date
  items = [it for it in items if str(it.get("id")) != args.date and str(it.get("date")) != args.date]
  items.append(entry)

  # Sort desc by date string
  items.sort(key=lambda x: str(x.get("date", "")), reverse=True)
  items = items[: max(1, args.limit)]

  feed["items"] = items
  feed["updated_at"] = _now_iso()
  if args.place:
    feed["place"] = args.place

  _atomic_write(feed_path, feed)
  _atomic_write(latest_path, {"date": args.date, "text": entry["text"], "place": args.place, "updated_at": feed["updated_at"]})
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
