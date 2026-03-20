#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Window:
    name: str
    start_hour: int
    end_hour: int
    weekdays: tuple[int, ...]
    slot: str
    label: str


@dataclass(frozen=True)
class Decision:
    should_run: bool
    slot: str
    window_id: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--requested-slot", default="")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--workflow-file", required=True)
    parser.add_argument("--github-api-url", default="https://api.github.com")
    parser.add_argument("--github-token", default="")
    parser.add_argument("--current-run-id", type=int, default=0)
    parser.add_argument("--github-output", default="")
    return parser.parse_args()


def load_windows(config_path: str) -> tuple[str, list[Window]]:
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    tz_name = data.get("timezone", "Asia/Tokyo")
    windows: list[Window] = []
    for raw in data.get("windows", []):
        windows.append(
            Window(
                name=str(raw["name"]),
                start_hour=int(raw["start_hour"]),
                end_hour=int(raw["end_hour"]),
                weekdays=tuple(int(x) for x in raw["weekdays"]),
                slot=str(raw["slot"]),
                label=str(raw["label"]),
            )
        )
    if not windows:
        raise SystemExit("feed_schedule.json: windows is empty")
    return tz_name, windows


def now_local(tz_name: str) -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def active_window(now: datetime, windows: list[Window]) -> Window | None:
    weekday = now.isoweekday()
    hour = now.hour
    for window in windows:
        if weekday not in window.weekdays:
            continue
        if window.start_hour <= hour < window.end_hour:
            return window
    return None


def window_bounds(now: datetime, window: Window) -> tuple[datetime, datetime]:
    start = now.replace(hour=window.start_hour, minute=0, second=0, microsecond=0)

    if window.end_hour == 24:
        end = (start + timedelta(days=1)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
    else:
        end = now.replace(hour=window.end_hour, minute=0, second=0, microsecond=0)

    return start, end


def request_json(url: str, token: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "feed-schedule-gate",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def find_blocking_run(
    *,
    api_url: str,
    repo: str,
    workflow_file: str,
    token: str,
    current_run_id: int,
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[bool, str]:
    if not token:
        return False, "missing_github_token"

    quoted_workflow = urllib.parse.quote(workflow_file, safe="")
    url = (
        f"{api_url}/repos/{repo}/actions/workflows/{quoted_workflow}/runs"
        f"?event=schedule&per_page=100"
    )
    payload = request_json(url, token)
    for run in payload.get("workflow_runs", []):
        run_id = int(run.get("id", 0))
        if current_run_id and run_id == current_run_id:
            continue

        created_at_raw = run.get("created_at")
        if not created_at_raw:
            continue
        created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        if not (start_utc <= created_at < end_utc):
            continue

        status = str(run.get("status") or "")
        conclusion = str(run.get("conclusion") or "")

        if status in {"queued", "in_progress", "pending", "requested", "waiting"}:
            return True, f"already_{status}"
        if conclusion == "success":
            return True, "already_success"

    return False, "no_prior_success"


def decide(args: argparse.Namespace) -> Decision:
    requested_slot = (args.requested_slot or "").strip()
    if requested_slot and requested_slot != "auto":
        return Decision(
            should_run=True,
            slot=requested_slot,
            window_id="manual-explicit",
            reason="manual_explicit_slot",
        )

    tz_name, windows = load_windows(args.config)
    current = now_local(tz_name)
    window = active_window(current, windows)
    if window is None:
        return Decision(
            should_run=False,
            slot="",
            window_id="",
            reason="outside_active_window",
        )

    window_id = f"{current.date().isoformat()}-{window.label}"
    if args.event_name == "workflow_dispatch":
        return Decision(
            should_run=True,
            slot=window.slot,
            window_id=window_id,
            reason="manual_auto_bypass_history",
        )

    start_local, end_local = window_bounds(current, window)
    start_utc = start_local.astimezone(UTC)
    end_utc = end_local.astimezone(UTC)

    blocked, reason = find_blocking_run(
        api_url=args.github_api_url,
        repo=args.repo,
        workflow_file=args.workflow_file,
        token=args.github_token,
        current_run_id=args.current_run_id,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    if blocked:
        return Decision(
            should_run=False,
            slot=window.slot,
            window_id=window_id,
            reason=reason,
        )

    return Decision(
        should_run=True,
        slot=window.slot,
        window_id=window_id,
        reason="scheduled_window_open",
    )


def emit(decision: Decision, github_output: str) -> None:
    pairs = {
        "should_run": "true" if decision.should_run else "false",
        "slot": decision.slot,
        "window_id": decision.window_id,
        "reason": decision.reason,
    }
    text = "".join(f"{k}={v}\n" for k, v in pairs.items())
    if github_output:
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


def main() -> int:
    args = parse_args()
    decision = decide(args)
    emit(decision, args.github_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
