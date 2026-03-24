#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_WEEKDAY_GROUPS: dict[str, tuple[int, ...]] = {
    "weekday": (1, 2, 3, 4, 5),
    "weekend": (6, 7),
    "all": (1, 2, 3, 4, 5, 6, 7),
}

WEEKDAY_NAMES: dict[int, str] = {
    1: "mon",
    2: "tue",
    3: "wed",
    4: "thu",
    5: "fri",
    6: "sat",
    7: "sun",
}


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


def _normalize_weekday_groups(raw_groups: dict[str, object]) -> dict[str, tuple[int, ...]]:
    groups = dict(DEFAULT_WEEKDAY_GROUPS)
    for name, weekdays in raw_groups.items():
        groups[str(name)] = tuple(int(day) for day in list(weekdays))
    return groups


def _resolve_weekdays(raw_value: object, groups: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    if raw_value is None:
        return groups["all"]
    if isinstance(raw_value, str):
        if raw_value not in groups:
            raise SystemExit(f"feed_schedule.json: unknown weekday group '{raw_value}'")
        return groups[raw_value]
    return tuple(int(day) for day in list(raw_value))


def _weekday_tag(weekdays: tuple[int, ...], groups: dict[str, tuple[int, ...]]) -> str:
    for group_name, group_days in groups.items():
        if weekdays == group_days:
            return group_name
    if len(weekdays) == 1:
        return WEEKDAY_NAMES.get(weekdays[0], str(weekdays[0]))
    return "-".join(WEEKDAY_NAMES.get(day, str(day)) for day in weekdays)


def _append_window(
    windows: list[Window],
    *,
    name: str,
    start_hour: int,
    end_hour: int,
    weekdays: tuple[int, ...],
    slot: str,
    groups: dict[str, tuple[int, ...]],
    label: str = "",
) -> None:
    window_label = label or f"{name}-{_weekday_tag(weekdays, groups)}-{slot}"
    windows.append(
        Window(
            name=name,
            start_hour=start_hour,
            end_hour=end_hour,
            weekdays=weekdays,
            slot=slot,
            label=window_label,
        )
    )


def _build_windows_from_profiles(data: dict[str, object]) -> list[Window]:
    groups = _normalize_weekday_groups(dict(data.get("weekday_groups") or {}))
    raw_profiles = dict(data.get("profiles") or {})
    if not raw_profiles:
        raise SystemExit("feed_schedule.json: either windows or profiles must be configured")

    windows: list[Window] = []
    for profile_name, profile_value in raw_profiles.items():
        profile = dict(profile_value)
        profile_weekdays = _resolve_weekdays(profile.get("weekdays", profile_name), groups)

        for block_value in list(profile.get("windows") or []):
            block = dict(block_value)
            name = str(block["name"])
            start_hour = int(block["start_hour"])
            end_hour = int(block["end_hour"])
            block_weekdays = _resolve_weekdays(block.get("weekdays"), groups)
            active_weekdays = tuple(day for day in profile_weekdays if day in block_weekdays)
            if not active_weekdays:
                continue

            explicit_label = str(block.get("label") or "")
            slot = str(block.get("slot") or "")
            slot_by_weekday = dict(block.get("slot_by_weekday") or {})

            if slot:
                _append_window(
                    windows,
                    name=name,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    weekdays=active_weekdays,
                    slot=slot,
                    groups=groups,
                    label=explicit_label,
                )
                continue

            if slot_by_weekday:
                for day in active_weekdays:
                    slot_for_day = str(slot_by_weekday.get(str(day)) or "")
                    if not slot_for_day:
                        continue
                    _append_window(
                        windows,
                        name=name,
                        start_hour=start_hour,
                        end_hour=end_hour,
                        weekdays=(day,),
                        slot=slot_for_day,
                        groups=groups,
                        label=explicit_label,
                    )
                continue

            raise SystemExit(
                f"feed_schedule.json: profile window '{name}' must define slot or slot_by_weekday"
            )

    return windows


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
        windows = _build_windows_from_profiles(data)

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


def resolve_slot(slot_expr: str, *, seed: str) -> str:
    choices = [part.strip() for part in slot_expr.split("|") if part.strip()]
    if not choices:
        return ""
    if len(choices) == 1:
        return choices[0]

    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(choices)
    return choices[idx]


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
    resolved_slot = resolve_slot(window.slot, seed=window_id)
    if args.event_name == "workflow_dispatch":
        return Decision(
            should_run=True,
            slot=resolved_slot,
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
            slot=resolved_slot,
            window_id=window_id,
            reason=reason,
        )

    return Decision(
        should_run=True,
        slot=resolved_slot,
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
