# Feed Scheduling (3x Daily, Profile-Based)

This document explains the simplified feed scheduling design for `rag_chat_bot`.

It is intended to accompany the profile-based schedule patch that reduces feed delivery to **3 windows per day** and makes the schedule easier to manage than the previous flat `windows[]` list.

---

## Goal

The previous schedule was hard to maintain because it relied on a long flat list of time windows and weekday-specific entries.

This design changes the schedule to:

- **3 deliveries per day** instead of a broader 4-window layout
- **Different weekday and weekend timing**
- **Profile-based configuration** so operators edit a few blocks instead of many repetitive rows
- **Backward-compatible gate logic** so old `windows[]` configs still work

---

## Files

### 1. `.github/feed_schedule.json`
Defines when each feed slot is allowed to run.

### 2. `scripts/feed_schedule_gate.py`
Loads the schedule, determines whether the current time is inside an active window, and blocks duplicate scheduled runs inside the same window.

### 3. `.github/workflows/feed.yml`
Runs on an hourly cron schedule and delegates the final decision to `feed_schedule_gate.py`.

---

## High-level flow

1. GitHub Actions starts `feed.yml` every hour.
2. `feed_schedule_gate.py` reads `.github/feed_schedule.json`.
3. The gate finds the active local-time window in `Asia/Tokyo`.
4. If the current time is outside all windows, the run is skipped.
5. If the current time is inside a window, the gate resolves the slot for that weekday.
6. The gate checks whether another scheduled run already succeeded in the same window.
7. If not blocked, the selected slot workflow proceeds.

This means **the workflow may still run every hour, but actual publishing is constrained by the schedule gate**.

---

## New schedule policy

### Weekdays

- **Morning:** `07:00-09:00`
- **Midday:** `12:00-13:00`
- **Night:** `20:00-22:00`

### Weekends

- **Morning:** `10:00-11:00`
- **Midday:** `15:00-17:00`
- **Night:** `20:00-22:00`

This is intentionally shaped around a simple rule:

- **Morning** = utility / intro
- **Midday** = light discovery
- **Night** = strongest content

---

## Example slot strategy

### Weekdays

| Window | Monday | Tuesday | Wednesday | Thursday | Friday |
|---|---|---|---|---|---|
| Morning | `garbage` | `spot_intro` | `garbage` | `spot_intro` | `garbage` |
| Midday | `restaurant` | `google_random` | `restaurant` | `google_random` | `restaurant` |
| Night | `event` | `spot` | `activity` | `event` | `spot_intro` |

### Weekends

| Window | Saturday | Sunday |
|---|---|---|
| Morning | `spot` | `spot` |
| Midday | `restaurant` | `google_random` |
| Night | `activity` | `event` |

This gives variation without forcing operators to manage a long per-day flat list.

---

## New JSON format

The new configuration supports three top-level sections:

```json
{
  "timezone": "Asia/Tokyo",
  "weekday_groups": { ... },
  "profiles": { ... }
}
```

### `timezone`
The local timezone used by the gate.

### `weekday_groups`
Named weekday sets.

Example:

```json
{
  "weekday": [1, 2, 3, 4, 5],
  "weekend": [6, 7]
}
```

### `profiles`
A small number of reusable scheduling profiles.

Each profile has:

- `weekdays`: either a weekday group name or an explicit array
- `windows`: the windows belonging to that profile

Each window supports either:

- `slot`: one fixed slot for all active weekdays in that block
- `slot_by_weekday`: weekday-specific slot overrides

---

## Example configuration

```json
{
  "timezone": "Asia/Tokyo",
  "weekday_groups": {
    "weekday": [1, 2, 3, 4, 5],
    "weekend": [6, 7]
  },
  "profiles": {
    "weekday": {
      "weekdays": "weekday",
      "windows": [
        {
          "name": "morning",
          "start_hour": 7,
          "end_hour": 9,
          "slot_by_weekday": {
            "1": "garbage",
            "2": "spot_intro",
            "3": "garbage",
            "4": "spot_intro",
            "5": "garbage"
          }
        },
        {
          "name": "midday",
          "start_hour": 12,
          "end_hour": 13,
          "slot_by_weekday": {
            "1": "restaurant",
            "2": "google_random",
            "3": "restaurant",
            "4": "google_random",
            "5": "restaurant"
          }
        },
        {
          "name": "night",
          "start_hour": 20,
          "end_hour": 22,
          "slot_by_weekday": {
            "1": "event",
            "2": "spot",
            "3": "activity",
            "4": "event",
            "5": "spot_intro"
          }
        }
      ]
    },
    "weekend": {
      "weekdays": "weekend",
      "windows": [
        {
          "name": "morning",
          "start_hour": 10,
          "end_hour": 11,
          "slot": "spot"
        },
        {
          "name": "midday",
          "start_hour": 15,
          "end_hour": 17,
          "slot_by_weekday": {
            "6": "restaurant",
            "7": "google_random"
          }
        },
        {
          "name": "night",
          "start_hour": 20,
          "end_hour": 22,
          "slot_by_weekday": {
            "6": "activity",
            "7": "event"
          }
        }
      ]
    }
  }
}
```

---

## Why this format is easier to manage

### Old style

The old style required many repeated entries like this:

- one row per window
- often one row per weekday
- frequent duplication of `name`, `start_hour`, and `end_hour`

### New style

The new style groups schedule logic by intent:

- one `weekday` profile
- one `weekend` profile
- three windows per profile
- weekday-specific variation only where needed

This makes it easier to answer:

- “What happens on weekday mornings?”
- “What time does weekend midday run?”
- “Which slot is used on Sunday night?”

without scanning a long flat array.

---

## Backward compatibility

`feed_schedule_gate.py` supports both formats:

### Legacy format

```json
{
  "timezone": "Asia/Tokyo",
  "windows": [ ... ]
}
```

### New format

```json
{
  "timezone": "Asia/Tokyo",
  "weekday_groups": { ... },
  "profiles": { ... }
}
```

If `windows[]` is present and non-empty, the gate can continue to use it.
If not, the gate builds runtime windows from `profiles`.

---

## Naming rules

The gate auto-generates labels when a window does not define one explicitly.

Pattern:

```text
{name}-{weekday_tag}-{slot}
```

Examples:

- `morning-weekday-garbage`
- `midday-weekend-restaurant`
- `night-mon-event`

This keeps labels predictable without forcing verbose JSON.

---

## Common operations

### Change weekday night timing

Edit only this block in the `weekday` profile:

```json
{
  "name": "night",
  "start_hour": 20,
  "end_hour": 22,
  ...
}
```

### Make all weekend mornings use another slot

Edit the weekend morning block:

```json
{
  "name": "morning",
  "start_hour": 10,
  "end_hour": 11,
  "slot": "spot"
}
```

### Change only Sunday night

Edit only this part:

```json
"slot_by_weekday": {
  "6": "activity",
  "7": "event"
}
```

---

## Validation

After changing the schedule, validate both JSON and Python.

### 1. Python compile check

```bash
python -m py_compile scripts/feed_schedule_gate.py
```

### 2. Manual gate run

```bash
python scripts/feed_schedule_gate.py \
  --config .github/feed_schedule.json \
  --event-name workflow_dispatch \
  --requested-slot auto \
  --repo dummy/repo \
  --workflow-file feed.yml
```

### 3. Optional JSON formatting check

```bash
python -m json.tool .github/feed_schedule.json > /dev/null
```

---

## Operational notes

- The workflow still runs hourly.
- The gate decides whether a publish window is currently active.
- Scheduled duplicates inside the same window are blocked.
- Manual runs can still be used for testing.
- The schedule is intentionally conservative to avoid over-posting.

---

## Recommended next step

This version improves **maintainability** first.

If you want more variety later, the next step is not to add more windows again. The next step is to extend the gate with controlled rotation logic such as:

- weighted slot selection
- cooldown by slot
- daily cap enforcement
- no-repeat rules for recent slot history

That should be added on top of this profile-based structure, not by returning to a long flat schedule list.

---

## Summary

This redesign simplifies feed scheduling by replacing a large flat time-window list with a small number of readable profiles.

The result is:

- **3 deliveries per day**
- **clear weekday/weekend behavior**
- **less repetitive JSON**
- **easier future maintenance**
- **backward compatibility with the previous schedule format**
