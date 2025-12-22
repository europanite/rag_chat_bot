#!/usr/bin/env bash
set -euo pipefail

# Generate a tweet using the backend RAG API, and write outputs to
# one or more JSON files (useful for GitHub Pages).
#
# Required env (or defaults below):
#   LAT, LON
#
# Optional env:
#   BACKEND_URL (default: http://localhost:8000)
#   TZ_NAME (default: Asia/Tokyo)
#   PLACE (default: empty)
#   RAG_TOP_K (default: 3)
#   HASHTAGS (default: empty)
#   RAG_TOKEN (optional bearer token)
#   DEBUG (default: 0)  # set 1 to print extra debug info
#
# Output paths:
#   FEED_PATH / LATEST_PATH (single) OR FEED_PATHS / LATEST_PATHS (colon-separated)

DEBUG="${DEBUG:-0}"
CURL_MAX_TIME="${CURL_MAX_TIME:-16}" 
CURL_RETRIES="${CURL_RETRIES:-2}"  

API_BASE="${BACKEND_URL:-http://localhost:8000}"
DEBUG="${DEBUG:-0}"

QUESTION="${QUESTION:-Write a short weather update (tweet-style) based on today\'s weather in my area.}"
export QUESTION

# Location (required)
LAT="${LAT:-}"
LON="${LON:-}"
TZ_NAME="${TZ_NAME:-Asia/Tokyo}"
PLACE="${PLACE:-}"

# Tweet config
TOP_K="${RAG_TOP_K:-3}"
MAX_CHARS="${MAX_CHARS:-1024}"
HASHTAGS="${HASHTAGS:-}"

echo ${TZ_NAME}

now_local="$(TZ=${TZ_NAME} date +%Y%m%d_%H%M%S_%Z)"
# Timestamp for artifact filenames (use local tz so the filenames match the place)
export now_local

# Output paths
FEED_PATH="${FEED_PATH}/feed/feed_${now_local}.json"
LATEST_PATH="${LATEST_PATH:-frontend/app/public/latest.json}"

# Support both single-path vars (FEED_PATH/LATEST_PATH) and multi-path vars (FEED_PATHS/LATEST_PATHS).
FEED_PATHS="${FEED_PATHS:-}"
if [[ -z "${FEED_PATHS//[[:space:]]/}" ]]; then FEED_PATHS="${FEED_PATH}"; fi
LATEST_PATHS="${LATEST_PATHS:-}"
if [[ -z "${LATEST_PATHS//[[:space:]]/}" ]]; then LATEST_PATHS="${LATEST_PATH}"; fi

RAG_TOKEN="${RAG_TOKEN:-}"

if [[ -z "${LAT}" || -z "${LON}" ]]; then
  echo "ERROR: LAT and LON must be set." >&2
  echo "Example: export LAT=35.2810 LON=139.6720" >&2
  exit 1
fi

echo "API_BASE: ${API_BASE}"
echo "WEATHER: lat=${LAT} lon=${LON} tz=${TZ_NAME} place='${PLACE}'"
echo "TOP_K=${TOP_K} MAX_CHARS=${MAX_CHARS}"
echo "FEED_PATHS=${FEED_PATHS}"
echo "LATEST_PATHS=${LATEST_PATHS}"

# Export values that are referenced from Python heredocs via os.environ / os.getenv.
# (Without export, Python sees nothing -> KeyError / empty context)
export TOP_K
export MAX_CHARS
export PLACE

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
wait_for_backend() {
  local tries="${1:-60}"
  local sleep_s="${2:-2}"

  for ((i=1; i<=tries; i++)); do
    if curl -fsS "${API_BASE}/rag/status" >/dev/null 2>&1; then
      echo "OK: ${API_BASE}/rag/status"
      return 0
    fi
    echo "Waiting for backend /rag/status ... (${i}/${tries})"
    sleep "${sleep_s}"
  done

  echo "ERROR: backend not ready: ${API_BASE}" >&2
  return 1
}

curl_json() {
  local method="${1}"; shift
  local url="${1}"; shift
  local payload="${1}"; shift

  local auth_header=()
  if [[ -n "${RAG_TOKEN}" ]]; then
    auth_header=(-H "Authorization: Bearer ${RAG_TOKEN}")
  fi

  local body="" code=0 attempt=0
  for ((attempt=1; attempt<=CURL_RETRIES+1; attempt++)); do
    set +e
    body="$(curl -sS --max-time "${CURL_MAX_TIME}" \
      -X "${method}" \
      -H "Content-Type: application/json" \
      "${auth_header[@]}" \
      ${payload:+-d "${payload}"} \
      "${url}")"
    code=$?
    set -e

    if [[ "${DEBUG}" == "1" ]]; then
      echo "DEBUG curl_json: attempt=${attempt} code=${code} bytes=${#body} url=${url}" >&2
      [[ "${#body}" -gt 0 ]] && echo "DEBUG curl_json body head: ${body:0:200}" >&2
    fi

    if [[ "${code}" -eq 0 && -n "${body}" ]]; then
      break
    fi
    sleep 2
  done

  if [[ -z "${body}" ]]; then
    echo "ERROR: backend response body is empty (${url})" >&2
    return 1
  fi

  python - <<'PY' "${body}"
import json, sys
raw = sys.argv[1]
def parse_possibly_concatenated_json(s: str):
    s = s.strip()
    if not s:
        raise ValueError("empty body")
    start_candidates = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if start_candidates:
        s = s[min(start_candidates):]
    dec = json.JSONDecoder()
    objs = []
    while s:
        obj, end = dec.raw_decode(s)
        objs.append(obj)
        s = s[end:].lstrip()
    return objs

try:
    objs = parse_possibly_concatenated_json(raw)
    last = objs[-1]
    # If backend returns an error-only JSON (e.g., {"detail": ...}), we still return it;
    # the caller can decide whether to retry or fail.
except Exception as e:
    print(f"ERROR: backend response is not usable JSON: {e}", file=sys.stderr)
    print("---- raw response ----", file=sys.stderr)
    print(raw, file=sys.stderr)
    raise SystemExit(1)

print(json.dumps(last, ensure_ascii=False))
PY
}

split_paths() {
  # Split colon-separated paths into lines
  python - <<'PY' "${1}"
import sys
s = sys.argv[1]
for p in [x.strip() for x in s.split(":") if x.strip()]:
    print(p)
PY
}

# -----------------------------------------------------------------------------
# 1) Fetch weather snapshot (Open-Meteo)
# -----------------------------------------------------------------------------
echo "Fetching weather snapshot (Open-Meteo)..."
SNAP_JSON="$(python scripts/fetch_weather.py \
  --format json \
  --lat "${LAT}" \
  --lon "${LON}" \
  --tz "${TZ_NAME}" \
  --place "${PLACE}")"

# Make sure the live weather JSON is available to Python payload builder.
export SNAP_JSON

if [[ "${DEBUG}" == "1" ]]; then
  echo "DEBUG: SNAP_JSON bytes=$(printf "%s" "${SNAP_JSON}" | wc -c | tr -d ' ')"
fi

# -----------------------------------------------------------------------------
# 2) Ensure backend ready + index
# -----------------------------------------------------------------------------
wait_for_backend

status_json="$(curl -fsS "${API_BASE}/rag/status")"
chunks_in_store="$(python - <<'PY' "${status_json}"
import json,sys
print(json.loads(sys.argv[1]).get("chunks_in_store", 0))
PY
)"
if [[ "${chunks_in_store}" == "0" ]]; then
  echo "No chunks in store -> POST /rag/reindex"
  curl -fsS -X POST "${API_BASE}/rag/reindex" >/dev/null

  # Optional: re-check status for visibility
  if [[ "${DEBUG}" == "1" ]]; then
    status2="$(curl -fsS "${API_BASE}/rag/status" || true)"
    echo "DEBUG: status(after reindex)=${status2}"
  fi
fi

# Warm up once (helps avoid cold-start timeouts)
curl -fsS -X POST -H "Content-Type: application/json" \
  -d '{"question":"ping","top_k":1,"extra_context":"{}","use_live_weather":false}' \
  "${API_BASE}/rag/query" >/dev/null 2>&1 || true

# -----------------------------------------------------------------------------
# 3) Query backend for today's tweet
# -----------------------------------------------------------------------------

QUESTION=$'Start with greeting on time.\n'\
$'Write short tweet-style post.\n'\
about TODAY\x27s weather and events.\n'\
$'Use the live weather JSON for the weather facts.\n'\
$'If RAG context contains events, mention upcoming events suitable for the weather and season.\n'\
$'Show URL if a topic contains it.\n'\
$'Keep it within about '"${MAX_CHARS}"' characters.\n'\
$'Output ONLY the tweet text (no quotes, no markdown).\n'

# THIS is the direct fix for your KeyError: export the bash var so Python can read it.
export QUESTION

JSON_PAYLOAD="$(python - <<'PY'
import json, os, sys

question = os.getenv("QUESTION")
if not question:
    print("ERROR: QUESTION is missing in environment. Did you forget 'export QUESTION'?", file=sys.stderr)
    raise SystemExit(1)

payload = {
  "question": question,
  "top_k": int(os.getenv("TOP_K", "3")),
  "extra_context": os.getenv("SNAP_JSON", ""),
  "use_live_weather": False,
}
print(json.dumps(payload, ensure_ascii=False))
PY
)"

if [[ "${DEBUG}" == "1" ]]; then
  echo "DEBUG: JSON_PAYLOAD=${JSON_PAYLOAD}"
fi

# Query with retries (Ollama cold start / transient backend errors happen)
tweet=""
resp=""
detail=""
for ((qa_attempt=1; qa_attempt<=CURL_RETRIES+2; qa_attempt++)); do
  set +e
  resp="$(curl_json POST "${API_BASE}/rag/query" "${JSON_PAYLOAD}")"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "WARN: /rag/query call failed (attempt ${qa_attempt}/${CURL_RETRIES+2}). Retrying..." >&2
    sleep 2
    continue
  fi

  tweet="$(python - <<'PY' "${resp}" 2>/dev/null || true
import json,sys,re
obj=json.loads(sys.argv[1])
ans=(obj.get("answer") or "").strip()
# Normalize whitespace/newlines
ans=re.sub(r"\s+", " ", ans).strip()
print(ans)
PY
)"
  if [[ -n "${tweet}" ]]; then
    break
  fi

  detail="$(python - <<'PY' "${resp}" 2>/dev/null || true
import json,sys
obj=json.loads(sys.argv[1])
d=obj.get("detail")
if isinstance(d, (list, dict)):
    import json as _json
    print(_json.dumps(d, ensure_ascii=False))
elif d is not None:
    print(str(d))
PY
)"
  echo "WARN: backend did not return an answer (attempt ${qa_attempt}/${CURL_RETRIES+2}). detail=${detail}" >&2
  sleep 2
done

if [[ -z "${tweet}" ]]; then
  echo "ERROR: tweet is empty after parsing (after retries)." >&2
  if [[ -n "${detail}" ]]; then
    echo "Last backend detail: ${detail}" >&2
  fi
  echo "---- raw response ----" >&2
  echo "${resp}" >&2
  exit 1
fi

if [[ -n "${HASHTAGS}" && "${tweet}" != *"#"* ]]; then
  tweet="${tweet} ${HASHTAGS}"
fi

# Export for the ENTRY_JSON builder below (prevents the next KeyError chain).
export tweet

# -----------------------------------------------------------------------------
# 4) Write outputs (feed + latest + snapshot) to all configured paths
# -----------------------------------------------------------------------------
today="$(date -u +%Y-%m-%d)"
now_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export today
export now_iso

ENTRY_JSON="$(python - <<'PY'
import json, os, sys

snap_raw = os.getenv("SNAP_JSON","{}")
try:
    snap = json.loads(snap_raw)
except Exception:
    snap = {"raw": snap_raw}

t = os.getenv("today")
n = os.getenv("now_iso")
tw = os.getenv("tweet")

missing = [k for k,v in [("today", t), ("now_iso", n), ("tweet", tw)] if not v]
if missing:
    print(f"ERROR: missing env vars for ENTRY_JSON: {', '.join(missing)}", file=sys.stderr)
    raise SystemExit(1)

entry = {
  "date": t,
  "generated_at": n,
  "text": tw,
  "place": os.getenv("PLACE",""),
  "weather": snap,
}
print(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")
PY
)"

write_feed_and_latest() {
  local feed_path="${1}"
  local latest_path="${2}"

  mkdir -p "$(dirname "${feed_path}")" "$(dirname "${latest_path}")"

  # Write latest
  printf "%s" "${ENTRY_JSON}" > "${latest_path}"

  # Update feed (object with items)
  python - <<'PY' "${feed_path}" "${ENTRY_JSON}"
import json, sys
from pathlib import Path

feed_path = Path(sys.argv[1])
entry_txt = sys.argv[2]
entry = json.loads(entry_txt)

def to_item(e):
    if not isinstance(e, dict):
        return None
    date = e.get("date") or ""
    text = e.get("text") or ""
    if not date or not text:
        return None
    _id = e.get("id") or e.get("generated_at") or date
    place = e.get("place")
    if place is None:
        place = ""
    return {
        "id": str(_id),
        "date": str(date),
        "text": str(text),
        "place": str(place),
        # keep extra fields for future UI/debugging
        "generated_at": e.get("generated_at"),
        "weather": e.get("weather"),
    }

entry_item = to_item(entry)
if not entry_item:
    raise SystemExit("ERROR: entry JSON missing required fields (date/text)")

feed_obj = {"items": []}
if feed_path.exists() and feed_path.stat().st_size > 0:
    try:
        loaded = json.loads(feed_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("items"), list):
            feed_obj = loaded
        elif isinstance(loaded, list):
            # legacy format: a list of entries
            items = [to_item(x) for x in loaded]
            feed_obj = {
                "items": [i for i in items if i],
                "updated_at": (loaded[-1].get("generated_at") if loaded else None),
                "place": (loaded[-1].get("place") if loaded else ""),
            }
        else:
            feed_obj = {"items": []}
    except Exception:
        feed_obj = {"items": []}

items = feed_obj.get("items") if isinstance(feed_obj.get("items"), list) else []
# replace today
items = [i for i in items if isinstance(i, dict) and i.get("date") != entry_item.get("date")]
items.append(entry_item)
items.sort(key=lambda x: x.get("date", ""))
feed_obj["items"] = items
feed_obj["updated_at"] = entry.get("generated_at")
# set feed-level place once (optional)
if not feed_obj.get("place"):
    feed_obj["place"] = entry.get("place", "")

feed_path.write_text(json.dumps(feed_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote: {feed_path} ({len(items)} entries)")
PY

  # Also write weather snapshot next to latest (for debugging / transparency)
  local snap_path
  snap_path="$(dirname "${latest_path}")/snapshot/snapshot_${now_local}.json"
  printf "%s\n" "${SNAP_JSON}" > "${snap_path}"
  echo "Wrote: ${latest_path}"
  echo "Wrote: ${snap_path}"
}

# Pair paths by index. If counts mismatch, fall back to pairing each feed with the first latest.
mapfile -t FEEDS < <(split_paths "${FEED_PATHS}")
mapfile -t LATESTS < <(split_paths "${LATEST_PATHS}")

if [[ "${#FEEDS[@]}" -eq 0 || "${#LATESTS[@]}" -eq 0 ]]; then
  echo "ERROR: FEED_PATHS / LATEST_PATHS resolved to empty." >&2
  exit 1
fi

for idx in "${!FEEDS[@]}"; do
  feed="${FEEDS[$idx]}"
  latest="${LATESTS[0]}"
  if [[ $idx -lt ${#LATESTS[@]} ]]; then
    latest="${LATESTS[$idx]}"
  fi
  write_feed_and_latest "${feed}" "${latest}"
done

echo "DONE"