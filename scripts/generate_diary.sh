#!/usr/bin/env bash
set -euo pipefail

# Generate a daily weather tweet using the backend RAG API, and write outputs to
# one or more JSON files (useful for GitHub Pages).
#
# Required env (or defaults below):
#   WEATHER_LAT, WEATHER_LON
#
# Optional env:
#   BACKEND_URL (default: http://localhost:8000)
#   WEATHER_TZ (default: Asia/Tokyo)
#   WEATHER_PLACE (default: empty)
#   RAG_TOP_K (default: 3)
#   TWEET_MAX_CHARS (default: 240)
#   RAG_HASHTAGS (default: empty)
#   RAG_TOKEN (optional bearer token)
#
# Output paths:
#   FEED_PATH / LATEST_PATH (single) OR FEED_PATHS / LATEST_PATHS (colon-separated)
#   Default:
#     FEED_PATH=frontend/app/public/feed.json
#     LATEST_PATH=frontend/app/public/latest.json

API_BASE="${BACKEND_URL:-http://localhost:8000}"

# Location (required)
LAT="${WEATHER_LAT:-}"
LON="${WEATHER_LON:-}"
TZ_NAME="${WEATHER_TZ:-Asia/Tokyo}"
WEATHER_PLACE="${WEATHER_PLACE:-}"

# Tweet config
TOP_K="${RAG_TOP_K:-3}"
MAX_CHARS="${TWEET_MAX_CHARS:-240}"
HASHTAGS="${RAG_HASHTAGS:-}"

# Output paths
FEED_PATH="${FEED_PATH:-frontend/app/public/feed.json}"
LATEST_PATH="${LATEST_PATH:-frontend/app/public/latest.json}"

# Support both single-path vars (FEED_PATH/LATEST_PATH) and multi-path vars (FEED_PATHS/LATEST_PATHS).
FEED_PATHS="${FEED_PATHS:-}"
if [[ -z "${FEED_PATHS//[[:space:]]/}" ]]; then FEED_PATHS="${FEED_PATH}"; fi
LATEST_PATHS="${LATEST_PATHS:-}"
if [[ -z "${LATEST_PATHS//[[:space:]]/}" ]]; then LATEST_PATHS="${LATEST_PATH}"; fi

RAG_TOKEN="${RAG_TOKEN:-}"

if [[ -z "${LAT}" || -z "${LON}" ]]; then
  echo "ERROR: WEATHER_LAT and WEATHER_LON must be set." >&2
  echo "Example: export WEATHER_LAT=35.2810 WEATHER_LON=139.6720" >&2
  exit 1
fi

echo "API_BASE: ${API_BASE}"
echo "WEATHER: lat=${LAT} lon=${LON} tz=${TZ_NAME} place='${WEATHER_PLACE}'"
echo "TOP_K=${TOP_K} MAX_CHARS=${MAX_CHARS}"
echo "FEED_PATHS=${FEED_PATHS}"
echo "LATEST_PATHS=${LATEST_PATHS}"

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

  local body
  body="$(curl -sS --retry 3 --retry-delay 2 --max-time 90 -X "${method}" \
    -H "Content-Type: application/json" \
    "${auth_header[@]}" \
    -d "${payload}" \
    "${url}" || true)"

  if [[ -z "${body}" ]]; then
    echo "ERROR: backend response body is empty (${url})" >&2
    return 1
  fi

  python - <<'PY' "${body}"
import json, sys
raw = sys.argv[1]
try:
    json.loads(raw)
except Exception as e:
    print("ERROR: backend response is not JSON:", e, file=sys.stderr)
    print("---- raw response ----", file=sys.stderr)
    print(raw, file=sys.stderr)
    raise SystemExit(1)
print(raw)
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
  --place "${WEATHER_PLACE}")"

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
fi

# Warm up once (helps avoid cold-start timeouts)
curl -fsS -X POST -H "Content-Type: application/json" \
  -d '{"question":"ping","top_k":1,"extra_context":"{}","use_live_weather":false}' \
  "${API_BASE}/rag/query" >/dev/null 2>&1 || true

# -----------------------------------------------------------------------------
# 3) Query backend for today's tweet
# -----------------------------------------------------------------------------
QUESTION=$'Write exactly ONE short tweet-style post about TODAY\\x27s weather.\\n'\
$'Use ONLY information from the provided live weather JSON.\\n'\
$'Keep it within about '"${MAX_CHARS}"' characters.\\n'\
$'Output ONLY the tweet text (no quotes, no markdown).\\n'

JSON_PAYLOAD="$(python - <<'PY'
import json, os
payload = {
  "question": os.environ["QUESTION"],
  "top_k": int(os.environ.get("TOP_K", "3")),
  "extra_context": os.environ.get("SNAP_JSON", ""),
  "use_live_weather": False,
}
print(json.dumps(payload, ensure_ascii=False))
PY
)"

resp="$(curl_json POST "${API_BASE}/rag/query" "${JSON_PAYLOAD}")"
tweet="$(python - <<'PY' "${resp}"
import json,sys,re
obj=json.loads(sys.argv[1])
ans=(obj.get("answer") or "").strip()
ans=re.sub(r"\\s+"," ",ans).strip()
print(ans)
PY
)"

if [[ -z "${tweet}" ]]; then
  echo "ERROR: tweet is empty after parsing." >&2
  echo "---- raw response ----" >&2
  echo "${resp}" >&2
  exit 1
fi

if [[ -n "${HASHTAGS}" && "${tweet}" != *"#"* ]]; then
  tweet="${tweet} ${HASHTAGS}"
fi

# -----------------------------------------------------------------------------
# 4) Write outputs (feed + latest + snapshot) to all configured paths
# -----------------------------------------------------------------------------
today="$(date -u +%Y-%m-%d)"
now_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

ENTRY_JSON="$(python - <<'PY'
import json, os
snap_raw = os.environ.get("SNAP_JSON","{}")
try:
    snap = json.loads(snap_raw)
except Exception:
    snap = {"raw": snap_raw}
entry = {
  "date": os.environ["today"],
  "generated_at": os.environ["now_iso"],
  "text": os.environ["tweet"],
  "place": os.environ.get("WEATHER_PLACE",""),
  "weather": snap,
}
print(json.dumps(entry, ensure_ascii=False, indent=2) + "\\n")
PY
)"

write_feed_and_latest() {
  local feed_path="${1}"
  local latest_path="${2}"

  mkdir -p "$(dirname "${feed_path}")" "$(dirname "${latest_path}")"

  # Write latest
  printf "%s" "${ENTRY_JSON}" > "${latest_path}"

  # Update feed (list)
  python - <<'PY' "${feed_path}" "${ENTRY_JSON}"
import json, sys
from pathlib import Path

feed_path = Path(sys.argv[1])
entry_txt = sys.argv[2]
entry = json.loads(entry_txt)

if feed_path.exists() and feed_path.stat().st_size > 0:
    try:
        feed = json.loads(feed_path.read_text(encoding="utf-8"))
        if not isinstance(feed, list):
            feed = []
    except Exception:
        feed = []
else:
    feed = []

# replace today's entry
feed = [e for e in feed if e.get("date") != entry.get("date")]
feed.append(entry)
feed.sort(key=lambda x: x.get("date",""))

feed_path.write_text(json.dumps(feed, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(f"Wrote: {feed_path} ({len(feed)} entries)")
PY

  # Also write weather snapshot next to latest (for debugging / transparency)
  local snap_path
  snap_path="$(dirname "${latest_path}")/weather_snapshot.json"
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
