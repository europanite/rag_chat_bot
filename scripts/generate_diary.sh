#!/usr/bin/env bash
set -euo pipefail

# Use JST by default (the workflow runs at 00:00 UTC = 09:00 JST)
TZ_NAME="${WEATHER_TZ:-Asia/Tokyo}"
TODAY="${TODAY:-$(TZ="${TZ_NAME}" date +%F)}"

mkdir -p _posts public

POST="_posts/${TODAY}-weather.md"
FEED_PATH="${FEED_PATH:-public/weather_feed.json}"
LATEST_PATH="${LATEST_PATH:-public/latest.json}"

API_BASE="${API_BASE:-http://localhost:${BACKEND_PORT:-8000}}"

# Personality + constraints for the bot (English, friendly, Yokosuka locals + visitors)
QUESTION="${QUESTION:-Write ONE short, friendly weather tweet for today.
You are a cheerful Yokosuka weather bot speaking to locals and tourists.
- Write in English.
- 1 tweet only (no numbering).
- 200 characters max.
- Mention Yokosuka and suggest ONE light activity (sea walk, port, hiking, etc.) if appropriate.
- Keep it upbeat, not salesy.
- Use 0-2 emojis (optional).
- Avoid citations, sources, and debugging text.
}"

export QUESTION

JSON_PAYLOAD="$(python - <<'PY'
import json, os
q = os.environ["QUESTION"]
print(json.dumps({"question": q, "use_live_weather": True}))
PY
)"

# Build query URL.
# IMPORTANT: env vars set in the workflow step won't reach the *already-running* backend container.
# So, when WEATHER_LAT/WEATHER_LON are available, pass them as query params.
QUERY_URL="${API_BASE}/rag/query"
if [[ -n "${WEATHER_LAT:-}" && -n "${WEATHER_LON:-}" ]]; then
  PLACE_Q=""
  if [[ -n "${WEATHER_PLACE:-}" ]]; then
    PLACE_Q="$(python - <<'PY'
import os, urllib.parse
print(urllib.parse.quote(os.environ.get("WEATHER_PLACE","")))
PY
)"
  fi
  QUERY_URL="${QUERY_URL}?lat=${WEATHER_LAT}&lon=${WEATHER_LON}"
  if [[ -n "${PLACE_Q}" ]]; then
    QUERY_URL="${QUERY_URL}&place=${PLACE_Q}"
  fi
fi

# Call backend (retry a bit in case ollama/model warmup is slow)
RES=""
for i in {1..8}; do
  if RES="$(curl -fsS "${QUERY_URL}" -H "Content-Type: application/json" -d "${JSON_PAYLOAD}")"; then
    break
  fi
  echo "curl failed (attempt ${i}/8). Retrying..." >&2
  sleep 3
done

if [[ -z "${RES}" ]]; then
  echo "ERROR: Backend returned an empty response from ${QUERY_URL}" >&2
  exit 1
fi

TWEET="$(python - <<'PY'
import json, sys
raw = sys.stdin.read().strip()
try:
    obj = json.loads(raw)
except Exception as e:
    print("ERROR: backend response is not JSON:", e, file=sys.stderr)
    print("---- raw response ----", file=sys.stderr)
    print(raw, file=sys.stderr)
    sys.exit(2)

# Support both success and error JSON.
if "answer" not in obj:
    print("ERROR: backend response did not include 'answer'.", file=sys.stderr)
    print(json.dumps(obj, ensure_ascii=False, indent=2), file=sys.stderr)
    sys.exit(3)

ans = (obj.get("answer") or "").strip()
# some models wrap quotes; strip one layer
if (ans.startswith('"') and ans.endswith('"')) or (ans.startswith("'") and ans.endswith("'")):
    ans = ans[1:-1].strip()

if not ans:
    print("ERROR: 'answer' is empty.", file=sys.stderr)
    print(json.dumps(obj, ensure_ascii=False, indent=2), file=sys.stderr)
    sys.exit(4)

print(ans)
PY
<<<"${RES}")"

# Update JSON feed for the frontend
python scripts/update_weather_feed.py   --feed "${FEED_PATH}"   --latest "${LATEST_PATH}"   --date "${TODAY}"   --text "${TWEET}"   --place "${WEATHER_PLACE:-}"

# Optional: keep a markdown diary post too
cat > "${POST}" <<EOF
---
layout: post
title: "Weather (Yokosuka) - ${TODAY}"
date: ${TODAY} 09:00:00 +0900
categories: weather
---

${TWEET}

EOF

echo "Wrote ${POST}"
echo "Updated ${FEED_PATH} and ${LATEST_PATH}"
