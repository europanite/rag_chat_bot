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

echo "Waiting for backend /rag/status ..."
for i in {1..60}; do
  STATUS_JSON="$(curl -sS "${API_BASE}/rag/status" || true)"
  if echo "$STATUS_JSON" | python -c 'import json,sys; d=json.load(sys.stdin); print(d.get("json_files"), d.get("chunks_in_store"))' >/dev/null 2>&1; then
    echo "OK: /rag/status"
    break
  fi
  sleep 2
done

echo "status: ${STATUS_JSON}"

JSON_FILES="$(echo "$STATUS_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("json_files",0))')"
CHUNKS="$(echo "$STATUS_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("chunks_in_store",0))')"

if [ "${JSON_FILES}" -le 0 ]; then
  echo "ERROR: No rag docs found (json_files=0). Is rag_docs committed and mounted?"
  exit 1
fi

if [ "${CHUNKS}" -le 0 ]; then
  echo "No chunks in store -> POST /rag/reindex"
  curl -fsS -X POST "${API_BASE}/rag/reindex" -H "Content-Type: application/json" -d '{}' >/dev/null

  echo "Waiting for chunks_in_store > 0 ..."
  for i in {1..120}; do
    STATUS_JSON="$(curl -sS "${API_BASE}/rag/status" || true)"
    CHUNKS="$(echo "$STATUS_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("chunks_in_store",0))' 2>/dev/null || echo 0)"
    [ "${CHUNKS}" -gt 0 ] && break
    sleep 2
  done

  if [ "${CHUNKS}" -le 0 ]; then
    echo "ERROR: reindex did not populate chunks. status=${STATUS_JSON}"
    exit 1
  fi
fi

# --- wait for backend + rag to be truly ready ---
echo "Waiting for backend /rag/status ..."
for i in {1..300}; do
  if curl -fsS "${API_BASE}/rag/status" >/dev/null; then
    echo "OK: /rag/status"
    break
  fi
  sleep 2
done

echo "Warming up /rag/query ..."
WARM_PAYLOAD='{"question":"ping","use_live_weather":false}'

for i in {1..300}; do
  RES_WARM="$(curl -sS --retry 5 --retry-all-errors --retry-delay 2 \
    "${API_BASE}/rag/query" -H "Content-Type: application/json" -d "${WARM_PAYLOAD}" || true)"
  python - <<'PY' <<<"${RES_WARM}" && break || true
import json,sys
obj=json.loads(sys.stdin.read())
assert "answer" in obj
PY
  sleep 2
done
# --- end wait ---

# Call backend (retry a bit in case ollama/model warmup is slow)
RES=""
ok_json=0
for i in {1..20}; do
  RES="$(curl -sS --retry 5 --retry-all-errors --retry-delay 2 \
    "${QUERY_URL}" -H "Content-Type: application/json" -d "${JSON_PAYLOAD}" || true)"
  python - <<'PY' <<<"${RES}" && { ok_json=1; break; } || true
import json,sys
obj=json.loads(sys.stdin.read())
assert isinstance(obj, dict)
assert "answer" in obj
PY
  echo "query not ready (attempt ${i}/20). raw_len=${#RES}" >&2
  sleep 3
done

if [ "${ok_json}" -ne 1 ]; then
  echo "ERROR: /rag/query never returned valid JSON with 'answer'." >&2
  echo "---- raw response ----" >&2
  echo "${RES}" >&2
  exit 1
fi

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

if [[ -z "${TWEET}" ]]; then
  echo "ERROR: tweet is empty after parsing." >&2
  exit 1
fi

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
