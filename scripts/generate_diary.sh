#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true

dump_debug() {
  {
    echo "---- docker compose ps ----"
    docker compose ps || true
    echo "---- backend logs ----"
    docker compose logs --no-color --tail=200 backend || true
    echo "---- ollama logs ----"
    docker compose logs --no-color --tail=200 ollama || true
  } >&2
}
trap dump_debug ERR

# Use JST by default (the workflow runs at 00:00 UTC = 09:00 JST)
TZ_NAME="${WEATHER_TZ:-Asia/Tokyo}"
TODAY="${TODAY:-$(TZ="${TZ_NAME}" date +%F)}"

OUT_DIR="${OUT_DIR:-frontend/app/public}"
POST_DIR="${POST_DIR:-${OUT_DIR}/posts}"
POST="${POST_DIR}/${TODAY}-weather.md"
FEED_PATH="${FEED_PATH:-${OUT_DIR}/weather_feed.json}"
LATEST_PATH="${LATEST_PATH:-${OUT_DIR}/latest.json}"
mkdir -p "${POST_DIR}" "$(dirname "${FEED_PATH}")" "$(dirname "${LATEST_PATH}")"

IFS=',' read -r -a FEEDS <<< "${FEED_PATHS}"
IFS=',' read -r -a LATESTS <<< "${LATEST_PATHS}"

API_BASE="${API_BASE:-http://localhost:${BACKEND_PORT:-8000}}"

LAT="${WEATHER_LAT:-35.2810}"
LON="${WEATHER_LON:-139.6722}"
TZ_NAME="${WEATHER_TZ:-Asia/Tokyo}"

# Warmup /rag/status
echo "Waiting for backend /rag/status ..."
for i in {1..120}; do
  if curl -fsS "${API_BASE}/rag/status" >/dev/null; then
    echo "OK: /rag/status"
    break
  fi
  sleep 1
done

STATUS="$(curl -fsS "${API_BASE}/rag/status")"
echo "status: ${STATUS}"

# If no chunks yet, trigger reindex (best-effort)
chunks_in_store="$(python - <<'PY'
import json, sys
try:
    obj=json.loads(sys.stdin.read())
except Exception:
    print(0); sys.exit(0)
print(int(obj.get("chunks_in_store") or 0))
PY
<<<"${STATUS}")"

if [ "${chunks_in_store}" -le 0 ]; then
  echo "No chunks in store -> POST /rag/reindex"
  curl -fsS -X POST "${API_BASE}/rag/reindex" >/dev/null || true

  echo "Waiting for chunks_in_store > 0 ..."
  for i in {1..120}; do
    s="$(curl -fsS "${API_BASE}/rag/status" || true)"
    c="$(python - <<'PY'
import json, sys
raw=sys.stdin.read().strip()
try:
    obj=json.loads(raw)
except Exception:
    print(0); sys.exit(0)
print(int(obj.get("chunks_in_store") or 0))
PY
<<<"${s}")"
    if [ "${c}" -gt 0 ]; then
      echo "OK: chunks_in_store=${c}"
      break
    fi
    sleep 1
  done
fi

echo "Warming up /rag/query ..."
WARM_URL="${API_BASE}/rag/query"
curl -fsS -H "Content-Type: application/json" -d '{"query":"hello","top_k":1}' "${WARM_URL}" >/dev/null || true

# Query (include lat/lon in query params)
QUERY_URL="${API_BASE}/rag/query?lat=${LAT}&lon=${LON}"
BODY="$(cat <<EOF
{
  "query": "Write today's short weather tweet. Include temperature and conditions. Keep it friendly and local."
}
EOF
)"

# Retry loop (models may warm up)
ok_json=0
RES=""
for i in {1..5}; do
  RES="$(curl -sS -H "Content-Type: application/json" -d "${BODY}" "${QUERY_URL}" || true)"

  # Validate: must be JSON and include non-empty 'answer'
  ok_json="$(python - <<'PY'
import json, sys
raw = sys.stdin.read().strip()
try:
    obj = json.loads(raw)
except Exception:
    print(0); sys.exit(0)
ans = (obj.get("answer") or "").strip()
print(1 if ans else 0)
PY
<<<"${RES}")"

  if [ "${ok_json}" -eq 1 ]; then
    break
  fi
  sleep 2
done

if [ "${ok_json}" -ne 1 ]; then
  echo "ERROR: /rag/query never returned valid JSON with non-empty 'answer'." >&2
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

# Update feed(s)
for idx in "${!FEEDS[@]}"; do
  feed="$(echo "${FEEDS[$idx]}" | xargs)"
  latest="$(echo "${LATESTS[$idx]:-${LATESTS[0]}}" | xargs)"
  python scripts/update_weather_feed.py \
    --feed "${feed}" \
    --latest "${latest}" \
    --date "${TODAY}" \
    --text "${TWEET}" \
    --place "${WEATHER_PLACE:-}"
done

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
echo "Updated feeds: ${FEED_PATHS} | latest: ${LATEST_PATHS}"
