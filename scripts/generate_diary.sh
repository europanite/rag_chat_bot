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

# Support both single-path vars (FEED_PATH/LATEST_PATH) and multi-path vars (FEED_PATHS/LATEST_PATHS).
FEED_PATHS="${FEED_PATHS:-}"
if [[ -z "${FEED_PATHS//[[:space:]]/}" ]]; then FEED_PATHS="${FEED_PATH}"; fi
LATEST_PATHS="${LATEST_PATHS:-}"
if [[ -z "${LATEST_PATHS//[[:space:]]/}" ]]; then LATEST_PATHS="${LATEST_PATH}"; fi

IFS=',' read -r -a FEEDS <<< "${FEED_PATHS}"
IFS=',' read -r -a LATESTS <<< "${LATEST_PATHS}"

# Ensure output dirs exist even when FEED_PATHS/LATEST_PATHS contains multiple paths.
for f in "${FEEDS[@]}"; do
  f="$(echo "${f}" | xargs)"
  [[ -n "${f}" ]] && mkdir -p "$(dirname "${f}")"
done
for l in "${LATESTS[@]}"; do
  l="$(echo "${l}" | xargs)"
  [[ -n "${l}" ]] && mkdir -p "$(dirname "${l}")"
done

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

# Read weather snapshot (best-effort)
SNAP_JSON="$(python scripts/fetch_weather.py --format json --lat "${LAT}" --lon "${LON}" --tz "${TZ_NAME}" 2>/dev/null || true)"
if [[ -z "${SNAP_JSON}" ]]; then
  # fallback: allow backend to still generate something
  SNAP_JSON="{}"
fi

QUERY_URL="${API_BASE}/rag/query"
JSON_PAYLOAD="$(python - <<'PY'
import json, os, sys
raw = sys.stdin.read().strip() or "{}"
try:
    snap = json.loads(raw)
except Exception:
    snap = {}
# Keep query consistent and clearly "weather"
place = os.environ.get("WEATHER_PLACE","").strip()
q = "Write a short Japanese weather diary/tweet for today based on this weather JSON. Mention the place if provided."
if place:
    q += f" Place: {place}."
payload = {"question": q, "top_k": 3, "context": json.dumps(snap, ensure_ascii=False)}
print(json.dumps(payload, ensure_ascii=False))
PY
<<<"${SNAP_JSON}")"

ok_json=0
RES=""

# --- wait: ensure /rag/query returns non-empty answer at least once ---
WARM_PAYLOAD='{"question":"hello","top_k":1}'
for i in {1..300}; do
  RES_WARM="$(curl -fsS --retry 5 --retry-all-errors --retry-delay 2 \
  "${API_BASE}/rag/query" -H "Content-Type: application/json" -d "${WARM_PAYLOAD}")"
  python - <<'PY' <<<"${RES_WARM}" && break || true
import json,sys
obj=json.loads(sys.stdin.read())
ans=(obj.get("answer") or "").strip()
if (ans.startswith('"') and ans.endswith('"')) or (ans.startswith("'") and ans.endswith("'")):
  ans=ans[1:-1].strip()
assert ans  # require non-empty tweet
PY
  sleep 2
done
# --- end wait ---

# Call backend (retry a bit in case ollama/model warmup is slow)
for i in {1..20}; do
  set +e
  RES="$(curl -fsS --retry 5 --retry-all-errors --retry-delay 2 \
    --connect-timeout 5 --max-time 180 \
    "${QUERY_URL}" -H "Content-Type: application/json" -d "${JSON_PAYLOAD}" 2>curl_err.txt)"
  rc=$?
  set -e

  if [ $rc -ne 0 ]; then
    echo "curl failed (attempt ${i}/20, rc=${rc})" >&2
    cat curl_err.txt >&2
    sleep 3
    continue
  fi

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
