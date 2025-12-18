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

RES="$(curl -sS "${API_BASE}/rag/query" -H "Content-Type: application/json" -d "${JSON_PAYLOAD}")"

TWEET="$(python - <<'PY'
import json, sys
obj = json.loads(sys.stdin.read())
ans = (obj.get("answer") or "").strip()
# some models wrap quotes; strip one layer
if (ans.startswith('"') and ans.endswith('"')) or (ans.startswith("'") and ans.endswith("'")):
  ans = ans[1:-1].strip()
print(ans)
PY
<<<"${RES}")"

# Update JSON feed for the frontend
python scripts/update_weather_feed.py \
  --feed "${FEED_PATH}" \
  --latest "${LATEST_PATH}" \
  --date "${TODAY}" \
  --text "${TWEET}" \
  --place "${WEATHER_PLACE:-}"

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
