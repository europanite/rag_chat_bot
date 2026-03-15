#!/bin/sh
set -eu

FEED_PATH="${FEED_PATH:-frontend/app/public}"
LATEST_PATH="${LATEST_PATH:-$FEED_PATH/latest.json}"

MODEL_ID="${MODEL_ID:-stabilityai/sdxl-turbo}"
SDXL_MODEL_ID="${SDXL_MODEL_ID:-$MODEL_ID}"

export FEED_PATH LATEST_PATH MODEL_ID SDXL_MODEL_ID LORA_PATH LORA_SCALE

if [ ! -f "$LATEST_PATH" ]; then
  echo "ERROR: latest.json not found: $LATEST_PATH" >&2
  exit 2
fi

 echo "[illustrate] MODEL_ID=${MODEL_ID}"
 echo "[illustrate] SDXL_MODEL_ID=${SDXL_MODEL_ID}"

if [ -n "${INPUT_IMAGE:-}" ]; then
  echo "[illustrate] INPUT_IMAGE provided -> running scripts/arrange.py (img2img/pillow)"
  python scripts/arrange.py
else
  echo "[illustrate] No INPUT_IMAGE -> running scripts/illustrate.py (text2img)"
  python scripts/illustrate.py
fi