#!/bin/sh
set -eu

FEED_PATH="${FEED_PATH:-frontend/app/public}"
LATEST_PATH="${LATEST_PATH:-$FEED_PATH/latest.json}"

MODEL_ID="${MODEL_ID:-stabilityai/sd-turbo}"
SD_MODEL_ID="${SD_MODEL_ID:-$MODEL_ID}"

export FEED_PATH LATEST_PATH SD_MODEL_ID

if [ ! -f "$LATEST_PATH" ]; then
  echo "ERROR: latest.json not found: $LATEST_PATH" >&2
  exit 2
fi

python scripts/illustrate.py
