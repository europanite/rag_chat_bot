#!/usr/bin/env bash
# scripts/feed.sh
#
#   bash scripts/feed.sh
#   bash scripts/feed.sh --ci

set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT_DIR}/scripts/pipeline.sh" feed "$@"
