#!/usr/bin/env bash
# scripts/feed.sh
#
# 使い方:
#   bash scripts/feed.sh
#   bash scripts/feed.sh --ci          # CI
#   bash scripts/feed.sh --build       # build
#   bash scripts/feed.sh --no-build    # bno uild
#   bash scripts/feed.sh --no-down     # no down 
#
#   COMPOSE_FILE= docker-compose.yml
#   PYTHON_BIN=python
#   BACKEND_PORT=8000

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
COMPOSE=(docker compose -f "$COMPOSE_FILE")
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

MODE="local"        # local|ci
DO_BUILD="auto"     # auto|0|1
DO_DOWN="1"         # 0|1

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/feed.sh [options]

Options:
  --ci        CI
  --build     build
  --no-build  no build
  --no-down   no docker compose down
  -h,--help   help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ci) MODE="ci" ;;
    --build) DO_BUILD="1" ;;
    --no-build) DO_BUILD="0" ;;
    --no-down) DO_DOWN="0" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if [[ "$DO_BUILD" == "auto" ]]; then
  if [[ "$MODE" == "ci" || "${GITHUB_ACTIONS:-}" == "true" ]]; then
    DO_BUILD="1"
  else
    DO_BUILD="0"
  fi
fi

log() { printf "\n[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"; }

dump_debug() {
  set +e
  log "docker compose ps"
  "${COMPOSE[@]}" ps || true
  log "docker compose logs (tail=300)"
  "${COMPOSE[@]}" logs --no-color --tail=300 backend ollama db 2>/dev/null || true
  log "docker system df"
  docker system df || true
  log "df -h"
  df -h || true
}

on_error() {
  local rc=$?
  log "ERROR (exit=$rc). Showing debug info..."
  dump_debug
  exit "$rc"
}
trap on_error ERR

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  set -a; source ".env"; set +a
fi

wait_backend_health() {
  log "Waiting for backend /health..."
  local ok=0
  for i in $(seq 1 600); do
    if curl -fsS --connect-timeout 1 --max-time 2 "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
      ok=$((ok+1))
      [[ "$ok" -ge 3 ]] && break
    else
      ok=0
    fi
    sleep 2
  done
  [[ "$ok" -ge 3 ]]
}

fix_permissions() {
  log "Fix permissions for frontend/app/public (best effort)"
  if command -v sudo >/dev/null 2>&1; then
    sudo chown -R "${USER:-runner}:${USER:-runner}" "frontend/app/public" 2>/dev/null || true
  fi
  chmod -R u+rwX "frontend/app/public" 2>/dev/null || true
}

main() {
  log "FEED start (mode=$MODE build=$DO_BUILD down=$DO_DOWN)"
  if [[ "$DO_BUILD" == "1" ]]; then
    log "docker compose build --pull"
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    export BUILDKIT_PROGRESS=plain
    "${COMPOSE[@]}" build --pull
  fi

  log "docker compose up -d"
  "${COMPOSE[@]}" up -d
  "${COMPOSE[@]}" ps

  if ! wait_backend_health; then
    log "Backend did not become ready in time"
    dump_debug
    exit 1
  fi

  fix_permissions

  log "Run: ${PYTHON_BIN} scripts/generate_talk.py"
  "${PYTHON_BIN}" scripts/generate_talk.py

  if [[ "$DO_DOWN" == "1" ]]; then
    log "docker compose down"
    "${COMPOSE[@]}" down
  else
    log "Skip docker compose down (--no-down)"
  fi

  log "FEED done"
}

main "$@"