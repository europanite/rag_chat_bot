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

load_env_file() {
  local env_file="${1:-.env}"
  [[ -f "$env_file" ]] || return 0

  # .env may reference variables that are defined later in the same file.
  # Keep nounset disabled only while sourcing dotenv-style configuration.
  set +u
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
  set -u
}

load_env_file ".env"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
COMPOSE=(docker compose -f "$COMPOSE_FILE")
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

RUNTIME_SERVICES=(db ollama backend)

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

LOG_DIR="${FEED_LOG_DIR:-tmp/feed-debug}"
RUN_LOG="${LOG_DIR%/}/feed.log"
GENERATE_TALK_LOG="${LOG_DIR%/}/generate_talk.log"
DOCKER_DEBUG_LOG="${LOG_DIR%/}/docker_debug.log"
mkdir -p "$LOG_DIR"
: > "$RUN_LOG"
: > "$GENERATE_TALK_LOG"
: > "$DOCKER_DEBUG_LOG"

exec > >(tee -a "$RUN_LOG") 2>&1

if [[ "${GITHUB_ACTIONS:-}" == "true" && -z "${FEED_TRACE_SHELL:-}" ]]; then
  FEED_TRACE_SHELL="1"
fi
if [[ "${FEED_TRACE_SHELL:-0}" == "1" ]]; then
  export PS4='+(${BASH_SOURCE##*/}:${LINENO}): '
  set -x
fi

log() { printf "\n[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"; }

append_summary_block() {
  local title="$1"
  local file="$2"
  local lines="${3:-200}"
  [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] || return 0
  [[ -f "$file" ]] || return 0
  {
    echo "## ${title}"
    echo
    echo '```text'
    tail -n "$lines" "$file" || true
    echo '```'
    echo
  } >> "$GITHUB_STEP_SUMMARY"
}

dump_debug() {
  set +e
  {
    log "docker compose ps"
    "${COMPOSE[@]}" ps || true
    log "docker compose logs (tail=300)"
    "${COMPOSE[@]}" logs --no-color --tail=300 backend ollama db 2>/dev/null || true
    log "docker system df"
    docker system df || true
    log "df -h"
    df -h || true
  } | tee -a "$DOCKER_DEBUG_LOG"
}

on_error() {
  local rc=$?
  local line_no="${BASH_LINENO[0]:-unknown}"
  local cmd="${BASH_COMMAND:-unknown}"
  log "ERROR (exit=$rc line=$line_no cmd=$cmd). Showing debug info..."
  if [[ -s "$GENERATE_TALK_LOG" ]]; then
    log "generate_talk.py log tail (200)"
    tail -n 200 "$GENERATE_TALK_LOG" || true
  fi
  dump_debug
  append_summary_block "feed.sh tail" "$RUN_LOG" 200
  append_summary_block "generate_talk.py tail" "$GENERATE_TALK_LOG" 200
  append_summary_block "docker debug tail" "$DOCKER_DEBUG_LOG" 200
  log "Log files: $RUN_LOG | $GENERATE_TALK_LOG | $DOCKER_DEBUG_LOG"
  exit "$rc"
}
trap on_error ERR

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

debug_chroma_host() {
  log "Host chroma_db inventory"
  if [[ -d "chroma_db" ]]; then
    echo "host chroma_db file count: $(find chroma_db -type f | wc -l)"
    du -sh chroma_db || true
    test -f chroma_db/chroma.sqlite3 && ls -lh chroma_db/chroma.sqlite3 || true
    find chroma_db -maxdepth 2 -type f | sed -n '1,20p' || true
  else
    echo "host chroma_db directory is missing"
  fi
}

debug_chroma_container() {
  log "Backend-visible /chroma inventory"
  "${COMPOSE[@]}" exec -T backend sh -lc '
    echo "CHROMA_DB_DIR=${CHROMA_DB_DIR:-}";
    if [ -d /chroma ]; then
      echo "container /chroma file count: $(find /chroma -type f | wc -l)";
      du -sh /chroma || true;
      test -f /chroma/chroma.sqlite3 && ls -lh /chroma/chroma.sqlite3 || true;
      find /chroma -maxdepth 2 -type f | sed -n "1,20p" || true;
    else
      echo "/chroma directory is missing inside backend";
    fi
  ' || true
}
show_rag_status() {
  log "Backend /rag/status"
  curl -fsS --connect-timeout 2 --max-time 10 "http://localhost:${BACKEND_PORT:-8000}/rag/status" || true
  echo
}

debug_rag_visibility() {
  log "Backend direct Chroma verification"
  "${COMPOSE[@]}" exec -T backend python /scripts/verify_rag_visibility.py || true
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
  log "FEED logs -> $LOG_DIR"
  debug_chroma_host

  if [[ "$DO_BUILD" == "1" ]]; then
    log "docker compose build --pull ${RUNTIME_SERVICES[*]}"
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    export BUILDKIT_PROGRESS=plain
    "${COMPOSE[@]}" build --pull "${RUNTIME_SERVICES[@]}"
  fi

  log "docker compose up -d ${RUNTIME_SERVICES[*]}"
  "${COMPOSE[@]}" up -d "${RUNTIME_SERVICES[@]}"
  "${COMPOSE[@]}" ps

  if ! wait_backend_health; then
    log "Backend did not become ready in time"
    dump_debug
    exit 1
  fi

  debug_chroma_container
  show_rag_status
  debug_rag_visibility

  fix_permissions

  log "Run: ${PYTHON_BIN} scripts/generate_talk.py"
  set +e
  "${PYTHON_BIN}" scripts/generate_talk.py 2>&1 | tee "$GENERATE_TALK_LOG"
  local py_rc=${PIPESTATUS[0]}
  set -e
  if [[ "$py_rc" -ne 0 ]]; then
    log "generate_talk.py failed (exit=$py_rc)"
    return "$py_rc"
  fi

  if [[ "$DO_DOWN" == "1" ]]; then
    log "docker compose down"
    "${COMPOSE[@]}" down
  else
    log "Skip docker compose down (--no-down)"
  fi

  append_summary_block "feed.sh tail" "$RUN_LOG" 120
  append_summary_block "generate_talk.py tail" "$GENERATE_TALK_LOG" 120
  log "FEED done"
}

main "$@"