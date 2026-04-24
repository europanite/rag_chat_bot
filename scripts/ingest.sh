#!/usr/bin/env bash
# scripts/ingest.sh
#
#   bash scripts/ingest.sh feed
#   bash scripts/ingest.sh ingest
#   bash scripts/ingest.sh feed --ci   # GitHub Actions build 
#   bash scripts/ingest.sh ingest --ci

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
COMPOSE=(docker compose -f "$COMPOSE_FILE")

PYTHON_BIN="${PYTHON_BIN:-python}"

MODE="auto"          # auto | local | ci
DO_BUILD="auto"      # auto | 0 | 1
DO_DOWN="1"          # 0 | 1
WITH_INDEX="0"       # 0 | 1 
SERVICES=""          # ingest up workflow

SUBCMD="${1:-}"
shift || true

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/ingest.sh <feed|ingest> [options]

Options:
  --ci              CI
  --local           local
  --build           docker compose build
  --no-build        build skip
  --no-down         no docker compose down 
  --with-index      RAG_AUTO_INDEX / RAG_REBUILD_ON_START  true
  --services "..."  ingest up

Environment (optional):
  COMPOSE_FILE, PYTHON_BIN
USAGE
}

if [[ -z "$SUBCMD" ]]; then
  usage; exit 2
fi

# ---- option parse ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ci) MODE="ci";;
    --local) MODE="local";;
    --build) DO_BUILD="1";;
    --no-build) DO_BUILD="0";;
    --no-down) DO_DOWN="0";;
    --with-index) WITH_INDEX="1";;
    --services)
      shift
      SERVICES="${1:-}"
      ;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2
      usage; exit 2;;
  esac
  shift || true
done

if [[ "$MODE" == "auto" ]]; then
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    MODE="ci"
  else
    MODE="local"
  fi
fi

if [[ "$DO_BUILD" == "auto" ]]; then
  if [[ "$MODE" == "ci" ]]; then
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

# ---- load .env (for local + for python script env) ----
if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  set -a; source ".env"; set +a
fi

# ---- shared defaults (workflow) ----
export REACT_NATIVE_PACKAGER_HOSTNAME="${REACT_NATIVE_PACKAGER_HOSTNAME:-127.0.0.1}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL}"
export OLLAMA_TIMEOUT_S="${OLLAMA_TIMEOUT_S}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL}"
export RAG_MODEL="${RAG_MODEL}"
export AUDIT_MODEL="${AUDIT_MODEL}"

# feed workflow audit
export RAG_AUDIT="${RAG_AUDIT}"
export RAG_AUDIT_REWRITE="${RAG_AUDIT_REWRITE}"
export RAG_AUDIT_MAX_ATTEMPTS="${RAG_AUDIT_MAX_ATTEMPTS}"

# feed index/rebuild（workflow）
if [[ "$WITH_INDEX" != "1" ]]; then
  export RAG_AUTO_INDEX="false"
  export RAG_REBUILD_ON_START="false"
fi

# ---- subcommand impl ----
wait_backend_health() {
  log "Waiting for backend health..."
  local ok=0
  for i in $(seq 1 600); do
    if curl -fsS --connect-timeout 1 --max-time 2 "http://localhost:${BACKEND_PORT:-8000}/health" >/dev/null 2>&1; then
      ok=$((ok+1))
      [[ "$ok" -ge 3 ]] && break
    else
      ok=0
    fi
    sleep 2
  done
  if [[ "$ok" -ge 3 ]]; then
    log "backend is stable"
  else
    log "Backend did not become ready in time"
    "${COMPOSE[@]}" logs --no-color --tail=400 backend ollama db || true
    return 1
  fi
}

fix_feed_permissions() {
  # GitHub runner
  log "Fix permissions for feed dirs (best effort)"
  if command -v sudo >/dev/null 2>&1; then
    sudo chown -R "${USER:-runner}:${USER:-runner}" "frontend/app/public" 2>/dev/null || true
  fi
  chmod -R u+rwX "frontend/app/public" 2>/dev/null || true
}

run_feed() {
  log "Running FEED pipeline (mode=$MODE build=$DO_BUILD down=$DO_DOWN)"
  # workflow Generate post env
  export LAT="${LAT:-35.2810}"
  export LON="${LON:-139.6722}"
  export TZ_NAME="${TZ_NAME:-Asia/Tokyo}"
  export RAG_TOP_K="${RAG_TOP_K:-16}"
  export CURL_MAX_TIME="${CURL_MAX_TIME:-512}"
  export CURL_RETRIES="${CURL_RETRIES:-0}"
  export PLACE="${PLACE:-Yokosuka}"
  export FEED_PATH="${FEED_PATH:-frontend/app/public}"
  export LATEST_PATH="${LATEST_PATH:-frontend/app/public/latest.json}"
  export MAX_CHARS="${MAX_CHARS:-256}"
  export RAG_VARIETY="${RAG_VARIETY:-1.0}"
  export DEBUG="${DEBUG:-1}"
  if [[ -z "${TOPIC_VARIANT:-}" ]]; then
    TOPIC_VARIANT="$(date +%s)"
    export TOPIC_VARIANT
  fi

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

  wait_backend_health
  fix_feed_permissions

  log "python scripts/generate_talk.py"
  "$PYTHON_BIN" scripts/generate_talk.py

  if [[ "$DO_DOWN" == "1" ]]; then
    log "docker compose down"
    "${COMPOSE[@]}" down
  else
    log "Skip docker compose down (--no-down)"
  fi
}

wait_backend_status() {
  log "Waiting for backend /rag/status..."
  for i in $(seq 1 120); do
    if curl -fsS --connect-timeout 2 --max-time 10 "http://localhost:${BACKEND_PORT:-8000}/rag/status" >/dev/null 2>&1; then
      log "backend is up"
      return 0
    fi
    sleep 2
  done
  log "Backend did not become ready in time"
  "${COMPOSE[@]}" logs --no-color --tail=400 backend ollama db || true
  return 1
}

run_ingest() {
  log "Running INGEST pipeline (mode=$MODE build=$DO_BUILD down=$DO_DOWN)"
  export RAG_REINDEX_ENABLED="${RAG_REINDEX_ENABLED:-true}"
  export RAG_AUTO_INDEX="${RAG_AUTO_INDEX:-false}"
  export RAG_REBUILD_ON_START="${RAG_REBUILD_ON_START:-false}"
  export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
  export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ollama}"
  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
  export OLLAMA_TIMEOUT_S="${OLLAMA_TIMEOUT_S:-512}"
  export EMBEDDING_MODEL="${EMBEDDING_MODEL:-mxbai-embed-large}"
  export RAG_MODEL="${RAG_MODEL:-qwen3:8b}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
  export OPENAI_CHAT_MODEL="${OPENAI_CHAT_MODEL:-gpt-4.1-mini}"
  export OPENAI_EMBEDDING_MODEL="${OPENAI_EMBEDDING_MODEL:-text-embedding-3-large}"

  local up_services
  if [[ -n "$SERVICES" ]]; then
    up_services="$SERVICES"
  else
    up_services="db ollama backend"
  fi

  if [[ "$DO_BUILD" == "1" ]]; then
    log "docker compose build --pull ${up_services}"
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    export BUILDKIT_PROGRESS=plain
    "${COMPOSE[@]}" build --pull ${up_services}
  fi

  log "docker compose up -d ${up_services}"
  "${COMPOSE[@]}" up -d ${up_services}
  "${COMPOSE[@]}" ps

  wait_backend_status

  log "POST /rag/reindex (with retries)"
  local rc=0
  local attempt
  for attempt in $(seq 1 10); do
    set +e
    curl -sS --fail-with-body \
      -X POST "http://localhost:${BACKEND_PORT:-8000}/rag/reindex" \
      -H "Content-Type: application/json" \
      -d '{"force": true, "reset": true}' \
      -o /tmp/reindex_body.json \
      -D /tmp/reindex_headers.txt
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
      log "Reindex succeeded (attempt ${attempt})"
      break
    fi
    log "Reindex failed (rc=$rc) attempt ${attempt}/10; retrying..."
    sleep 10
  done

  if [[ "$rc" -ne 0 ]]; then
    log "ERROR: Reindex failed after retries"
    log "Headers:"
    sed -n '1,200p' /tmp/reindex_headers.txt || true
    log "Body:"
    sed -n '1,200p' /tmp/reindex_body.json || true
    return 1
  fi

  log "Verify chroma_db seems written"
  local n
  n="$(find chroma_db -type f 2>/dev/null | wc -l | tr -d ' ')"
  log "chroma_db file count: ${n}"
  if [[ "${n}" -lt 5 ]]; then
    log "ERROR: chroma_db seems empty after ingest"
    return 1
  fi

  if [[ "$DO_DOWN" == "1" ]]; then
    log "docker compose down"
    "${COMPOSE[@]}" down
  else
    log "Skip docker compose down (--no-down)"
  fi
}

case "$SUBCMD" in
  feed)   run_feed;;
  ingest) run_ingest;;
  *) echo "Unknown subcommand: $SUBCMD" >&2; usage; exit 2;;
esac
