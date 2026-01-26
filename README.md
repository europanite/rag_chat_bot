# [RAG Chat Bot](https://github.com/europanite/rag_chat_bot "RAG Chat Bot")

Under Development

---

## 🚀 Getting Started

### 1. Prerequisites
- [Docker Compose](https://docs.docker.com/compose/)
- [Expo Go](https://expo.dev/go)

### 2. Build and start all services:

```bash
# set environment variables:
export REACT_NATIVE_PACKAGER_HOSTNAME=${YOUR_HOST}

# Build the image
docker compose build

# Run the container
docker compose up
```
---

### 3. Test:

```bash
# Backend pytest
docker compose \
  -f docker-compose.test.yml run \
  --rm \
  --entrypoint /bin/sh backend_test \
  -lc 'pytest -q'

# Backend Lint
docker compose \
  -f docker-compose.test.yml run \
  --rm \
  --entrypoint /bin/sh backend_test \
  -lc 'ruff check /app /tests'

# Frontend Test
docker compose \
  -f docker-compose.test.yml run \
  --rm frontend_test
```

```bash
# Update info
python3 local/md_dir_to_json.py --recursive data/md/ data/json/data.json
```

```bash
# Ingest
docker compose up -d --build db ollama backend
until curl -fsS http://localhost:8000/rag/status >/dev/null; do sleep 2; done
curl -sS --fail-with-body -X POST http://localhost:8000/rag/reindex
docker compose down
```

```bash
# Feed
docker compose -f docker-compose.yml build --pull
docker compose -f docker-compose.yml up -d
for i in {1..600}; do
  if curl -fsS --connect-timeout 1 --max-time 2 http://localhost:8000/health >/dev/null 2>&1; then
    echo "backend ok"; break
  fi
  sleep 2
done
python scripts/generate_talk.py
docker compose -f docker-compose.yml down
```

---

# License
- No License Granted for Now.