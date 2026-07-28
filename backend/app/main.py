import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import rag_store
import requests
from database import engine
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from models import Base
from routers import auth, rag
from sqlalchemy import text

logger = logging.getLogger(__name__)

def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB tables
    Base.metadata.create_all(bind=engine)

    # Optional: build the vector DB from local JSON files at startup
    if _truthy(os.getenv("RAG_AUTO_INDEX", "false")):
        docs_dir = os.getenv("RAG_DOCS_DIR") or os.getenv("DOCS_DIR", "/data/json")
        rebuild = _truthy(os.getenv("RAG_REBUILD_ON_START", "true"))
        fail_fast = _truthy(os.getenv("RAG_FAIL_FAST") or os.getenv("RAG_INDEX_FAIL_FAST", "false"))


        try:
            if rebuild:
                stats = rag_store.rebuild_from_json_dir(docs_dir)
            else:
                stats = rag_store.ingest_json_dir(docs_dir)
            logger.info("RAG index ready: %s", stats)
        except Exception as exc:
            logger.exception("RAG auto-index failed: %s", exc)
            if fail_fast:
                raise

    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(rag.router)


    # static /public
    public_dir = Path(os.getenv("PUBLIC_DIR", "/public"))
    app.mount("/public", StaticFiles(directory=str(public_dir), check_dir=False), name="public")

    return app

app = create_app()


def _vllm_health_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/embeddings", "/v1"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip("/")
            break
    return f"{normalized}/health"


def _check_vllm_endpoint(base_url: str, name: str) -> bool:
    if not base_url:
        raise RuntimeError(f"{name} base URL is not configured")
    response = requests.get(_vllm_health_url(base_url), timeout=3)
    response.raise_for_status()
    return True


@app.get("/live")
def live():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    checks: dict[str, object] = {}
    try:
        with engine.connect() as conn:
            checks["db"] = conn.execute(text("SELECT 1")).scalar() == 1

        llm_provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
        if llm_provider in {"vllm", "openai-compatible", "openai_compatible"}:
            chat_base_url = (
                os.getenv("CHAT_BASE_URL")
                or os.getenv("VLLM_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or ""
            )
            checks["vllm_chat"] = _check_vllm_endpoint(chat_base_url, "chat")

        embedding_provider = (os.getenv("EMBEDDING_PROVIDER") or "ollama").strip().lower()
        if embedding_provider in {"vllm", "openai-compatible", "openai_compatible"}:
            embedding_base_url = (
                os.getenv("EMBEDDING_BASE_URL")
                or os.getenv("VLLM_EMBEDDING_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or ""
            )
            checks["vllm_embedding"] = _check_vllm_endpoint(embedding_base_url, "embedding")

        checks["chroma_collection"] = os.getenv("CHROMA_COLLECTION_NAME", "documents")
        checks["chroma_chunks"] = rag_store.get_collection_count()
    except Exception as exc:
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks, "error": str(exc)}) from exc

    return {"status": "ok", "checks": checks}


@app.get("/health")
def health():
    # Backward-compatible endpoint used by existing scripts.
    with engine.connect() as conn:
        ok = conn.execute(text("SELECT 1")).scalar() == 1
    return {"status": "ok", "db": ok}
