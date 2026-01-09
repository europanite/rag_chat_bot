import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import rag_store
from database import engine
from fastapi import FastAPI
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

@app.get("/health")
def health():
    with engine.connect() as conn:
        ok = conn.execute(text("SELECT 1")).scalar() == 1
    return {"status": "ok", "db": ok}
