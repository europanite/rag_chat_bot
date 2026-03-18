#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import chromadb


def emit(key: str, value: Any) -> None:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    print(f"VERIFY {key}={value}")


def main() -> int:
    db_dir = Path(os.getenv("CHROMA_DB_DIR", "/chroma"))
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "documents")

    emit("python", sys.version.replace("\n", " "))
    emit("cwd", os.getcwd())
    emit("CHROMA_DB_DIR", str(db_dir))
    emit("CHROMA_COLLECTION_NAME", collection_name)
    emit("db_dir_exists", db_dir.is_dir())

    if db_dir.is_dir():
        files = sorted(str(p.relative_to(db_dir)) for p in db_dir.rglob("*") if p.is_file())
        emit("db_file_count", len(files))
        emit("db_files_head", files[:20])
    else:
        emit("db_file_count", 0)
        emit("db_files_head", [])

    sqlite_path = db_dir / "chroma.sqlite3"
    emit("sqlite_path", str(sqlite_path))
    emit("sqlite_exists", sqlite_path.is_file())

    if sqlite_path.is_file():
        con = sqlite3.connect(str(sqlite_path))
        try:
            cur = con.cursor()
            tables = [
                row[0]
                for row in cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            ]
            emit("sqlite_tables", tables)
            if "collections" in tables:
                rows = cur.execute(
                    "SELECT id, name FROM collections ORDER BY name"
                ).fetchall()
                emit("sqlite_collections", rows)
            if "segments" in tables:
                rows = cur.execute(
                    "SELECT id, type, scope, collection FROM segments ORDER BY id LIMIT 20"
                ).fetchall()
                emit("sqlite_segments_head", rows)
            if "embeddings" in tables:
                row = cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()
                emit("sqlite_embeddings_count", row[0] if row else None)
            elif "embedding_metadata" in tables:
                row = cur.execute("SELECT COUNT(*) FROM embedding_metadata").fetchone()
                emit("sqlite_embedding_metadata_count", row[0] if row else None)
        finally:
            con.close()

    client = chromadb.PersistentClient(path=str(db_dir))
    emit("client_type", type(client).__name__)

    try:
        collections = client.list_collections()
        names = [getattr(c, "name", str(c)) for c in collections]
        emit("client_list_collections", names)
    except Exception as exc:
        emit("client_list_collections_error", repr(exc))

    try:
        col = client.get_collection(name=collection_name)
        emit("get_collection_count", int(col.count()))
        try:
            peek = col.peek(limit=3)
            emit("get_collection_peek_ids", peek.get("ids", []))
            emit("get_collection_peek_metadatas", peek.get("metadatas", []))
        except Exception as exc:
            emit("get_collection_peek_error", repr(exc))
    except Exception as exc:
        emit("get_collection_error", repr(exc))

    try:
        col = client.get_or_create_collection(name=collection_name)
        emit("get_or_create_collection_count", int(col.count()))
    except Exception as exc:
        emit("get_or_create_collection_error", repr(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
