#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def inspect_docs(docs_dir: Path, sample_limit: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": "docs",
        "docs_dir": str(docs_dir),
        "exists": docs_dir.exists(),
        "is_dir": docs_dir.is_dir(),
    }
    if not docs_dir.exists() or not docs_dir.is_dir():
        payload["file_count"] = 0
        payload["samples"] = []
        return payload

    files = sorted(p for p in docs_dir.rglob("*") if p.is_file())
    payload["file_count"] = len(files)
    payload["total_bytes"] = sum(p.stat().st_size for p in files)
    samples: list[dict[str, Any]] = []
    for path in files[:sample_limit]:
        item: dict[str, Any] = {
            "path": str(path),
            "size": path.stat().st_size,
        }
        try:
            raw = path.read_text(encoding="utf-8")
            item["utf8"] = True
            item["preview"] = raw[:200]
            try:
                data = json.loads(raw)
                item["json_valid"] = True
                item["json_type"] = type(data).__name__
                if isinstance(data, dict):
                    item["top_keys"] = list(data.keys())[:15]
                elif isinstance(data, list):
                    item["list_length"] = len(data)
            except Exception as exc:
                item["json_valid"] = False
                item["json_error"] = str(exc)
        except Exception as exc:
            item["utf8"] = False
            item["read_error"] = str(exc)
        samples.append(item)
    payload["samples"] = samples
    return payload


def table_columns(cur: sqlite3.Cursor, table: str) -> list[str]:
    rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(row[1]) for row in rows]


def inspect_chroma(chroma_dir: Path, collection_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": "chroma",
        "chroma_dir": str(chroma_dir),
        "exists": chroma_dir.exists(),
        "is_dir": chroma_dir.is_dir(),
        "file_count": len([p for p in chroma_dir.rglob('*') if p.is_file()]) if chroma_dir.exists() else 0,
    }
    sqlite_path = chroma_dir / "chroma.sqlite3"
    payload["sqlite_path"] = str(sqlite_path)
    payload["sqlite_exists"] = sqlite_path.exists()
    if not sqlite_path.exists():
        return payload

    con = sqlite3.connect(sqlite_path)
    try:
        cur = con.cursor()
        tables = [row[0] for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        payload["tables"] = tables
        payload["collections_columns"] = table_columns(cur, "collections") if "collections" in tables else []
        payload["segments_columns"] = table_columns(cur, "segments") if "segments" in tables else []
        payload["embeddings_columns"] = table_columns(cur, "embeddings") if "embeddings" in tables else []

        collection_rows = []
        collection_id = None
        if "collections" in tables:
            collection_rows = cur.execute(
                "SELECT * FROM collections ORDER BY name"
            ).fetchall()
            payload["collections"] = [list(row) for row in collection_rows]
            if collection_rows:
                cols = payload["collections_columns"]
                try:
                    idx_name = cols.index("name")
                    idx_id = cols.index("id")
                    for row in collection_rows:
                        if row[idx_name] == collection_name:
                            collection_id = row[idx_id]
                            break
                except ValueError:
                    pass
        payload["verify_collection"] = collection_name
        payload["collection_id"] = collection_id

        if "segments" in tables:
            segment_rows = cur.execute("SELECT * FROM segments ORDER BY scope, id").fetchall()
            payload["segments"] = [list(row) for row in segment_rows]
        else:
            segment_rows = []
            payload["segments"] = []

        total_embeddings = None
        if "embeddings" in tables:
            total_embeddings = cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        payload["sqlite_total_embeddings"] = total_embeddings

        collection_embeddings = None
        if collection_id is not None and "embeddings" in tables:
            emb_cols = payload["embeddings_columns"]
            seg_cols = payload["segments_columns"]
            if "collection_id" in emb_cols:
                collection_embeddings = cur.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE collection_id = ?",
                    (collection_id,),
                ).fetchone()[0]
            elif "segment_id" in emb_cols and "id" in seg_cols and "collection" in seg_cols:
                collection_embeddings = cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM embeddings e
                    JOIN segments s ON s.id = e.segment_id
                    WHERE s.collection = ?
                    """,
                    (collection_id,),
                ).fetchone()[0]
        payload["sqlite_collection_embeddings"] = collection_embeddings
    finally:
        con.close()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    docs = sub.add_parser("docs")
    docs.add_argument("--docs-dir", required=True)
    docs.add_argument("--sample-limit", type=int, default=10)

    chroma = sub.add_parser("chroma")
    chroma.add_argument("--chroma-dir", required=True)
    chroma.add_argument("--collection-name", default="documents")

    args = parser.parse_args()
    if args.cmd == "docs":
        print_json(inspect_docs(Path(args.docs_dir), args.sample_limit))
        return 0
    if args.cmd == "chroma":
        print_json(inspect_chroma(Path(args.chroma_dir), args.collection_name))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
