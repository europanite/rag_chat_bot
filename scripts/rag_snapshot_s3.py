#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sh(*args: str, capture: bool = False) -> str:
    proc = subprocess.run(args, check=True, text=True, capture_output=capture)
    return proc.stdout if capture else ""


def verify_local(chroma_dir: Path, collection_name: str, min_count: int) -> int:
    sqlite_path = chroma_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise SystemExit(f"ERROR: sqlite file not found: {sqlite_path}")

    with sqlite3.connect(sqlite_path) as con:
        cur = con.cursor()
        row = cur.execute(
            "SELECT id, name FROM collections WHERE name = ?",
            (collection_name,),
        ).fetchone()
        if row is None:
            names = [r[0] for r in cur.execute("SELECT name FROM collections ORDER BY name")]
            raise SystemExit(
                f"ERROR: collection '{collection_name}' not found. collections={json.dumps(names)}"
            )
        collection_id, _ = row

        total_embeddings = cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        segment_rows = cur.execute(
            "SELECT id, scope, type, collection FROM segments WHERE collection = ? ORDER BY id",
            (collection_id,),
        ).fetchall()
        segment_ids = [r[0] for r in segment_rows]

        if segment_ids:
            placeholders = ",".join("?" for _ in segment_ids)
            count = cur.execute(
                f"SELECT COUNT(*) FROM embeddings WHERE segment_id IN ({placeholders})",
                segment_ids,
            ).fetchone()[0]
        else:
            count = 0

        print(
            json.dumps(
                {
                    "verify_collection": collection_name,
                    "collection_id": collection_id,
                    "segments": segment_rows,
                    "sqlite_total_embeddings": total_embeddings,
                    "sqlite_collection_embeddings": count,
                },
                ensure_ascii=False,
            )
        )
        if count < min_count:
            raise SystemExit(
                f"ERROR: collection '{collection_name}' has only {count} embeddings; expected >= {min_count}"
            )
        return int(count)


def make_archive(chroma_dir: Path) -> Path:
    if not chroma_dir.exists():
        raise SystemExit(f"ERROR: chroma dir not found: {chroma_dir}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(tempfile.gettempdir()) / f"chroma-{stamp}.tgz"
    if out.exists():
        out.unlink()
    with tarfile.open(out, "w:gz") as tf:
        tf.add(chroma_dir, arcname="chroma_db")
    return out


def upload_latest(bucket: str, prefix: str, chroma_dir: Path) -> None:
    archive = make_archive(chroma_dir)
    key = f"{prefix.rstrip('/')}/{archive.name}"
    latest_key = f"{prefix.rstrip('/')}/latest.txt"
    sh("aws", "s3", "cp", str(archive), f"s3://{bucket}/{key}")
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as fh:
        fh.write(key + "\n")
        latest_file = fh.name
    try:
        sh("aws", "s3", "cp", latest_file, f"s3://{bucket}/{latest_key}")
    finally:
        Path(latest_file).unlink(missing_ok=True)
    print(json.dumps({"bucket": bucket, "key": key, "latest": latest_key}, ensure_ascii=False))


def download_latest(bucket: str, prefix: str, out_dir: Path) -> None:
    latest_key = f"{prefix.rstrip('/')}/latest.txt"
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        latest_path = td_path / "latest.txt"
        sh("aws", "s3", "cp", f"s3://{bucket}/{latest_key}", str(latest_path))
        key = latest_path.read_text(encoding="utf-8").strip()
        if not key:
            raise SystemExit(f"ERROR: empty latest pointer at s3://{bucket}/{latest_key}")
        archive_path = td_path / Path(key).name
        sh("aws", "s3", "cp", f"s3://{bucket}/{key}", str(archive_path))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(td_path / "extract")
        extracted = td_path / "extract" / "chroma_db"
        if not extracted.exists():
            raise SystemExit("ERROR: downloaded archive does not contain chroma_db/")
        shutil.copytree(extracted, out_dir, dirs_exist_ok=True)
    print(json.dumps({"bucket": bucket, "latest": latest_key, "out_dir": str(out_dir)}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify-local")
    v.add_argument("--chroma-dir", required=True)
    v.add_argument("--collection-name", default="documents")
    v.add_argument("--min-count", type=int, default=1)

    u = sub.add_parser("upload-latest")
    u.add_argument("--bucket", required=True)
    u.add_argument("--prefix", required=True)
    u.add_argument("--chroma-dir", required=True)

    d = sub.add_parser("download-latest")
    d.add_argument("--bucket", required=True)
    d.add_argument("--prefix", required=True)
    d.add_argument("--out-dir", required=True)

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "verify-local":
        verify_local(Path(args.chroma_dir), args.collection_name, args.min_count)
        return 0
    if args.cmd == "upload-latest":
        upload_latest(args.bucket, args.prefix, Path(args.chroma_dir))
        return 0
    if args.cmd == "download-latest":
        download_latest(args.bucket, args.prefix, Path(args.out_dir))
        return 0
    raise SystemExit(f"ERROR: unknown command: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
