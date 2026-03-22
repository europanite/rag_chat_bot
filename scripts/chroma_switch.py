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

FETCH_PUBLIC_FILE = Path(__file__).resolve().with_name("fetch_public_file.py")


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


def replace_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def restore_local(local_dir: Path, out_dir: Path) -> None:
    if not local_dir.exists():
        raise SystemExit(f"ERROR: local chroma seed dir not found: {local_dir}")
    sqlite_path = local_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise SystemExit(f"ERROR: local chroma seed dir does not contain chroma.sqlite3: {local_dir}")
    replace_tree(local_dir, out_dir)
    print(json.dumps({"source": "local", "out_dir": str(out_dir)}, ensure_ascii=False))


def restore_gdrive(url: str, out_dir: Path, min_bytes: int) -> None:
    if not url:
        raise SystemExit("ERROR: --gdrive-url is required when --source=gdrive")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        archive_path = td_path / "chroma-db.tgz"
        sh(
            sys.executable,
            str(FETCH_PUBLIC_FILE),
            "--url",
            url,
            "--out",
            str(archive_path),
            "--min-bytes",
            str(min_bytes),
        )
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(td_path / "extract")
        extracted = td_path / "extract" / "chroma_db"
        if not extracted.exists():
            raise SystemExit("ERROR: downloaded archive does not contain chroma_db/")
        replace_tree(extracted, out_dir)
    print(json.dumps({"source": "gdrive", "out_dir": str(out_dir)}, ensure_ascii=False))


def upload_latest_s3(bucket: str, prefix: str, chroma_dir: Path) -> dict[str, str]:
    if not bucket:
        raise SystemExit("ERROR: --s3-bucket is required when --source=s3")
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
    result = {"source": "s3", "bucket": bucket, "key": key, "latest": latest_key}
    print(json.dumps(result, ensure_ascii=False))
    return result


def restore_s3(bucket: str, prefix: str, out_dir: Path) -> None:
    if not bucket:
        raise SystemExit("ERROR: --s3-bucket is required when --source=s3")
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
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(td_path / "extract")
        extracted = td_path / "extract" / "chroma_db"
        if not extracted.exists():
            raise SystemExit("ERROR: downloaded archive does not contain chroma_db/")
        replace_tree(extracted, out_dir)
    print(json.dumps({"source": "s3", "bucket": bucket, "latest": latest_key, "out_dir": str(out_dir)}, ensure_ascii=False))


def restore(source: str, out_dir: Path, gdrive_url: str, s3_bucket: str, s3_prefix: str, local_dir: Path, min_bytes: int) -> None:
    if source == "local":
        restore_local(local_dir, out_dir)
        return
    if source == "gdrive":
        restore_gdrive(gdrive_url, out_dir, min_bytes)
        return
    if source == "s3":
        restore_s3(s3_bucket, s3_prefix, out_dir)
        return
    raise SystemExit(f"ERROR: unsupported source: {source}")


def publish(source: str, chroma_dir: Path, s3_bucket: str, s3_prefix: str, artifact_out: Path | None) -> None:
    archive = make_archive(chroma_dir)
    if artifact_out is not None:
        artifact_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, artifact_out)

    if source == "s3":
        upload_latest_s3(s3_bucket, s3_prefix, chroma_dir)
        return

    result = {
        "source": source,
        "mode": "manual_handoff",
        "archive": str(artifact_out or archive),
    }
    print(json.dumps(result, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify-local")
    v.add_argument("--chroma-dir", required=True)
    v.add_argument("--collection-name", default="documents")
    v.add_argument("--min-count", type=int, default=1)

    r = sub.add_parser("restore")
    r.add_argument("--source", choices=["local", "gdrive", "s3"], default="gdrive")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--gdrive-url", default="")
    r.add_argument("--s3-bucket", default="")
    r.add_argument("--s3-prefix", default="rag/chroma")
    r.add_argument("--local-dir", default="chroma_db_seed")
    r.add_argument("--min-bytes", type=int, default=1024)

    pub = sub.add_parser("publish")
    pub.add_argument("--source", choices=["local", "gdrive", "s3"], default="gdrive")
    pub.add_argument("--chroma-dir", required=True)
    pub.add_argument("--s3-bucket", default="")
    pub.add_argument("--s3-prefix", default="rag/chroma")
    pub.add_argument("--artifact-out", default="")

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "verify-local":
        verify_local(Path(args.chroma_dir), args.collection_name, args.min_count)
        return 0
    if args.cmd == "restore":
        restore(
            source=args.source,
            out_dir=Path(args.out_dir),
            gdrive_url=args.gdrive_url,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            local_dir=Path(args.local_dir),
            min_bytes=args.min_bytes,
        )
        return 0
    if args.cmd == "publish":
        publish(
            source=args.source,
            chroma_dir=Path(args.chroma_dir),
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            artifact_out=Path(args.artifact_out) if args.artifact_out else None,
        )
        return 0
    raise SystemExit(f"ERROR: unknown command: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
