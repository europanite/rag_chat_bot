from __future__ import annotations

import json
from pathlib import Path

import pytest
import rag_store


def test_list_json_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "b.json").write_text("[]", encoding="utf-8")
    (tmp_path / "a.json").write_text("[]", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    got = rag_store.list_json_files(str(tmp_path))
    assert got == [str(tmp_path / "a.json"), str(tmp_path / "b.json")]


def test_escape_control_chars_makes_invalid_json_parseable() -> None:
    # JSON
    raw = '{"id":"1","text":"hello\nworld","metadata":{}}'
    fixed = rag_store._escape_control_chars_inside_json_strings(raw)
    data = json.loads(fixed)
    assert data["text"] == "hello\nworld"


def test_load_json_file_repairs_and_normalizes(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    # text -> _load_json_file
    p.write_text('{"id":"doc1","text":"hello\nworld","metadata":{"tags":["miura"]}}',
                 encoding="utf-8")

    docs = rag_store._load_json_file(str(p))
    assert len(docs) == 1
    d = docs[0]
    assert d["id"] == "doc1"
    assert d["text"] == "hello\nworld"
    assert d["file"] == "x.json"
    assert d["metadata"]["tags"] == ["miura"]
    assert d["source"].startswith("file://")


def test_ingest_json_dir_counts_documents_chunks_files(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(
        json.dumps([{"id": "a1", "text": "aaa", "metadata": {"tags": ["miura"]}}],
                   ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text(
        json.dumps(
            [
                {"id": "b1", "text": "bbb"},
                {"id": "b2", "text": "ccc", "metadata": {"k": 1}},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_upsert(doc_id: str, text: str, *, source=None, metadata=None, max_tokens=None) -> int:
        calls.append((doc_id, text, source, metadata))
        return 2  # 1 doc 2 chunks

    monkeypatch.setattr(rag_store, "upsert_document", fake_upsert)

    stats = rag_store.ingest_json_dir(str(tmp_path))
    assert stats == {"documents": 3, "chunks": 6, "files": 2}

    # metadata  meta.setdefault("file", ...)
    assert any((m or {}).get("file") == "a.json" for _, _, _, m in calls)
    assert any((m or {}).get("file") == "b.json" for _, _, _, m in calls)


def test_rebuild_from_json_dir_calls_reset_then_ingest(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path) -> None:
    events = []

    monkeypatch.setattr(rag_store, "reset_collection", lambda: events.append("reset"))
    monkeypatch.setattr(
        rag_store,
        "ingest_json_dir",
        lambda d: (events.append(f"ingest:{d}") or {"documents": 0, "chunks": 0, "files": 0}),
    )

    rag_store.rebuild_from_json_dir(str(tmp_path))
    assert events[0] == "reset"
    assert events[1] == f"ingest:{tmp_path!s}"
