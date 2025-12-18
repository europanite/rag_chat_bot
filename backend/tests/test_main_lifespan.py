from __future__ import annotations

import asyncio

import main
import pytest


def test_truthy_variants() -> None:
    assert main._truthy("1")
    assert main._truthy("true")
    assert main._truthy("YES")
    assert not main._truthy("0")
    assert not main._truthy("")
    assert not main._truthy(None)


def test_lifespan_auto_index_ingest(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    called = {"create_all": 0, "ingest": None, "rebuild": None}

    monkeypatch.setattr(
        main.Base.metadata,
        "create_all",
        lambda bind=None: called.__setitem__("create_all", called["create_all"] + 1))
    monkeypatch.setattr(
        main.rag_store, "ingest_json_dir",
        lambda d: called.__setitem__("ingest", d))
    monkeypatch.setattr(
        main.rag_store, "rebuild_from_json_dir",
        lambda d: called.__setitem__("rebuild", d))
    monkeypatch.setenv("RAG_AUTO_INDEX", "true")
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_REBUILD_ON_START", "false")
    monkeypatch.setenv("RAG_INDEX_FAIL_FAST", "false")

    async def run():
        async with main.lifespan(main.app):
            pass

    asyncio.run(run())

    assert called["create_all"] == 1
    assert called["ingest"] == str(tmp_path)
    assert called["rebuild"] is None


def test_lifespan_auto_index_rebuild(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    called = {"rebuild": None}
    monkeypatch.setattr(main.Base.metadata, "create_all", lambda bind=None: None)
    monkeypatch.setattr(
        main.rag_store, "rebuild_from_json_dir",
        lambda d: called.__setitem__("rebuild", d))
    monkeypatch.setattr(main.rag_store, "ingest_json_dir", lambda d: None)
    monkeypatch.setenv("RAG_AUTO_INDEX", "true")
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_REBUILD_ON_START", "true")
    monkeypatch.setenv("RAG_INDEX_FAIL_FAST", "false")

    async def run():
        async with main.lifespan(main.app):
            pass

    asyncio.run(run())
    assert called["rebuild"] == str(tmp_path)


def test_lifespan_fail_fast_re_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(main.Base.metadata, "create_all", lambda bind=None: None)

    def boom(_d: str):
        raise RuntimeError("ingest failed")

    monkeypatch.setattr(main.rag_store, "ingest_json_dir", boom)
    monkeypatch.setattr(main.rag_store, "rebuild_from_json_dir", lambda d: None)

    monkeypatch.setenv("RAG_AUTO_INDEX", "true")
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_REBUILD_ON_START", "false")
    monkeypatch.setenv("RAG_INDEX_FAIL_FAST", "true")

    async def run():
        async with main.lifespan(main.app):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(run())
