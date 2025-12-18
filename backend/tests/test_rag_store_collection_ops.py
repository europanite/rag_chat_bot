from __future__ import annotations

from typing import Any

import pytest
import rag_store


class DummyCollectionUpsert:
    def __init__(self) -> None:
        self.deleted: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self._count = 0

    def count(self) -> int:
        return self._count

    def delete(self, **kwargs: Any) -> None:
        self.deleted.append(kwargs)

    def add(self, **kwargs: Any) -> None:
        self.add_calls.append(kwargs)
        self._count += len(kwargs.get("documents") or [])

    def upsert(self, **kwargs: Any) -> None:
        self.upsert_calls.append(kwargs)
        self._count = max(self._count, len(kwargs.get("documents") or []))


class DummyCollectionNoUpsert:
    def __init__(self) -> None:
        self.deleted: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []
        self._count = 0

    def count(self) -> int:
        return self._count

    def delete(self, **kwargs: Any) -> None:
        self.deleted.append(kwargs)

    def add(self, **kwargs: Any) -> None:
        self.add_calls.append(kwargs)
        self._count += len(kwargs.get("documents") or [])

def test_get_collection_count_returns_count(monkeypatch: pytest.MonkeyPatch) -> None:
    COLLECTION_COUNT = 7
    dummy = DummyCollectionUpsert()
    dummy._count = COLLECTION_COUNT
    monkeypatch.setattr(rag_store, "_get_collection", lambda: dummy)
    assert rag_store.get_collection_count() == COLLECTION_COUNT


def test_get_collection_count_returns_zero_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        def count(self) -> int:
            raise RuntimeError("boom")

    monkeypatch.setattr(rag_store, "_get_collection", lambda: Boom())
    assert rag_store.get_collection_count() == 0


def test_delete_by_doc_id_calls_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyCollectionNoUpsert()
    monkeypatch.setattr(rag_store, "_get_collection", lambda: dummy)

    rag_store._delete_by_doc_id("doc-1")

    assert dummy.deleted
    assert dummy.deleted[0].get("where") == {"doc_id": "doc-1"}


def test_delete_by_doc_id_ignores_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        def delete(self, **_kwargs: Any) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(rag_store, "_get_collection", lambda: Boom())
    rag_store._delete_by_doc_id("doc-1")  # 例外が外に出ないこと


def test_upsert_document_uses_upsert_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyCollectionUpsert()
    monkeypatch.setattr(rag_store, "_get_collection", lambda: dummy)

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[float(i)] for i, _ in enumerate(texts)]

    monkeypatch.setattr(rag_store, "embed_texts", fake_embed_texts)

    n = rag_store.upsert_document(
        doc_id="doc-1",
        text="三浦半島は神奈川県にあります。海がきれいです。",
        metadata={"source": "test", "tags": ["miura", "sea"]},
    )

    assert n > 0
    assert len(dummy.upsert_calls) == 1
    assert dummy.add_calls == []
    assert dummy.deleted == []


def test_upsert_document_falls_back_to_delete_and_add(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyCollectionNoUpsert()
    monkeypatch.setattr(rag_store, "_get_collection", lambda: dummy)

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[float(i)] for i, _ in enumerate(texts)]

    monkeypatch.setattr(rag_store, "embed_texts", fake_embed_texts)

    n = rag_store.upsert_document(
        doc_id="doc-1",
        text="三浦半島は神奈川県にあります。海がきれいです。",
        metadata={"source": "test"},
    )

    assert n > 0
    assert dummy.deleted
    assert dummy.add_calls

CHUNK_SIZE = 17

def test_get_chunk_size_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", CHUNK_SIZE)
    assert rag_store._get_chunk_size() == CHUNK_SIZE

    monkeypatch.setenv("RAG_CHUNK_SIZE", "nope")
    assert rag_store._get_chunk_size() == rag_store._DEFAULT_CHUNK_SIZE
