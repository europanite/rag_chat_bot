import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import chromadb
import requests

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Simple data containers
# -------------------------------------------------------------------
def _slug(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s).strip("_").lower()
    return s or "tag"

def _chroma_safe_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """
    Chroma metadata: list/dict
    - list[str] -> "a,b,c"
    - dict/list -> JSON
    """
    out: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        key = str(k)

        if v is None or isinstance(v, (str, int, float, bool)):
            out[key] = v
            continue

        if isinstance(v, list):
            if all(isinstance(x, str) for x in v):
                out[key] = ",".join(v)
                if key == "tags":
                    for t in v:
                        out[f"tag__{_slug(t)}"] = True
            else:
                out[key] = json.dumps(v, ensure_ascii=False)
            continue

        if isinstance(v, dict):
            out[key] = json.dumps(v, ensure_ascii=False)
            continue

        out[key] = str(v)

    return out

class DocumentChunk:
    """Internal representation of a chunked piece of text."""

    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata or {}


class RAGChunk:
    """Chunk plus distance from query, used as RAG context."""

    def __init__(self, text, distance, metadata=None):
        self.text = text
        self.distance = distance
        self.metadata = metadata or {}


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

_CHROMA_DB_DIR_ENV = "CHROMA_DB_DIR"
_CHROMA_COLLECTION_ENV = "CHROMA_COLLECTION_NAME"
_DEFAULT_CHROMA_DB_DIR = "/chroma"
_DEFAULT_COLLECTION_NAME = "documents"

_OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
_OLLAMA_EMBED_MODEL_ENV = "EMBEDDING_MODEL"
_DEFAULT_OLLAMA_BASE_URL = "http://ollama:11434"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"

_RAG_CHUNK_SIZE_ENV = "RAG_CHUNK_SIZE"

# Lint-friendly constants
_HTTP_STATUS_NOT_FOUND = 404
_JSON_STRING_CONTROL_CHAR_MAX_EXCLUSIVE = 0x20
_OLLAMA_REQUEST_TIMEOUT_SECONDS = 30
_OLLAMA_ROUTE_NOT_FOUND_MARKER = "404 page not found"


_client = None
_collection = None


# -------------------------------------------------------------------
# Helpers to read environment
# -------------------------------------------------------------------


def _get_chroma_db_dir():
    """Return the directory where Chroma will store data."""
    value = os.getenv(_CHROMA_DB_DIR_ENV)
    if value:
        return value
    return _DEFAULT_CHROMA_DB_DIR


def _get_chroma_collection_name():
    """Return the Chroma collection name."""
    value = os.getenv(_CHROMA_COLLECTION_ENV)
    if value:
        return value
    return _DEFAULT_COLLECTION_NAME


def _get_ollama_base_url():
    """Return base URL for Ollama HTTP API, e.g. http://ollama:11434"""
    value = os.getenv(_OLLAMA_BASE_URL_ENV)
    if value:
        return value
    return _DEFAULT_OLLAMA_BASE_URL


def _get_embedding_model():
    """Return embedding model name, e.g. `nomic-embed-text`."""
    value = os.getenv(_OLLAMA_EMBED_MODEL_ENV)
    if value:
        return value
    return _DEFAULT_EMBED_MODEL


# -------------------------------------------------------------------
# Chroma client / collection singletons
# -------------------------------------------------------------------


def _get_chroma_client():
    """
    Return a module-level singleton Chroma client.

    Tests expect:
      * The path is taken from CHROMA_DB_DIR (or default).
      * The same instance is returned when called twice.
    """

    if _client is not None:
        return _client

    db_dir = _get_chroma_db_dir()
    logger.info("Creating Chroma client at %s", db_dir)
    client = chromadb.PersistentClient(path=db_dir)
    # keep module-level cache without using `global`
    globals()["_client"] = client
    return client


def _get_collection():
    """
    Return a module-level singleton Chroma collection.

    Tests monkeypatch `_collection` directly, so if `_collection` is not
    None we just give it back and do NOT create a new one.
    """
    if _collection is not None:
        return _collection

    client = _get_chroma_client()
    name = _get_chroma_collection_name()
    logger.info("Getting/creating Chroma collection %s", name)
    collection = client.get_or_create_collection(name=name)
    # store on module without `global`
    globals()["_collection"] = collection
    return collection


# -------------------------------------------------------------------
# Chunking
# -------------------------------------------------------------------

_DEFAULT_CHUNK_SIZE=256


def _get_chunk_size() -> int:
    """
    Read chunk size from env (RAG_CHUNK_SIZE). Falls back to _DEFAULT_CHUNK_SIZE.
    """
    raw = os.getenv(_RAG_CHUNK_SIZE_ENV, "").strip()
    if not raw:
        return _DEFAULT_CHUNK_SIZE
    try:
        v = int(raw)
        return v if v > 0 else _DEFAULT_CHUNK_SIZE
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to default=%d",
            _RAG_CHUNK_SIZE_ENV,
            raw,
            _DEFAULT_CHUNK_SIZE,
        )
        return _DEFAULT_CHUNK_SIZE

JP_SENT_SPLIT = re.compile(r"(?<=[。！？])")  # noqa: RUF001
CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

def chunk_text(text: str, max_tokens: int = _DEFAULT_CHUNK_SIZE) -> list[DocumentChunk]:
    """Split text into chunks with basic sentence-aware behavior.

    - For CJK text (Japanese/Chinese/Korean), split on Japanese punctuation and
      *do not* pack multiple sentences into a single chunk by default. This keeps
      retrieval snippets semantically tight.
    - For non-CJK text, split by whitespace into ~`max_tokens` words.

    Notes:
    - This is a simple heuristic chunker. If you need more accurate token counts,
      integrate a tokenizer (e.g., tiktoken) and chunk by tokens instead of chars.
    """
    if not text:
        return []

    has_cjk = CJK_RE.search(text) is not None
    chunks: list[DocumentChunk] = []

    if has_cjk:
        # JP_SENT_SPLIT uses look-behind so punctuation stays with the sentence.
        sentences = [s.strip() for s in JP_SENT_SPLIT.split(text) if s.strip()]
        for s in sentences:
            if len(s) <= max_tokens:
                chunk_index = len(chunks)
                chunks.append(
                    DocumentChunk(
                        s,
                        {"chunk_index": chunk_index, "index": chunk_index},
                    )
                )
                continue

            # Very long "sentence" (or no punctuation) -> split by characters.
            for i in range(0, len(s), max_tokens):
                part = s[i : i + max_tokens].strip()
                if not part:
                    continue
                chunk_index = len(chunks)
                chunks.append(
                    DocumentChunk(
                        part,
                        {"chunk_index": chunk_index, "index": chunk_index},
                    )
                )

    else:
        words = text.split()
        current: list[str] = []
        count = 0

        for w in words:
            if count >= max_tokens and current:
                chunk_index = len(chunks)
                chunk_text_value = " ".join(current)
                chunks.append(
                    DocumentChunk(
                        chunk_text_value,
                        {"chunk_index": chunk_index, "index": chunk_index},
                    )
                )
                current = []
                count = 0

            current.append(w)
            count += 1

        if current:
            chunk_index = len(chunks)
            chunk_text_value = " ".join(current)
            chunks.append(
                DocumentChunk(
                    chunk_text_value,
                    {"chunk_index": chunk_index, "index": chunk_index},
                )
            )

    total = len(chunks)
    for c in chunks:
        c.metadata["total_chunks"] = total

    return chunks

# -------------------------------------------------------------------
# Embeddings (Ollama)
# -------------------------------------------------------------------

def _ollama_embed_attempts(model: str, text: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("/api/embed", {"model": model, "input": text}),        # newer
        ("/api/embeddings", {"model": model, "prompt": text}),  # older
    ]


def _safe_response_text(response: requests.Response) -> str:
    try:
        return response.text or ""
    except Exception:
        return ""


def _extract_ollama_error_message(response: requests.Response, fallback: str) -> str:
    try:
        data = response.json()
        if isinstance(data, dict) and isinstance(data.get("error"), str):
            return data["error"]
    except Exception:
        pass
    return fallback


def _should_try_next_endpoint(
    response: requests.Response,
    model: str,
    http_err: requests.HTTPError,
) -> bool:
    status = getattr(response, "status_code", None)
    if status != _HTTP_STATUS_NOT_FOUND:
        return False

    body_text = _safe_response_text(response)
    err_msg = _extract_ollama_error_message(response, body_text)
    lower = (err_msg or "").lower()

    # Ollama: 404 can mean "model does not exist"
    if "model" in lower and ("not found" in lower or "does not exist" in lower):
        raise RuntimeError(
            f"Ollama model is not available: {model!r}. "
            f"Pull it first: `docker compose exec ollama ollama pull {model}`"
        ) from http_err

    # Missing route => try next endpoint
    return _OLLAMA_ROUTE_NOT_FOUND_MARKER in (body_text or "").lower()


def _extract_embedding_from_response(data: Any) -> list[float] | None:
    if not isinstance(data, dict):
        return None

    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
        return embeddings[0]

    embedding = data.get("embedding")
    if isinstance(embedding, list):
        return embedding

    rows = data.get("data")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        emb = rows[0].get("embedding")
        if isinstance(emb, list):
            return emb

    return None


def _embed_with_ollama(text: str) -> list[float]:
    base_url = _get_ollama_base_url().rstrip("/")
    model = _get_embedding_model()

    last_error: Exception | None = None
    for endpoint, payload in _ollama_embed_attempts(model, text):
        url = base_url + endpoint
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=_OLLAMA_REQUEST_TIMEOUT_SECONDS,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as http_err:
                if _should_try_next_endpoint(response, model, http_err):
                    last_error = http_err
                    continue
                raise

            data = response.json()
            embedding = _extract_embedding_from_response(data)
            if isinstance(embedding, list) and embedding:
                return embedding
            raise ValueError(f"Unexpected embedding response format at {endpoint}")
        except Exception as exc:
            last_error = exc

    if last_error is None:
        raise RuntimeError("Ollama embedding failed")
    raise RuntimeError(f"Ollama embedding failed: {last_error}") from last_error


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    cleaned_texts = []
    for t in texts:
        if not t:
            continue
        stripped = t.strip()
        if not stripped:
            continue
        cleaned_texts.append(stripped)

    if not cleaned_texts:
        return []

    embeddings: list[list[float]] = []
    errors: list[Exception] = []

    for t in cleaned_texts:
        try:
            emb = _embed_with_ollama(t)
            embeddings.append(emb)
        except Exception as exc:  # requests.HTTPError
            logger.exception("Embedding failed for text chunk", exc_info=exc)
            errors.append(exc)

    if not embeddings:
        raise RuntimeError(
            f"Failed to embed any of the {len(cleaned_texts)} text chunks; "
            f"last error: {errors[-1] if errors else 'unknown'}"
        )

    return embeddings


# -------------------------------------------------------------------
# Ingestion
# -------------------------------------------------------------------

def add_document(text: str) -> None:
    """Split text into chunks, embed them, and store in Chroma."""
    chunks = chunk_text(text, max_tokens=_get_chunk_size())
    if not chunks:
        logger.warning("No chunks produced from text; nothing to add.")
        return

    documents = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    embeddings = embed_texts(documents)

    if not embeddings:
        raise RuntimeError("No embeddings produced; document not stored.")

    if len(embeddings) != len(documents):
        raise ValueError(
            f"Embedding count mismatch: {len(embeddings)} vs {len(documents)}"
        )

    collection = _get_collection()
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

# -------------------------------------------------------------------
# Retrieval
# -------------------------------------------------------------------


def query_similar_chunks(question, top_k=3):
    """
    Embed the question, query Chroma, and return a list of RAGChunk.

    Tests expect:

        query_similar_chunks(question: str, top_k: int = 3) -> list[RAGChunk]

    and then they monkeypatch this function in some API tests, but in
    `test_rag_store.py` they call the real one with a DummyCollection that
    implements:

        def query(self,
                  query_embeddings: list[list[float]],
                  n_results: int,
                  include: list[str] | None = None) -> dict[str, Any]

    The expected keys in the response are:

        {
          "documents": [[...]],
          "metadatas": [[...]],
          "distances": [[...]],
        }
    """
    cleaned = (question or "").strip()
    if not cleaned:
        return []

    query_embeddings = embed_texts([cleaned])
    if not query_embeddings:
        # Could happen if embedding fails; in that case, just return empty.
        return []

    collection = _get_collection()
    result = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs_lists = result.get("documents") or [[]]
    metas_lists = result.get("metadatas") or [[]]
    dists_lists = result.get("distances") or [[]]

    docs = docs_lists[0] if docs_lists else []
    metas = metas_lists[0] if metas_lists else []
    dists = dists_lists[0] if dists_lists else []

    chunks: list[RAGChunk] = []
    for doc, raw_meta, dist in zip(docs, metas, dists, strict=False):
        if isinstance(raw_meta, dict):
            normalized_meta = raw_meta
        elif raw_meta is None:
            normalized_meta = {}
        else:
            # Normalize strange metadata formats used in some backends.
            normalized_meta = {"value": raw_meta}

        chunks.append(
            RAGChunk(
                text=doc,
                distance=dist,
                metadata=normalized_meta,
            )
        )

    return chunks

# -------------------------------------------------------------------
# JSON directory ingestion (no UI / no input form)
# -------------------------------------------------------------------


def list_json_files(docs_dir: str) -> list[str]:
    """Return a sorted list of *.json files under `docs_dir` (non-recursive)."""
    try:
        root = Path(docs_dir)
    except Exception:
        return []

    if not root.exists() or not root.is_dir():
        return []

    return sorted([str(p) for p in root.glob("*.json")])

def _escape_control_chars_inside_json_strings(raw: str) -> str:
    """
    Make 'almost JSON' parseable by escaping control characters (< 0x20)
    that appear *inside* JSON double-quoted strings.

    This fixes hand-edited JSON like:
      "text": "hello
      world"
    which is invalid JSON (literal newline in a string).
    """
    out: list[str] = []
    in_string = False
    escaped = False

    for ch in raw:
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
                escaped = False
            continue

        # inside a JSON string
        if escaped:
            out.append(ch)
            escaped = False
            continue

        if ch == "\\":
            out.append(ch)
            escaped = True
            continue

        if ch == '"':
            out.append(ch)
            in_string = False
            continue

        code = ord(ch)
        if code < _JSON_STRING_CONTROL_CHAR_MAX_EXCLUSIVE:
            if ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(f"\\u{code:04x}")
        else:
            out.append(ch)

    return "".join(out)


def _load_json_file(path: str) -> list[dict]:
    p = Path(path)

    raw = p.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        fixed = _escape_control_chars_inside_json_strings(raw)
        logger.warning("Invalid JSON fixed by escaping control characters: %s", path)
        data = json.loads(fixed)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError(f"JSON must be an object or list of objects: {path}")

    docs: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("id")
        text = item.get("text")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"Missing/invalid 'id' in {path}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing/invalid 'text' for id={doc_id!r} in {path}")
        default_source = p.resolve().as_uri()  # file:///...
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        docs.append(
            {
                "id": doc_id.strip(),
                "text": text,
                "source": item.get("source") or default_source,
                "file": str(p.name),
                "metadata": meta,
            }
        )

    return docs



def get_collection_count() -> int:
    """Return number of stored records (chunks) in the Chroma collection."""
    try:
        return int(_get_collection().count())
    except Exception:
        return 0


def _delete_by_doc_id(doc_id: str) -> None:
    """Best-effort delete of all chunks for a given doc_id."""
    col = _get_collection()
    try:
        # Chroma collections usually support `where` filtering.
        col.delete(where={"doc_id": doc_id})
    except Exception:
        # Fallback: try deleting by ids if we can infer them later; otherwise ignore.
        return


def upsert_document(
    doc_id: str,
    text: str,
    *,
    source: str | None = None,
    metadata: dict | None = None,
    max_tokens: int | None = None,
) -> int:
    """Chunk + embed + store one document with deterministic chunk IDs.

    IDs are generated as: "{doc_id}:{chunk_index}" so re-indexing updates the same
    logical records.

    Returns:
        Number of chunks stored.
    """
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    size = int(max_tokens) if isinstance(max_tokens, int) and max_tokens > 0 else _get_chunk_size()
    chunks = chunk_text(text, max_tokens=size)
    if not chunks:
        return 0

    chunk_texts = [c.text for c in chunks]
    embeddings = embed_texts(chunk_texts)

    base_meta = _chroma_safe_metadata(dict(metadata or {}))
    base_meta["doc_id"] = doc_id
    if source:
        base_meta.setdefault("source", source)

    metadatas: list[dict] = []
    ids: list[str] = []
    for c in chunks:
        m = dict(base_meta)
        if isinstance(c.metadata, dict):
            m.update(c.metadata)

        m = _chroma_safe_metadata(m)

        # Ensure deterministic chunk id.
        idx = m.get("chunk_index")
        if idx is None:
            idx = m.get("index", 0)

        metadatas.append(m)
        ids.append(f"{doc_id}:{idx}")

    col = _get_collection()

    # Prefer upsert if available. Otherwise, delete existing doc chunks and add.
    if hasattr(col, "upsert"):
        col.upsert(
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
            ids=ids,
        )
    else:
        _delete_by_doc_id(doc_id)
        col.add(
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
            ids=ids,
        )

    return len(chunks)


def ingest_json_dir(docs_dir: str) -> dict[str, int]:
    """Ingest every document found in JSON files under `docs_dir`.

    Expected JSON format (per entry):
      {
        "id": "unique_doc_id",
        "text": "document text ...",
        "source": "optional",
        "metadata": {"optional": "dict"}
      }

    Returns:
        {"documents": <count>, "chunks": <count>, "files": <count>}
    """
    json_files = list_json_files(docs_dir)
    documents = 0
    chunks = 0

    for path in json_files:
        entries = _load_json_file(path)
        for e in entries:
            documents += 1
            meta = dict(e.get("metadata") or {})
            # Preserve the file name for traceability.
            meta.setdefault("file", e.get("file"))
            chunks += upsert_document(
                e["id"],
                e["text"],
                source=e.get("source"),
                metadata=meta,
            )

    return {"documents": documents, "chunks": chunks, "files": len(json_files)}


def reset_collection() -> None:
    """Delete and recreate the Chroma collection (best-effort)."""
    name = _get_chroma_collection_name()
    try:
        _get_chroma_client().delete_collection(name)
    except Exception:
        pass
    globals()["_collection"] = None
    _get_collection()


def rebuild_from_json_dir(docs_dir: str) -> dict[str, int]:
    """Clear the collection and ingest JSON documents from `docs_dir`."""
    reset_collection()
    return ingest_json_dir(docs_dir)

