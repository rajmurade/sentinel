"""Incident memory.

Stores short incident narratives and retrieves similar past incidents so a
recurring pattern surfaces with its own history.

Two interchangeable backends sit behind one small interface:

* **Chroma** (default) — a persistent collection under ``CHROMA_DIR`` (the
  project-root ``.chroma/`` directory by default) using Chroma's built-in
  default embedding function. This is lighter than sentence-transformers and
  needs no PyTorch; the only cost is a one-time ONNX model download into
  Chroma's cache.
* **TF-IDF + cosine** (fallback) — a pure in-memory store built on
  scikit-learn, used when Chroma cannot be imported, initialized, or embed
  (for example on a machine with no network for the one-time model download).

The public helpers — :func:`add_incident`, :func:`find_similar`,
:func:`seed_demo_incidents` — have the same signatures on both backends, so the
app never breaks when only the fallback is available.

The embedding function and collection name live in :func:`_embedding_function`
and :data:`_COLLECTION_NAME`, the single place to swap either out later.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

CHROMA_DIR_DEFAULT: Path = Path(".chroma")

_UNSET: Any = object()
_EMBEDDING_FUNCTION: Any = _UNSET
COLLECTION_KEY: str = "sentinel_incidents"
DEFAULT_K: int = 3


def _chroma_dir() -> Path:
    """Where the persistent Chroma collection lives."""
    return Path(os.getenv("CHROMA_DIR", str(CHROMA_DIR_DEFAULT)))


def _embedding_function() -> Any:
    """The embedding function, in one place so it can be swapped later.

    Returns Chroma's built-in default embedding function. If it is unavailable
    the caller falls back to TF-IDF, so this may return ``None``.

    The result is cached on the function itself: the ONNX model behind
    Chroma's default embedding function takes a noticeable moment to load, and
    the app should pay that cost once at startup rather than on the first click.
    """
    global _EMBEDDING_FUNCTION
    if _EMBEDDING_FUNCTION is not _UNSET:
        return _EMBEDDING_FUNCTION
    try:
        from chromadb.utils.embedding_functions import (  # type: ignore[import-untyped]
            DefaultEmbeddingFunction,
        )

        _EMBEDDING_FUNCTION = DefaultEmbeddingFunction()
    except Exception:  # noqa: BLE001 - fall back to TF-IDF below
        _EMBEDDING_FUNCTION = None
    return _EMBEDDING_FUNCTION





def _new_id() -> str:
    return uuid.uuid4().hex


class _TfidfStore:
    """In-memory TF-IDF + cosine store. Same interface as the Chroma store."""

    kind: str = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]

        self._TfidfVectorizer = TfidfVectorizer
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._vectorizer: Any = None
        self._matrix: Any = None

    def count(self) -> int:
        return len(self._ids)

    def add_many(self, items: list[tuple[str, dict[str, Any]]]) -> list[str]:
        """Add (document, metadata) pairs, assigning each an id."""
        ids: list[str] = []
        for document, metadata in items:
            ids.append(_new_id())
            self._ids.append(ids[-1])
            self._documents.append(document)
            self._metadatas.append(metadata)
        self._vectorizer = None
        self._matrix = None
        return ids

    def _ensure_fitted(self) -> None:
        if self._vectorizer is None and self._documents:
            self._vectorizer = self._TfidfVectorizer(stop_words="english")
            self._matrix = self._vectorizer.fit_transform(self._documents)

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Top-``k`` documents by cosine similarity to ``query``."""
        if not self._documents:
            return []
        self._ensure_fitted()
        import numpy as np  # type: ignore[import-untyped]
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        order = np.argsort(-scores)[:k]
        return [
            {
                "id": self._ids[i],
                "document": self._documents[i],
                "metadata": dict(self._metadatas[i]),
                "similarity": float(scores[i]),
            }
            for i in order
        ]


class _ChromaStore:
    """Persistent Chroma collection with cosine distance."""

    kind: str = "chroma"

    def __init__(self, dir_path: Path) -> None:
        import chromadb  # type: ignore[import-untyped]

        client = chromadb.PersistentClient(path=str(dir_path))
        self._collection = client.get_or_create_collection(
            name=COLLECTION_KEY,
            embedding_function=_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self._collection.count())

    def add_many(self, items: list[tuple[str, dict[str, Any]]]) -> list[str]:
        if not items:
            return []
        ids = [_new_id() for _ in items]
        self._collection.add(
            ids=ids,
            documents=[document for document, _ in items],
            metadatas=[metadata for _, metadata in items],
        )
        return ids

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_texts=[query],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        matches: list[dict[str, Any]] = []
        for index in range(len(ids)):
            distance = float(distances[index]) if index < len(distances) else 1.0
            matches.append(
                {
                    "id": ids[index],
                    "document": documents[index] if index < len(documents) else "",
                    "metadata": dict(metadatas[index]) if index < len(metadatas) else {},
                    "similarity": max(0.0, 1.0 - distance),
                }
            )
        return matches


_STORE: Any = None


def _get_store() -> Any:
    """The process-wide store, built lazily so tests can redirect the directory."""
    global _STORE
    if _STORE is None:
        try:
            _STORE = _ChromaStore(_chroma_dir())
        except Exception:  # noqa: BLE001 - any Chroma problem -> TF-IDF
            _STORE = _TfidfStore()
    return _STORE


def reset_store() -> None:
    """Drop the cached store. Used by tests and after a config change."""
    global _STORE
    _STORE = None


def warm_up_embedding() -> bool:
    """Force the embedding model to load now instead of on the first query.

    Returns ``True`` when a real embedding function is available, ``False`` when
    the caller should expect the TF-IDF fallback. Never raises.
    """
    try:
        return _embedding_function() is not None
    except Exception:  # noqa: BLE001 - memory is optional, degrade the run
        return False


def _document(headline: str, metric: str, driver: str, cause: str) -> str:
    return f"{headline} {metric} {driver} {cause}"


def add_incident(
    headline: str,
    metric: str,
    driver: str,
    cause: str,
    date: str,
    is_demo: bool = False,
) -> dict[str, Any]:
    """Store one incident and its recorded cause.

    ``headline + metric + driver + cause`` becomes the embedded document; the
    fields are kept in the metadata so they can be shown verbatim when the
    incident is recalled later.
    """
    store = _get_store()
    metadata: dict[str, Any] = {
        "headline": str(headline),
        "date": str(date),
        "metric": str(metric),
        "driver": str(driver),
        "cause": str(cause),
        "is_demo": bool(is_demo),
    }
    ids = store.add_many([(_document(headline, metric, driver, cause), metadata)])
    return {"stored": True, "id": ids[0] if ids else None, "backend": backend()}


def find_similar(report: dict[str, Any], k: int = DEFAULT_K) -> list[dict[str, Any]]:
    """Recall up to ``k`` past incidents resembling ``report``.

    The query is built from the report's headline, metric, and likely driver,
    so a similar *pattern* surfaces — not a claim that the cause is the same.
    Each match carries a ``similarity`` score in ``[0, 1]`` and its metadata.
    """
    parts = [
        str(report.get("headline", "")),
        str(report.get("metric", "")),
        str(report.get("likely_driver", "")),
    ]
    query = " ".join(part for part in parts if part)

    store = _get_store()
    matches = store.search(query, k)
    results: list[dict[str, Any]] = []
    for match in matches:
        metadata = match.get("metadata", {})
        results.append(
            {
                "id": match.get("id"),
                "date": metadata.get("date"),
                "headline": metadata.get("headline", ""),
                "metric": metadata.get("metric"),
                "driver": metadata.get("driver"),
                "cause": metadata.get("cause"),
                "is_demo": bool(metadata.get("is_demo", False)),
                "similarity": float(match.get("similarity", 0.0)),
            }
        )
    return results


def _demo_incidents() -> list[dict[str, str]]:
    return [
        {
            "headline": "Revenue drop on mobile and organic",
            "metric": "revenue",
            "driver": "channel=organic, device=mobile",
            "cause": "checkout tracking tag broke after app release",
            "date": "2026-01-12",
        },
        {
            "headline": "Sessions surge with conversion collapse in paid",
            "metric": "sessions",
            "driver": "channel=paid",
            "cause": "low-quality traffic from a new ad campaign",
            "date": "2026-02-03",
        },
        {
            "headline": "Refunds spike in one product category",
            "metric": "refunds",
            "driver": "product_category=electronics",
            "cause": "faulty batch shipped from supplier",
            "date": "2026-03-09",
        },
        {
            "headline": "Orders dip across all regions on one day",
            "metric": "orders",
            "driver": "region=north_america, region=europe, region=apac",
            "cause": "payment gateway outage",
            "date": "2026-04-21",
        },
    ]


def seed_demo_incidents() -> dict[str, Any]:
    """Insert the four historical demo incidents, but only into an empty store."""
    store = _get_store()
    if store.count() > 0:
        return {"seeded": 0, "backend": backend(), "reason": "store not empty"}

    demos = _demo_incidents()
    items: list[tuple[str, dict[str, Any]]] = []
    for demo in demos:
        metadata = dict(demo)
        metadata["is_demo"] = True
        document = _document(
            demo["headline"], demo["metric"], demo["driver"], demo["cause"]
        )
        items.append((document, metadata))
    store.add_many(items)
    return {"seeded": len(demos), "backend": backend()}


def backend() -> str:
    """Which backend is active: ``"chroma"`` or ``"tfidf"``."""
    return getattr(_get_store(), "kind", "tfidf")
