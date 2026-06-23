"""ChromaDB-backed semantic vector store for algorithm capability retrieval.

Uses an offline character n-gram hashing embedding so no network access or
model download is required.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import chromadb
from chromadb import Documents, Embeddings, EmbeddingFunction

if TYPE_CHECKING:
    from knowledge.graph import KnowledgeGraph
    from knowledge.schema import AlgorithmCapability

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "algorithm_capabilities"
_DEFAULT_PERSIST_DIR = str(Path(__file__).parent / "data" / "chroma_db")
_EMBED_DIM = 256  # dimensionality of the hashing embedding


# ---------------------------------------------------------------------------
# Offline embedding function
# ---------------------------------------------------------------------------

class _NgramHashEmbedding(EmbeddingFunction[Documents]):
    """Deterministic offline embedding via character n-gram hashing trick.

    Splits each text into character 1-, 2-, and 3-grams, hashes each gram
    with MD5, and accumulates counts into a fixed-size dense vector.
    Works without any network access or downloaded model.

    Chromadb's ``EmbeddingFunction.__init_subclass__`` automatically
    L2-normalises the returned vectors, so cosine distance equals L2 distance
    on the unit sphere.
    """

    def __init__(self, dim: int = _EMBED_DIM) -> None:
        self._dim = dim

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        return [self._embed_one(text) for text in input]

    def _embed_one(self, text: str) -> List[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        for n in (1, 2, 3):
            for i in range(len(text) - n + 1):
                gram = text[i : i + n]
                h = int(hashlib.md5(gram.encode("utf-8", errors="replace")).hexdigest(), 16)
                vec[h % self._dim] += 1.0
        # The base class wrapper normalises automatically; returning raw counts
        # is fine, but we guard against the all-zero vector just in case.
        if vec.sum() == 0:
            vec[0] = 1.0
        return vec.tolist()

    @staticmethod
    def name() -> str:
        return "NgramHashEmbedding"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "_NgramHashEmbedding":
        return _NgramHashEmbedding(dim=config.get("dim", _EMBED_DIM))

    def get_config(self) -> Dict[str, Any]:
        return {"dim": self._dim}


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """Semantic search layer over algorithm capabilities using ChromaDB.

    Embeddings are produced by :class:`_NgramHashEmbedding`, which is fully
    offline and requires no model downloads.

    Typical lifecycle
    -----------------
    vs = VectorStore()
    vs.build_from_graph(kg)          # index all capabilities once
    hits = vs.search("不平衡分类")   # returns ranked dicts with scores
    """

    def __init__(self, persist_dir: str = _DEFAULT_PERSIST_DIR) -> None:
        """Initialise a persistent ChromaDB client and open the collection.

        Parameters
        ----------
        persist_dir:
            Directory where ChromaDB stores its SQLite data. Created
            automatically if it does not exist.
        """
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._embed_fn = _NgramHashEmbedding(dim=_EMBED_DIM)
        self._client = chromadb.PersistentClient(path=persist_dir)
        try:
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except ValueError:
            # Stale collection was created with a different embedding function;
            # delete and recreate with the correct one.
            self._client.delete_collection(_COLLECTION_NAME)
            self._collection = self._client.create_collection(
                name=_COLLECTION_NAME,
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
        logger.debug(
            "VectorStore ready: collection=%s dir=%s", _COLLECTION_NAME, persist_dir
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_from_graph(self, kg: "KnowledgeGraph") -> None:
        """Index every capability in *kg* into the collection.

        The retrieval text concatenates the fields most useful for matching a
        user's plain-language query::

            "{name}. {description}. 适用条件: {conditions}. 指标: {metrics}"

        Uses ``upsert`` so calling this method multiple times is safe.

        Parameters
        ----------
        kg:
            Populated :class:`~knowledge.graph.KnowledgeGraph` instance.
        """
        capabilities = kg.get_all_capabilities()
        if not capabilities:
            logger.warning("build_from_graph: knowledge graph is empty, nothing indexed.")
            return

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for cap in capabilities:
            conditions_text = "、".join(cap.applicable_conditions)
            metrics_text = "、".join(cap.metrics)
            doc = (
                f"{cap.name}. {cap.description}. "
                f"适用条件: {conditions_text}. "
                f"指标: {metrics_text}"
            )
            ids.append(cap.id)
            documents.append(doc)
            metadatas.append(
                {
                    "name": cap.name,
                    "category": cap.category,
                    "metrics": ",".join(cap.metrics),
                    "description": cap.description,
                }
            )

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Indexed %d capabilities into VectorStore.", len(ids))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Semantic search over indexed capabilities.

        Parameters
        ----------
        query:
            Natural-language search string.
        n_results:
            Maximum number of results to return.

        Returns
        -------
        List of dicts sorted by descending similarity score::

            [{"id": str, "name": str, "category": str,
              "score": float,          # cosine similarity in [0, 1]
              "description": str}, ...]
        """
        count = self._collection.count()
        if count == 0:
            logger.warning("search called but collection is empty.")
            return []

        actual_n = min(n_results, count)
        results = self._collection.query(
            query_texts=[query],
            n_results=actual_n,
            include=["metadatas", "distances"],
        )

        hits: List[Dict] = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for cap_id, distance, meta in zip(ids, distances, metadatas):
            hits.append(
                {
                    "id": cap_id,
                    "name": meta.get("name", ""),
                    "category": meta.get("category", ""),
                    "score": round(1.0 - float(distance), 4),
                    "description": meta.get("description", ""),
                }
            )

        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits

    def get_by_id(self, capability_id: str) -> Optional[Dict]:
        """Fetch metadata for a single capability by exact id.

        Returns ``None`` if the id is not found in the collection.
        """
        try:
            result = self._collection.get(
                ids=[capability_id],
                include=["metadatas", "documents"],
            )
        except Exception as exc:
            logger.error("get_by_id failed for id=%s: %s", capability_id, exc)
            return None

        ids = result.get("ids", [])
        if not ids:
            return None

        meta = (result.get("metadatas") or [{}])[0]
        doc = (result.get("documents") or [""])[0]
        return {
            "id": ids[0],
            "name": meta.get("name", ""),
            "category": meta.get("category", ""),
            "description": meta.get("description", ""),
            "document": doc,
        }

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update_capability(self, capability: "AlgorithmCapability") -> None:
        """Re-index a single capability after it has been modified in the graph.

        Deletes the old entry first to ensure the embedding is fully
        regenerated (``update`` only patches metadata, not the stored vector).

        Parameters
        ----------
        capability:
            The updated :class:`~knowledge.schema.AlgorithmCapability`.
        """
        try:
            self._collection.delete(ids=[capability.id])
        except Exception:
            pass  # not present yet — that's fine

        conditions_text = "、".join(capability.applicable_conditions)
        metrics_text = "、".join(capability.metrics)
        doc = (
            f"{capability.name}. {capability.description}. "
            f"适用条件: {conditions_text}. "
            f"指标: {metrics_text}"
        )
        self._collection.add(
            ids=[capability.id],
            documents=[doc],
            metadatas=[
                {
                    "name": capability.name,
                    "category": capability.category,
                    "metrics": ",".join(capability.metrics),
                    "description": capability.description,
                }
            ],
        )
        logger.debug("Updated capability %s in VectorStore.", capability.id)
