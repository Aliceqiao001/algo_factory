"""Knowledge retrieval Agent: semantic search + graph enrichment for algorithm capabilities."""

import logging
from typing import TYPE_CHECKING, Dict, List

from agents.state import AgentState

if TYPE_CHECKING:
    from knowledge.graph import KnowledgeGraph
    from knowledge.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Keywords that signal class-imbalance handling is needed
_IMBALANCE_KEYWORDS = ("不平衡", "imbalance", "imbalanced", "smote", "过采样", "少数类")
_SMOTE_ID = "smote_oversampling"


class RetrievalAgent:
    """Retrieves the most relevant algorithm capabilities for the current task.

    Combines semantic vector search with graph-based enrichment and applies
    constraint-driven injection rules (e.g. always include SMOTE when the
    user mentions class imbalance).
    """

    def __init__(self, kg: "KnowledgeGraph", vs: "VectorStore") -> None:
        self._kg = kg
        self._vs = vs

    def run(self, state: AgentState) -> dict:
        """Search for capabilities matching ``state["user_query"]`` and enrich results.

        Returns
        -------
        dict with key ``retrieved_capabilities``: a list of enriched capability dicts,
        each containing id, name, category, score, description,
        applicable_conditions, dependencies, and metrics.
        """
        query = state["user_query"]
        constraints = state.get("constraints", [])

        # 1. Semantic search
        raw_results = self._vs.search(query, n_results=3)
        logger.debug("Vector search returned %d results for query: %s", len(raw_results), query)

        # 2. Constraint-driven injection: ensure SMOTE is present when needed
        if self._needs_smote(constraints, query):
            ids_in_results = {r["id"] for r in raw_results}
            if _SMOTE_ID not in ids_in_results:
                smote_hit = self._vs.get_by_id(_SMOTE_ID)
                if smote_hit:
                    smote_hit["score"] = 0.0  # injected, not ranked
                    raw_results.append(smote_hit)
                    logger.debug("Injected %s due to imbalance constraint", _SMOTE_ID)

        # 3. Enrich each hit with full graph data
        enriched: List[Dict] = []
        for hit in raw_results:
            cap = self._kg.get_capability(hit["id"])
            if cap is None:
                enriched.append(hit)
                continue
            enriched.append(
                {
                    "id": cap.id,
                    "name": cap.name,
                    "category": cap.category,
                    "score": hit.get("score", 0.0),
                    "description": cap.description,
                    "applicable_conditions": cap.applicable_conditions,
                    "dependencies": cap.dependencies,
                    "metrics": cap.metrics,
                }
            )

        return {"retrieved_capabilities": enriched}

    @staticmethod
    def _needs_smote(constraints: List[str], query: str) -> bool:
        combined = " ".join(constraints) + " " + query
        return any(kw in combined for kw in _IMBALANCE_KEYWORDS)
