"""NetworkX-based knowledge graph for storing, querying, and persisting algorithm capabilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from knowledge.schema import (
    AlgorithmCapability,
    DataPattern,
    FailureCase,
    ValidationRecord,
)

_DEFAULT_JSON = Path(__file__).parent / "data" / "capabilities.json"

# Canonical relation type labels used on graph edges
RELATION_REQUIRES = "REQUIRES"
RELATION_SIMILAR_TO = "SIMILAR_TO"
RELATION_EXCLUDES = "EXCLUDES"


class KnowledgeGraph:
    """Directed knowledge graph of algorithm capabilities backed by NetworkX.

    Nodes represent AlgorithmCapability objects; edges encode semantic
    relations (REQUIRES, SIMILAR_TO, EXCLUDES) between them.

    Typical lifecycle
    -----------------
    kg = KnowledgeGraph()          # loads default capabilities.json
    caps = kg.search_by_conditions(["目标变量为二分类", "数据量 > 1000"])
    kg.update_validation_history(cap_id, record)
    kg.save_to_json()
    """

    def __init__(self, json_path: Optional[Path] = None) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._capabilities: Dict[str, AlgorithmCapability] = {}
        self._data_patterns: Dict[str, DataPattern] = {}
        self._failure_cases: Dict[str, FailureCase] = {}
        self._json_path: Path = Path(json_path) if json_path else _DEFAULT_JSON

        if self._json_path.exists():
            self.load_from_json(self._json_path)

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------

    def load_from_json(self, path: Path | str) -> None:
        """Parse capabilities.json and populate graph nodes and edges.

        Automatically wires semantic edges after loading; see
        :meth:`_wire_default_edges` for the full edge inventory.
        """
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        for cap_dict in data.get("capabilities", []):
            cap = AlgorithmCapability.from_dict(cap_dict)
            self._capabilities[cap.id] = cap
            self._graph.add_node(cap.id, capability=cap)

        for pat_dict in data.get("data_patterns", []):
            pat = DataPattern.from_dict(pat_dict)
            self._data_patterns[pat.id] = pat

        for fc_dict in data.get("failure_cases", []):
            fc = FailureCase.from_dict(fc_dict)
            self._failure_cases[fc.id] = fc

        self._wire_default_edges()

    def _wire_default_edges(self) -> None:
        """Add semantic edges between capability nodes after loading.

        Edge inventory
        --------------
        SMOTE → all classification nodes          (imbalanced data preprocessing)
        standard_scaler → logistic_regression     (linear model needs normalisation)
        standard_scaler → neural_network          (MLP is scale-sensitive)
        pca_reduction   → neural_network          (dimensionality reduction before NN)
        pca_reduction   → logistic_regression     (PCA + linear model pipeline)
        missing_imputer → logistic_regression     (LR requires no NaN)
        missing_imputer → neural_network          (MLP requires no NaN)
        xgboost    SIMILAR_TO random_forest
        lightgbm   SIMILAR_TO xgboost             (same GBDT family)
        catboost   SIMILAR_TO xgboost             (same GBDT family)
        neural_net SIMILAR_TO random_forest       (both strong non-linear models)
        feature_rfe REQUIRES  xgboost / random_forest / lightgbm / catboost
        """
        has = self._capabilities.__contains__

        # ── SMOTE → every classification node ─────────────────────────────
        if has("smote_oversampling"):
            for cap_id, cap in self._capabilities.items():
                if cap.category == "classification":
                    self.add_edge("smote_oversampling", cap_id, RELATION_REQUIRES)

        # ── standard_scaler → scale-sensitive models ───────────────────────
        for dst in ("logistic_regression_churn", "neural_network_churn"):
            if has("standard_scaler_preprocessing") and has(dst):
                self.add_edge("standard_scaler_preprocessing", dst, RELATION_REQUIRES)

        # ── PCA → downstream models ────────────────────────────────────────
        for dst in ("neural_network_churn", "logistic_regression_churn"):
            if has("pca_reduction") and has(dst):
                self.add_edge("pca_reduction", dst, RELATION_REQUIRES)

        # ── missing_value_imputer → NaN-sensitive models ───────────────────
        for dst in ("logistic_regression_churn", "neural_network_churn"):
            if has("missing_value_imputer") and has(dst):
                self.add_edge("missing_value_imputer", dst, RELATION_REQUIRES)

        # ── SIMILAR_TO edges among classifiers ─────────────────────────────
        similar_pairs = [
            ("xgboost_churn",        "random_forest_churn"),
            ("lightgbm_churn",       "xgboost_churn"),
            ("catboost_churn",       "xgboost_churn"),
            ("neural_network_churn", "random_forest_churn"),
        ]
        for src, dst in similar_pairs:
            if has(src) and has(dst):
                self.add_edge(src, dst, RELATION_SIMILAR_TO)

        # ── feature_selection_rfe → tree-based models ─────────────────────
        if has("feature_selection_rfe"):
            for dst in ("xgboost_churn", "random_forest_churn",
                        "lightgbm_churn", "catboost_churn"):
                if has(dst):
                    self.add_edge("feature_selection_rfe", dst, RELATION_REQUIRES)

    def save_to_json(self, path: Optional[Path | str] = None) -> None:
        """Serialise the current graph state back to JSON.

        Parameters
        ----------
        path:
            Destination file. Defaults to the path used at construction time.
        """
        dest = Path(path) if path else self._json_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "capabilities": [c.to_dict() for c in self._capabilities.values()],
            "data_patterns": [p.to_dict() for p in self._data_patterns.values()],
            "failure_cases": [f.to_dict() for f in self._failure_cases.values()],
        }
        with dest.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def export_graphml(self, path: Path | str) -> None:
        """Export the graph topology to GraphML format for external visualisation."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # GraphML requires string attributes; strip the Python object from nodes
        export_graph = nx.DiGraph()
        for node_id in self._graph.nodes:
            cap = self._capabilities.get(node_id)
            export_graph.add_node(
                node_id,
                name=cap.name if cap else node_id,
                category=cap.category if cap else "",
                description=cap.description if cap else "",
            )
        for src, dst, attrs in self._graph.edges(data=True):
            export_graph.add_edge(src, dst, relation=attrs.get("relation", ""))
        nx.write_graphml(export_graph, str(path))

    # ------------------------------------------------------------------
    # Node / edge mutation
    # ------------------------------------------------------------------

    def add_capability(self, cap: AlgorithmCapability) -> None:
        """Add or replace a capability node in the graph.

        Parameters
        ----------
        cap:
            The capability to insert. Overwrites any existing node with the
            same ``id``.
        """
        self._capabilities[cap.id] = cap
        self._graph.add_node(cap.id, capability=cap)

    def add_edge(self, src_id: str, dst_id: str, relation: str) -> None:
        """Add a directed edge between two capability nodes.

        Silently skips if either node does not exist in the graph.

        Parameters
        ----------
        src_id:
            Source capability id.
        dst_id:
            Destination capability id.
        relation:
            Semantic label, e.g. ``RELATION_REQUIRES``.
        """
        if src_id not in self._graph or dst_id not in self._graph:
            return
        self._graph.add_edge(src_id, dst_id, relation=relation)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_capability(self, cap_id: str) -> Optional[AlgorithmCapability]:
        """Return the AlgorithmCapability with the given id, or None."""
        return self._capabilities.get(cap_id)

    def get_all_capabilities(self) -> List[AlgorithmCapability]:
        """Return all capability objects as an unordered list."""
        return list(self._capabilities.values())

    def search_by_conditions(self, conditions: List[str]) -> List[AlgorithmCapability]:
        """Return capabilities whose applicable_conditions overlap with *conditions*.

        Matching is case-insensitive substring: a capability matches if **any**
        of its ``applicable_conditions`` contains any of the query strings as a
        substring.

        Parameters
        ----------
        conditions:
            List of condition strings from the user's task specification.

        Returns
        -------
        List of matching capabilities sorted by number of matching conditions
        (descending).
        """
        query_lower = [c.lower() for c in conditions]
        scored: List[tuple[int, AlgorithmCapability]] = []

        for cap in self._capabilities.values():
            cap_conds_lower = [c.lower() for c in cap.applicable_conditions]
            score = sum(
                1
                for q in query_lower
                if any(q in cap_cond or cap_cond in q for cap_cond in cap_conds_lower)
            )
            if score > 0:
                scored.append((score, cap))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [cap for _, cap in scored]

    def get_neighbors(
        self, cap_id: str, relation: Optional[str] = None
    ) -> List[AlgorithmCapability]:
        """Return successor capabilities reachable via edges from *cap_id*.

        Parameters
        ----------
        cap_id:
            The source node id.
        relation:
            If given, filter edges to only those with this relation label.
            If None, return all successors regardless of edge type.
        """
        if cap_id not in self._graph:
            return []
        neighbors = []
        for _, dst, attrs in self._graph.out_edges(cap_id, data=True):
            if relation is None or attrs.get("relation") == relation:
                cap = self._capabilities.get(dst)
                if cap:
                    neighbors.append(cap)
        return neighbors

    # ------------------------------------------------------------------
    # Validation history
    # ------------------------------------------------------------------

    def update_validation_history(
        self, cap_id: str, record: ValidationRecord
    ) -> None:
        """Append a ValidationRecord to a capability's history.

        Parameters
        ----------
        cap_id:
            Target capability id.
        record:
            The result to append.

        Raises
        ------
        KeyError
            If *cap_id* is not found in the graph.
        """
        cap = self._capabilities.get(cap_id)
        if cap is None:
            raise KeyError(f"Capability '{cap_id}' not found in knowledge graph.")
        cap.validation_history.append(record)
        # Keep the node attribute in sync
        self._graph.nodes[cap_id]["capability"] = cap

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_data_pattern(self, pattern_id: str) -> Optional[DataPattern]:
        """Return a DataPattern by id, or None."""
        return self._data_patterns.get(pattern_id)

    def get_failure_case(self, fc_id: str) -> Optional[FailureCase]:
        """Return a FailureCase by id, or None."""
        return self._failure_cases.get(fc_id)

    def get_failure_cases_for_capability(self, cap_id: str) -> List[FailureCase]:
        """Return all FailureCases associated with a given capability id."""
        return [fc for fc in self._failure_cases.values() if fc.related_capability_id == cap_id]

    def summary(self) -> dict:
        """Return a brief statistics summary of the graph."""
        return {
            "n_capabilities": len(self._capabilities),
            "n_edges": self._graph.number_of_edges(),
            "n_data_patterns": len(self._data_patterns),
            "n_failure_cases": len(self._failure_cases),
            "capability_ids": list(self._capabilities.keys()),
        }
