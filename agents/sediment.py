"""Sediment Agent: persists validated results back into the knowledge graph and vector store."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from agents.state import AgentState
from knowledge.schema import ValidationRecord

if TYPE_CHECKING:
    from knowledge.graph import KnowledgeGraph
    from knowledge.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SedimentAgent:
    """Writes execution results back to the knowledge base after a pipeline run.

    On success, the validated code and metrics are appended to the capability's
    ``validation_history``; the graph is saved to JSON and the vector store
    entry is refreshed so future semantic searches reflect the updated record.
    """

    def __init__(self, kg: "KnowledgeGraph", vs: "VectorStore") -> None:
        self._kg = kg
        self._vs = vs

    def run(self, state: AgentState) -> dict:
        """Record the run outcome and persist both the graph and vector store.

        Returns
        -------
        dict with keys: sediment_done, final_report
        """
        cap_id = state.get("selected_capability_id", "")
        timestamp = datetime.now().isoformat()

        # 1. Build ValidationRecord
        record = ValidationRecord(
            timestamp=timestamp,
            success=state.get("validation_passed", False),
            metrics=state.get("metrics_result", {}),
            error_message=state.get("execution_error") or None,
            code_version=str(state.get("code_version", 0)),
            notes=(
                f"修复次数:{state.get('code_version', 0)}, "
                f"查询:{state.get('user_query', '')[:50]}"
            ),
        )

        # 2. Update knowledge graph in memory
        if cap_id:
            try:
                self._kg.update_validation_history(cap_id, record)
                logger.debug("Appended validation record to capability %s", cap_id)
            except KeyError as exc:
                logger.warning("SedimentAgent: %s", exc)

        # 3. Persist graph to JSON
        try:
            self._kg.save_to_json()
            logger.info("Knowledge graph saved to JSON")
        except Exception as exc:
            logger.error("SedimentAgent: save_to_json failed: %s", exc)

        # 4. Refresh vector store entry
        if cap_id:
            cap = self._kg.get_capability(cap_id)
            if cap:
                try:
                    self._vs.update_capability(cap)
                    logger.debug("Vector store entry updated for %s", cap_id)
                except Exception as exc:
                    logger.error("SedimentAgent: vs.update_capability failed: %s", exc)

        # 5. Build final report
        report = {
            "query": state.get("user_query", ""),
            "selected_algorithm": cap_id,
            "implementation_plan": state.get("implementation_plan", ""),
            "code_version": state.get("code_version", 0),
            "validation_passed": state.get("validation_passed", False),
            "metrics": state.get("metrics_result", {}),
            "repair_attempts": state.get("code_version", 0),
            "timestamp": timestamp,
        }

        return {"sediment_done": True, "final_report": report}
