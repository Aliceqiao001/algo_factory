"""LangGraph shared state: TypedDict container passed between every agent node."""

from typing import Any, Dict, List, Optional, TypedDict

from knowledge.schema import AlgorithmCapability, ValidationRecord


class AgentState(TypedDict):
    # --- user input ---
    user_query: str                     # original natural-language request

    # --- understanding node outputs ---
    task_type: str                      # classification / regression / clustering
    target_metric: str                  # primary evaluation metric requested
    data_requirements: Dict[str, Any]   # parsed data constraints (columns, min_rows …)
    constraints: List[str]              # free-text constraint conditions

    # --- retrieval node outputs ---
    retrieved_capabilities: List[Dict]  # ranked hits from VectorStore.search()
    selected_capability_id: str         # algorithm id chosen by the planning node

    # --- planning node outputs ---
    implementation_plan: str            # natural-language step-by-step plan
    selected_dependencies: List[str]    # pip packages required by the plan

    # --- code generation outputs ---
    generated_code: str                 # full Python source produced by codegen
    code_version: int                   # increments on each repair attempt (starts 0)

    # --- execution / validation outputs ---
    execution_success: bool             # True if subprocess exited with code 0
    execution_output: str               # captured stdout from sandbox run
    execution_error: str                # captured stderr from sandbox run
    metrics_result: Dict[str, float]    # parsed metric scores {accuracy: 0.85, …}
    validation_passed: bool             # True if all metrics exceed thresholds

    # --- repair control ---
    max_repair_attempts: int            # ceiling on repair loops (default 3)
    repair_history: List[Dict]          # [{attempt, error, fix}, …] across all loops

    # --- sediment / output ---
    sediment_done: bool                 # True once capability written back to graph
    final_report: Dict[str, Any]        # structured summary returned to the UI


def create_initial_state(user_query: str) -> AgentState:
    """Return an AgentState with sensible defaults for every field.

    Only ``user_query`` carries real content; everything else is a zero-value
    so downstream nodes can safely read any key without a KeyError.
    """
    return AgentState(
        user_query=user_query,
        task_type="",
        target_metric="",
        data_requirements={},
        constraints=[],
        retrieved_capabilities=[],
        selected_capability_id="",
        implementation_plan="",
        selected_dependencies=[],
        generated_code="",
        code_version=0,
        execution_success=False,
        execution_output="",
        execution_error="",
        metrics_result={},
        validation_passed=False,
        max_repair_attempts=3,
        repair_history=[],
        sediment_done=False,
        final_report={},
    )
