"""LangGraph workflow: wires all agent nodes into a directed pipeline with repair loop."""

import logging

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.codegen import CodeGenAgent
from agents.planning import PlanningAgent
from agents.repair import RepairAgent
from agents.retrieval import RetrievalAgent
from agents.sediment import SedimentAgent
from agents.state import AgentState, create_initial_state
from agents.understanding import UnderstandingAgent
from agents.validator import ValidatorAgent
from executor.sandbox import CodeSandbox
from knowledge.graph import KnowledgeGraph
from knowledge.vector_store import VectorStore

logger = logging.getLogger(__name__)


class AlgoFactoryWorkflow:
    """End-to-end LangGraph pipeline for automated algorithm selection and code generation.

    Graph topology
    --------------
    understand → retrieve → plan → codegen → validate
                                                ↓              ↓
                                           (passed/limit)   (retry)
                                             sediment ←── repair ←┘
                                                ↓
                                               END
    """

    def __init__(self) -> None:
        # Shared knowledge components
        self.kg = KnowledgeGraph()
        self.vs = VectorStore()
        self.vs.build_from_graph(self.kg)
        self.sandbox = CodeSandbox()

        # Agent instances
        self.understanding = UnderstandingAgent()
        self.retrieval = RetrievalAgent(self.kg, self.vs)
        self.planning = PlanningAgent(self.kg)
        self.codegen = CodeGenAgent(self.kg)
        self.validator = ValidatorAgent(self.sandbox)
        self.repair = RepairAgent()
        self.sediment = SedimentAgent(self.kg, self.vs)

        self.app: CompiledStateGraph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)

        # Nodes — each lambda delegates to the corresponding agent
        workflow.add_node("understand", lambda s: self.understanding.run(s))
        workflow.add_node("retrieve",   lambda s: self.retrieval.run(s))
        workflow.add_node("plan",       lambda s: self.planning.run(s))
        workflow.add_node("codegen",    lambda s: self.codegen.run(s))
        workflow.add_node("validate",   lambda s: self.validator.run(s))
        workflow.add_node("repair",     lambda s: self.repair.run(s))
        workflow.add_node("sediment",   lambda s: self.sediment.run(s))

        # Linear path
        workflow.set_entry_point("understand")
        workflow.add_edge("understand", "retrieve")
        workflow.add_edge("retrieve",   "plan")
        workflow.add_edge("plan",       "codegen")
        workflow.add_edge("codegen",    "validate")

        # Conditional branch after validation
        workflow.add_conditional_edges(
            "validate",
            _route_after_validate,
            {"sediment": "sediment", "repair": "repair"},
        )

        # Repair loops back to validate; sediment terminates
        workflow.add_edge("repair",   "validate")
        workflow.add_edge("sediment", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, user_query: str) -> AgentState:
        """Execute the full pipeline synchronously and return the final state.

        Parameters
        ----------
        user_query:
            Free-text description of the machine-learning task.
        """
        initial = create_initial_state(user_query)
        logger.info("Starting workflow for query: %s", user_query[:80])
        final_state = self.app.invoke(initial)
        logger.info("Workflow completed. validation_passed=%s", final_state.get("validation_passed"))
        return final_state

    def run_with_stream(self, user_query: str):
        """Execute the pipeline and yield each node's state-update dict as it completes.

        Yields
        ------
        dict
            ``{node_name: {updated_fields}}`` — the same format LangGraph uses
            for streaming graph output.
        """
        initial = create_initial_state(user_query)
        logger.info("Streaming workflow for query: %s", user_query[:80])
        for chunk in self.app.stream(initial):
            yield chunk


# ------------------------------------------------------------------
# Routing function (module-level so it is picklable)
# ------------------------------------------------------------------

def _route_after_validate(state: AgentState) -> str:
    """Decide next node after the validate step."""
    if state["validation_passed"]:
        return "sediment"
    if state["code_version"] >= state["max_repair_attempts"]:
        # Exceeded repair budget — sediment the failure for future learning
        return "sediment"
    return "repair"
