"""Planning Agent: selects the best algorithm capability and produces an implementation plan."""

import json
import logging
import re
from typing import TYPE_CHECKING, List

from agents.llm_client import chat, create_client
from agents.state import AgentState

if TYPE_CHECKING:
    from knowledge.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_MAX_TOKENS = 2048

_SYSTEM = (
    "你是一名资深机器学习工程师。"
    "请根据用户需求和可用算法能力，选择最合适的算法并制定实施方案。"
    "只输出合法JSON，不要任何解释或Markdown。"
)

_PROMPT_TMPL = """用户任务类型：{task_type}
用户约束条件：{constraints}
目标指标：{target_metric}

可用分类算法列表（category=classification，只能从这里选择）：
{capabilities_text}

注意：必须选择 category=classification 的算法作为主算法，预处理步骤会在代码中自动处理。

请选择最合适的分类算法，返回JSON（只输出JSON）：
{{
  "selected_capability_id": "选择的能力id（必须是classification类别）",
  "implementation_plan": "详细实施步骤，1.加载数据 2.预处理 3.训练 4.评估...",
  "reasoning": "选择理由"
}}"""


def _extract_json(text: str) -> dict:
    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(\{.*?\})\s*```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("No JSON object found in LLM response")


class PlanningAgent:
    """Asks Claude to pick the best capability from the retrieved set and draft a plan."""

    def __init__(self, kg: "KnowledgeGraph") -> None:
        self._client = create_client()
        self._kg = kg

    def run(self, state: AgentState) -> dict:
        """Select an algorithm and produce an implementation plan.

        Returns
        -------
        dict with keys: selected_capability_id, implementation_plan, selected_dependencies
        """
        retrieved = state.get("retrieved_capabilities", [])
        if not retrieved:
            logger.warning("PlanningAgent: no retrieved capabilities, using fallback")
            return self._fallback(state)

        # Only expose classification nodes to the LLM — preprocessing is handled in codegen
        clf_candidates = [
            c for c in retrieved if c.get("category") == "classification"
        ]
        if not clf_candidates:
            # Fallback: enrich from graph and re-filter
            clf_candidates = [
                {"id": cap.id, "name": cap.name, "category": cap.category,
                 "description": cap.description}
                for cap in self._kg.get_all_capabilities()
                if cap.category == "classification"
            ][:5]

        capabilities_text = "\n".join(
            f"- id={c['id']} (category={c.get('category','?')}): {c['name']} — {c.get('description', '')}"
            for c in clf_candidates
        )
        prompt = _PROMPT_TMPL.format(
            task_type=state.get("task_type", "classification"),
            constraints="、".join(state.get("constraints", [])) or "无",
            target_metric=state.get("target_metric", "auc"),
            capabilities_text=capabilities_text,
        )

        try:
            raw = chat(self._client, _SYSTEM, prompt, max_tokens=_MAX_TOKENS)
            logger.debug("PlanningAgent raw response: %s", raw[:300])
            parsed = _extract_json(raw)

            cap_id = parsed.get("selected_capability_id", "")
            plan = parsed.get("implementation_plan", "")

            # Resolve dependencies from graph
            deps = self._resolve_deps(cap_id)

            return {
                "selected_capability_id": cap_id,
                "implementation_plan": plan,
                "selected_dependencies": deps,
            }
        except json.JSONDecodeError as exc:
            logger.warning("PlanningAgent JSON parse error: %s", exc)
            return self._fallback(state)
        except Exception as exc:
            logger.error("PlanningAgent LLM call failed: %s", exc)
            return self._fallback(state)

    def _resolve_deps(self, cap_id: str) -> List[str]:
        cap = self._kg.get_capability(cap_id)
        return cap.dependencies if cap else []

    def _fallback(self, state: AgentState) -> dict:
        retrieved = state.get("retrieved_capabilities", [])
        # Prefer classification nodes in fallback
        clf = [c for c in retrieved if c.get("category") == "classification"]
        cap_id = (clf or retrieved or [{}])[0].get("id", "xgboost_churn")
        return {
            "selected_capability_id": cap_id,
            "implementation_plan": "1. 加载数据 2. 预处理 3. 训练模型 4. 评估指标",
            "selected_dependencies": self._resolve_deps(cap_id),
        }
