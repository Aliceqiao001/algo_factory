"""Requirement understanding Agent: parses a free-text user query into a structured task spec."""

import json
import logging
import re
from typing import Any, Dict

from agents.llm_client import chat, create_client
from agents.state import AgentState

logger = logging.getLogger(__name__)

_MAX_TOKENS = 2048

_SYSTEM = (
    "你是一名机器学习需求分析专家。"
    "请将用户的自然语言描述解析为结构化的任务规格，只输出合法JSON，不要任何解释。"
)

_PROMPT_TMPL = """请分析以下机器学习需求，返回严格的JSON格式：

用户需求：{query}

返回格式（只输出JSON，不要Markdown包裹）：
{{
  "task_type": "classification 或 regression 或 clustering",
  "target_metric": "用户最关心的评估指标，如auc/f1/accuracy/rmse",
  "data_requirements": {{
    "target_column": "目标列名称",
    "feature_types": ["numerical", "categorical"],
    "expected_rows": "估计数据量，如>1000"
  }},
  "constraints": ["约束条件1", "约束条件2"]
}}"""

_DEFAULT_RESULT: Dict[str, Any] = {
    "task_type": "classification",
    "target_metric": "auc",
    "data_requirements": {
        "target_column": "churn",
        "feature_types": ["numerical", "categorical"],
        "expected_rows": ">1000",
    },
    "constraints": [],
}


def _extract_json(text: str) -> dict:
    for pattern in (
        r"```json\s*(.*?)\s*```",
        r"```\s*(\{.*?\})\s*```",
    ):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("No JSON object found in LLM response")


class UnderstandingAgent:
    """Calls Claude to convert a user query into a structured AgentState update."""

    def __init__(self) -> None:
        self._client = create_client()

    def run(self, state: AgentState) -> dict:
        """Parse ``state["user_query"]`` and return structured task fields.

        Returns
        -------
        dict with keys: task_type, target_metric, data_requirements, constraints
        """
        query = state["user_query"]
        prompt = _PROMPT_TMPL.format(query=query)
        try:
            raw = chat(self._client, _SYSTEM, prompt, max_tokens=_MAX_TOKENS)
            logger.debug("UnderstandingAgent raw response: %s", raw[:300])
            parsed = _extract_json(raw)
            return {
                "task_type": parsed.get("task_type", _DEFAULT_RESULT["task_type"]),
                "target_metric": parsed.get("target_metric", _DEFAULT_RESULT["target_metric"]),
                "data_requirements": parsed.get("data_requirements", _DEFAULT_RESULT["data_requirements"]),
                "constraints": parsed.get("constraints", []),
            }
        except json.JSONDecodeError as exc:
            logger.warning("UnderstandingAgent JSON parse error: %s", exc)
            return dict(_DEFAULT_RESULT)
        except Exception as exc:
            logger.error("UnderstandingAgent LLM call failed: %s", exc)
            return dict(_DEFAULT_RESULT)
