"""Repair Agent: diagnoses execution failures or low metrics and patches the generated code."""

import logging
import re
from datetime import datetime
from typing import List

from agents.llm_client import chat, create_client
from agents.state import AgentState

logger = logging.getLogger(__name__)

_MAX_TOKENS = 2048

_SYSTEM = (
    "你是一名Python调试专家。"
    "请修复给定代码的问题，只返回修复后的完整Python代码（```python ... ```），不要任何解释。"
)

_PROMPT_ERROR_TMPL = """以下Python代码执行时出错，请修复：

错误信息：
{error}

原始代码：
```python
{code}
```

要求：
- 修复所有错误
- 保持原有逻辑和数据路径
- 脚本最后一行必须输出：import json; print(json.dumps({{"accuracy": float, "auc": float, "f1": float}}))
- 只返回```python ... ```代码块
"""

_PROMPT_METRIC_TMPL = """以下Python代码执行成功但指标不达标，请优化：

当前指标：{metrics}
目标：accuracy > 0.6, auc > 0.65

原始代码：
```python
{code}
```

优化方向：考虑调整超参数、处理类别不平衡、增加特征工程
只返回```python ... ```代码块
"""


def _extract_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


class RepairAgent:
    """Asks Claude to fix code that either crashed or failed metric thresholds."""

    def __init__(self) -> None:
        self._client = create_client()

    def run(self, state: AgentState) -> dict:
        """Attempt one repair iteration.

        Returns the original state slice unchanged (with validation_passed=False)
        if the repair attempt ceiling has been reached.

        Returns
        -------
        dict with keys: generated_code, code_version, repair_history
        """
        version = state.get("code_version", 0)
        max_attempts = state.get("max_repair_attempts", 3)

        if version >= max_attempts:
            logger.info("RepairAgent: reached max repair attempts (%d), giving up", max_attempts)
            return {
                "generated_code": state.get("generated_code", ""),
                "code_version": version,
                "repair_history": state.get("repair_history", []),
                "validation_passed": False,
            }

        original_code = state.get("generated_code", "")
        error = state.get("execution_error", "")
        metrics = state.get("metrics_result", {})
        exec_success = state.get("execution_success", False)

        # Build prompt depending on failure mode
        if not exec_success or error:
            prompt = _PROMPT_ERROR_TMPL.format(error=error[:1000], code=original_code)
            failure_summary = f"RuntimeError: {error[:200]}"
        else:
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            prompt = _PROMPT_METRIC_TMPL.format(metrics=metrics_str, code=original_code)
            failure_summary = f"MetricsBelowThreshold: {metrics_str}"

        try:
            raw = chat(self._client, _SYSTEM, prompt, max_tokens=_MAX_TOKENS)
            fixed_code = _extract_code(raw)
            logger.info("RepairAgent produced fix for attempt %d", version + 1)
        except Exception as exc:
            logger.error("RepairAgent LLM call failed: %s", exc)
            fixed_code = original_code  # keep original; outer loop will give up next round

        history: List[dict] = list(state.get("repair_history", []))
        history.append(
            {
                "attempt": version + 1,
                "timestamp": datetime.now().isoformat(),
                "error": failure_summary,
                "fix": "LLM-generated patch" if fixed_code != original_code else "no change",
            }
        )

        return {
            "generated_code": fixed_code,
            "code_version": version + 1,
            "repair_history": history,
        }
