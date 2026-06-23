"""Code generation Agent: turns an implementation plan into a complete, runnable Python script."""

import logging
import re
from typing import TYPE_CHECKING

from agents.llm_client import chat, create_client
from agents.state import AgentState

if TYPE_CHECKING:
    from knowledge.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_MAX_TOKENS = 2048

_DATA_PATH = "data/churn_sample.csv"
_TARGET_COL = "Churn"

_SYSTEM = (
    "你是一名专业的Python机器学习工程师。"
    "请生成完整、可直接运行的Python脚本，不要任何说明文字，只输出```python ... ```代码块。"
)

_PROMPT_TMPL = """请根据以下信息生成完整的Python机器学习脚本：

实施方案：
{plan}

参考代码模板（可修改）：
```python
{template}
```

要求：
1. 数据路径固定为：{data_path}
2. 目标列固定为：{target_col}
3. 脚本必须能独立运行，不依赖外部变量
4. 脚本最后一行必须输出JSON格式的指标（不要有其他内容在这行之后）：
   import json; print(json.dumps({{"accuracy": float, "auc": float, "f1": float}}))
5. 只输出```python ... ```代码块，不要任何解释

"""


def _metrics_print_suffix() -> str:
    """JSON metrics line referencing the `results` dict defined in every template."""
    return (
        "\nimport json as _json\n"
        "print(_json.dumps({\n"
        "    'accuracy': float(results.get('accuracy', 0)),\n"
        "    'auc': float(results.get('roc_auc', results.get('auc', 0))),\n"
        "    'f1': float(results.get('f1', 0)),\n"
        "}))\n"
    )


def _template_to_runnable(template: str) -> str:
    """Convert a raw code template into a runnable script with proper JSON output.

    Replaces the template's ``print(results)`` / ``print(classification_report(...))``
    block with a single JSON metrics line so the validator can parse the output.
    """
    # Strip trailing whitespace-only lines, then append the JSON line
    code = template.rstrip()
    # Replace the two-print block common to all templates
    for old_block in (
        "print(results)\nprint(classification_report(y_test, y_pred))",
        "print(results)\nprint('Top features:')\nprint(feature_importance.head(10))",
        "print(results)\n",
        "print(results)",
    ):
        if old_block in code:
            code = code.replace(old_block, "", 1).rstrip()
            break
    code += _metrics_print_suffix()
    return code


def _extract_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


class CodeGenAgent:
    """Uses Claude to generate a complete Python script from a plan and code template."""

    def __init__(self, kg: "KnowledgeGraph") -> None:
        self._client = create_client()
        self._kg = kg

    def run(self, state: AgentState) -> dict:
        """Generate executable Python code for the selected capability.

        Returns
        -------
        dict with keys: generated_code, code_version
        """
        cap_id = state.get("selected_capability_id", "")
        cap = self._kg.get_capability(cap_id)
        template = (
            cap.code_template.replace("{data_path}", _DATA_PATH).replace("{target_col}", _TARGET_COL)
            if cap
            else f"# No template found for {cap_id}"
        )

        plan = state.get("implementation_plan", "训练一个分类模型")
        prompt = _PROMPT_TMPL.format(
            plan=plan,
            template=template,
            data_path=_DATA_PATH,
            target_col=_TARGET_COL,
        )

        try:
            raw = chat(self._client, _SYSTEM, prompt, max_tokens=_MAX_TOKENS)
            logger.debug("CodeGenAgent raw response length: %d", len(raw))
            code = _extract_code(raw)

            # Guarantee the metrics JSON print line exists
            if "json.dumps" not in code:
                code += _metrics_print_suffix()

            return {
                "generated_code": code,
                "code_version": state.get("code_version", 0),
            }
        except Exception as exc:
            logger.error("CodeGenAgent LLM call failed: %s", exc)
            fallback = _template_to_runnable(template)
            return {
                "generated_code": fallback,
                "code_version": state.get("code_version", 0),
            }
