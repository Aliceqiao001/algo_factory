"""Critic Agent: peer-reviews the planning decision before code is generated."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, List

from agents.llm_client import chat, create_client
from agents.state import AgentState
from knowledge.graph import RELATION_SIMILAR_TO

if TYPE_CHECKING:
    from knowledge.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_MAX_TOKENS = 2048

_SYSTEM = (
    "你是一名严格的机器学习方案评审专家。"
    "请评估给定的算法选择方案是否合理，返回严格JSON，不要任何解释。"
)

_PROMPT_TMPL = """请评审以下机器学习方案，返回JSON评审意见：

## 用户需求
- 任务类型：{task_type}
- 目标指标：{target_metric}
- 约束条件：{constraints}

## 规划方案
- 选定算法：{cap_name}（id: {cap_id}）
- 算法描述：{cap_desc}
- 适用条件：{applicable_conditions}
- 实施方案：{plan}

## 该算法历史验证记录
{history_summary}

## 备选算法（知识图谱SIMILAR_TO关系）
{alternatives}

请综合以上信息，返回如下格式JSON（只输出JSON）：
{{
  "approved": true或false,
  "score": 0到10的整数（10分为完全合适）,
  "concerns": ["担忧点1", "担忧点2"],
  "suggestions": ["改进建议1", "改进建议2"],
  "alternative_capability_id": "备选算法id（仅当approved=false时填写，否则填null）"
}}

评审标准：
- 算法是否符合用户约束（不平衡处理、可解释性等）
- 历史成功率是否可接受（无记录则宽松评审）
- 依赖包是否常见且易获取
"""


def _build_history_summary(validation_history: list) -> str:
    if not validation_history:
        return "暂无历史验证记录（首次使用该算法）"
    total = len(validation_history)
    success = sum(1 for r in validation_history if r.success)
    aucs = [r.metrics.get("auc", r.metrics.get("roc_auc", 0.0))
            for r in validation_history if r.metrics]
    avg_auc = sum(aucs) / len(aucs) if aucs else 0.0
    lines = [
        f"- 历史运行次数：{total}",
        f"- 成功次数：{success}（成功率 {success/total:.0%}）",
        f"- 平均 AUC：{avg_auc:.3f}" if aucs else "- AUC 数据不足",
    ]
    recent = validation_history[-3:]
    for rec in recent:
        status = "✓" if rec.success else "✗"
        m = rec.metrics
        metric_str = ", ".join(f"{k}={v:.3f}" for k, v in m.items() if k in ("accuracy", "auc", "roc_auc", "f1"))
        lines.append(f"  [{status}] {rec.timestamp[:10]}  {metric_str}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(\{.*?\})\s*```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("No JSON found in response")


class CriticAgent:
    """Reviews the planning decision and optionally substitutes a better algorithm.

    If the LLM approves the plan (``approved=true``) the state passes through
    unchanged.  If it rejects the plan and supplies an ``alternative_capability_id``
    that exists in the knowledge graph, the state's ``selected_capability_id`` is
    updated before codegen runs.
    """

    def __init__(self, kg: "KnowledgeGraph") -> None:
        self._client = create_client()
        self._kg = kg

    def run(self, state: AgentState) -> dict:
        """Evaluate the planning decision and return critic feedback.

        Returns
        -------
        dict with keys:
            critic_approved, critic_score, critic_feedback,
            selected_capability_id (may be overwritten)
        """
        cap_id = state.get("selected_capability_id", "")
        cap = self._kg.get_capability(cap_id)

        if cap is None:
            logger.warning("CriticAgent: capability '%s' not found, auto-approving", cap_id)
            return self._approve(cap_id, score=5, feedback="capability未找到，默认通过")

        # Build alternatives list from SIMILAR_TO neighbours
        similar = self._kg.get_neighbors(cap_id, RELATION_SIMILAR_TO)
        alt_lines: List[str] = [
            f"- {c.id}: {c.name} — {c.description[:60]}"
            for c in similar
        ] or ["- 暂无相似算法"]

        history_summary = _build_history_summary(cap.validation_history)

        prompt = _PROMPT_TMPL.format(
            task_type=state.get("task_type", "classification"),
            target_metric=state.get("target_metric", "auc"),
            constraints="、".join(state.get("constraints", [])) or "无",
            cap_name=cap.name,
            cap_id=cap.id,
            cap_desc=cap.description,
            applicable_conditions="、".join(cap.applicable_conditions),
            plan=state.get("implementation_plan", "")[:500],
            history_summary=history_summary,
            alternatives="\n".join(alt_lines),
        )

        try:
            raw = chat(self._client, _SYSTEM, prompt, max_tokens=_MAX_TOKENS)
            logger.debug("CriticAgent raw response: %s", raw[:400])
            parsed = _extract_json(raw)

            approved: bool = bool(parsed.get("approved", True))
            score: int = int(parsed.get("score", 7))
            concerns: List[str] = parsed.get("concerns", [])
            suggestions: List[str] = parsed.get("suggestions", [])
            alternative_id: str = parsed.get("alternative_capability_id") or ""

            feedback_parts = []
            if concerns:
                feedback_parts.append("【担忧】" + "；".join(concerns))
            if suggestions:
                feedback_parts.append("【建议】" + "；".join(suggestions))
            feedback = "  ".join(feedback_parts) or "方案评审通过，无重大问题。"

            # Override capability if critic rejects and supplies a valid alternative
            final_cap_id = cap_id
            if not approved and alternative_id and self._kg.get_capability(alternative_id):
                logger.info(
                    "CriticAgent: plan rejected (score=%d), switching %s → %s",
                    score, cap_id, alternative_id,
                )
                final_cap_id = alternative_id
                # Refresh dependencies for the new capability
                new_cap = self._kg.get_capability(alternative_id)
                feedback += f"  【已切换至备选算法: {new_cap.name}】"

            logger.info("CriticAgent: approved=%s score=%d cap=%s", approved, score, final_cap_id)
            return {
                "critic_approved": approved,
                "critic_score": score,
                "critic_feedback": feedback,
                "selected_capability_id": final_cap_id,
                "selected_dependencies": (
                    self._kg.get_capability(final_cap_id).dependencies
                    if final_cap_id != cap_id else state.get("selected_dependencies", [])
                ),
            }

        except json.JSONDecodeError as exc:
            logger.warning("CriticAgent JSON parse error: %s", exc)
            return self._approve(cap_id, score=7, feedback="评审响应解析失败，默认通过")
        except Exception as exc:
            logger.error("CriticAgent LLM call failed: %s", exc)
            return self._approve(cap_id, score=7, feedback="LLM调用失败，默认通过")

    @staticmethod
    def _approve(cap_id: str, score: int, feedback: str) -> dict:
        return {
            "critic_approved": True,
            "critic_score": score,
            "critic_feedback": feedback,
            "selected_capability_id": cap_id,
        }
