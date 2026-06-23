"""Streamlit main interface for the Algo Factory pipeline."""

import sys
from pathlib import Path

# Ensure project root is on sys.path when launched via `streamlit run ui/app.py`
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from knowledge.graph import KnowledgeGraph
from knowledge.vector_store import VectorStore
from agents.workflow import AlgoFactoryWorkflow

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="算法能力工厂",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
_EXAMPLES = [
    "客户流失预测，要求AUC>0.75，处理不平衡数据",
    "用随机森林预测客户流失",
    "用逻辑回归做分类，需要特征选择",
]

_NODE_LABELS = {
    "understand": "🧠 需求理解",
    "retrieve":   "🔍 知识检索",
    "plan":       "📋 方案规划",
    "codegen":    "⚙️  代码生成",
    "validate":   "✅ 验证执行",
    "repair":     "🔧 自动修复",
    "sediment":   "💾 知识沉淀",
}

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="正在加载知识图谱…")
def load_kg():
    return KnowledgeGraph()

@st.cache_resource(show_spinner="正在初始化向量检索…")
def load_workflow():
    return AlgoFactoryWorkflow()

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(kg: KnowledgeGraph):
    with st.sidebar:
        st.title("📚 知识图谱")
        caps = kg.get_all_capabilities()
        summary = kg.summary()

        col1, col2 = st.columns(2)
        col1.metric("能力节点", summary["n_capabilities"])
        col2.metric("关系边", summary["n_edges"])

        st.divider()
        st.subheader("已注册算法能力")

        category_colors = {
            "classification":    "🔵",
            "preprocessing":     "🟠",
            "feature_engineering": "🟢",
        }
        for cap in caps:
            icon = category_colors.get(cap.category, "⚪")
            with st.expander(f"{icon} {cap.name}", expanded=False):
                st.caption(f"ID: `{cap.id}`")
                st.caption(f"类别: {cap.category}")
                st.write(cap.description)
                if cap.applicable_conditions:
                    st.markdown("**适用条件:**")
                    for c in cap.applicable_conditions:
                        st.markdown(f"- {c}")
                if cap.validation_history:
                    last = cap.validation_history[-1]
                    status = "✅ 通过" if last.success else "❌ 失败"
                    st.caption(f"最近验证: {status} @ {last.timestamp[:10]}")

        st.divider()
        st.caption("Algo Factory v1.0 · 硅基流动驱动")

# ── Main area ─────────────────────────────────────────────────────────────────
def render_main(workflow: AlgoFactoryWorkflow):
    st.title("🏭 算法能力工厂")
    st.markdown("**输入需求描述，AI 自动完成算法选择 → 代码生成 → 验证 → 知识沉淀**")

    # Example buttons
    st.markdown("**快速示例：**")
    cols = st.columns(len(_EXAMPLES))
    for i, (col, example) in enumerate(zip(cols, _EXAMPLES)):
        if col.button(f"示例 {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state["query"] = example

    # Query input
    query = st.text_area(
        "需求描述",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="例如：我需要一个客户流失预测模型，要求AUC>0.75，需要处理类别不平衡问题",
        key="query_input",
    )

    run_btn = st.button("🚀 运行流水线", type="primary", use_container_width=True)

    if run_btn and query.strip():
        st.session_state["query"] = query.strip()
        run_pipeline(workflow, query.strip())
    elif run_btn:
        st.warning("请输入需求描述")

# ── Pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline(workflow: AlgoFactoryWorkflow, query: str):
    st.divider()
    st.subheader("⏳ 流水线执行")

    node_outputs: dict = {}
    final_state: dict = {}

    progress = st.progress(0, text="初始化…")
    node_order = list(_NODE_LABELS.keys())
    n_nodes = len(node_order)

    status_area = st.container()

    with status_area:
        for chunk in workflow.run_with_stream(query):
            node_name = list(chunk.keys())[0]
            state_update = chunk[node_name]
            node_outputs[node_name] = state_update
            final_state.update(state_update)

            step = node_order.index(node_name) + 1 if node_name in node_order else n_nodes
            progress.progress(step / n_nodes, text=f"{_NODE_LABELS.get(node_name, node_name)} 完成")

            label = _NODE_LABELS.get(node_name, node_name)
            with st.expander(f"{label}", expanded=(node_name in ("validate", "sediment"))):
                _render_node_output(node_name, state_update)

    progress.progress(1.0, text="✅ 流水线完成")
    st.divider()
    render_results(final_state, node_outputs)


def _render_node_output(node_name: str, state: dict):
    if node_name == "understand":
        col1, col2 = st.columns(2)
        col1.write(f"**任务类型:** {state.get('task_type', '-')}")
        col1.write(f"**目标指标:** {state.get('target_metric', '-')}")
        col2.write(f"**约束条件:**")
        for c in state.get("constraints", []):
            col2.markdown(f"- {c}")

    elif node_name == "retrieve":
        caps = state.get("retrieved_capabilities", [])
        for c in caps:
            st.markdown(f"- `{c['id']}` — {c.get('name', '')} (score: {c.get('score', 0):.3f})")

    elif node_name == "plan":
        st.write(f"**选定算法:** `{state.get('selected_capability_id', '-')}`")
        st.write(f"**实施方案:**")
        st.info(state.get("implementation_plan", "-"))
        deps = state.get("selected_dependencies", [])
        if deps:
            st.write(f"**依赖包:** {', '.join(deps)}")

    elif node_name == "codegen":
        code = state.get("generated_code", "")
        st.write(f"**代码行数:** {len(code.splitlines())}")
        with st.expander("查看代码", expanded=False):
            st.code(code, language="python")

    elif node_name == "validate":
        passed = state.get("validation_passed", False)
        success = state.get("execution_success", False)
        st.write(f"**执行状态:** {'✅ 成功' if success else '❌ 失败'}")
        st.write(f"**验证结果:** {'✅ 通过' if passed else '❌ 未通过'}")
        metrics = state.get("metrics_result", {})
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                col.metric(k.upper(), f"{v:.3f}")
        if not success:
            err = state.get("execution_error", "")
            if err:
                st.error(err[:400])

    elif node_name == "repair":
        st.write(f"**修复轮次:** {state.get('code_version', 0)}")
        history = state.get("repair_history", [])
        if history:
            last = history[-1]
            st.write(f"**错误摘要:** {last.get('error', '')[:100]}")
            st.write(f"**修复操作:** {last.get('fix', '-')}")

    elif node_name == "sediment":
        report = state.get("final_report", {})
        st.success("知识库已更新")
        if report:
            st.json(report)


def render_results(final_state: dict, node_outputs: dict):
    st.subheader("📊 最终结果")

    passed = final_state.get("validation_passed", False)
    if passed:
        st.success("🎉 验证通过！模型已成功生成并写入知识库。")
    else:
        st.error("⚠️ 验证未通过，已记录失败经验到知识库。")

    # Metrics row
    metrics = final_state.get("metrics_result", {})
    if metrics:
        st.markdown("**核心指标**")
        cols = st.columns(max(len(metrics), 3))
        for col, (k, v) in zip(cols, metrics.items()):
            delta = "✓" if (k in ("accuracy", "auc", "f1") and v > 0.65) else None
            col.metric(label=k.upper(), value=f"{v:.4f}", delta=delta)

    # Generated code
    code = final_state.get("generated_code", "")
    if code:
        st.markdown("**生成的代码**")
        st.code(code, language="python")

    # Final report
    report = final_state.get("final_report", {})
    if report:
        st.markdown("**验证报告**")
        col1, col2, col3 = st.columns(3)
        col1.write(f"**选定算法:** `{report.get('selected_algorithm', '-')}`")
        col2.write(f"**修复次数:** {report.get('repair_attempts', 0)}")
        col3.write(f"**时间戳:** {report.get('timestamp', '-')[:19]}")
        with st.expander("查看完整报告 JSON"):
            st.json(report)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    kg = load_kg()
    workflow = load_workflow()
    render_sidebar(kg)
    render_main(workflow)


if __name__ == "__main__":
    main()
