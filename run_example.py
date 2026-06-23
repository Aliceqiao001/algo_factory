"""End-to-end pipeline demo: generates data (if needed) then runs the full workflow."""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure algo_factory root is on sys.path when run directly
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.WARNING,           # suppress library noise
    format="%(levelname)s %(name)s: %(message)s",
)
logging.getLogger("agents").setLevel(logging.INFO)

from agents.workflow import AlgoFactoryWorkflow
from data.generate_data import save_datasets

_DATA_FILE = _ROOT / "data" / "churn_sample.csv"


def _ensure_data() -> None:
    if not _DATA_FILE.exists():
        print("Generating sample data …")
        save_datasets(output_dir=_ROOT / "data")
    else:
        print(f"Using existing data: {_DATA_FILE}")


if __name__ == "__main__":
    _ensure_data()

    query = (
        "我需要一个客户流失预测模型，数据存在data/churn_sample.csv，"
        "目标列是Churn，要求AUC>0.75，需要处理类别不平衡问题"
    )

    print("=" * 55)
    print(f"用户需求: {query}")
    print("=" * 55)

    workflow = AlgoFactoryWorkflow()

    for chunk in workflow.run_with_stream(query):
        node_name = list(chunk.keys())[0]
        state_update = chunk[node_name]
        print(f"\n[{node_name}] 完成")

        if node_name == "understand":
            print(f"  任务类型:   {state_update.get('task_type')}")
            print(f"  目标指标:   {state_update.get('target_metric')}")
            print(f"  约束条件:   {state_update.get('constraints')}")
        elif node_name == "retrieve":
            caps = state_update.get("retrieved_capabilities", [])
            print(f"  检索到:     {[c['id'] for c in caps]}")
        elif node_name == "plan":
            print(f"  选定算法:   {state_update.get('selected_capability_id')}")
            print(f"  依赖包:     {state_update.get('selected_dependencies')}")
        elif node_name == "codegen":
            code = state_update.get("generated_code", "")
            print(f"  代码行数:   {len(code.splitlines())}")
        elif node_name == "validate":
            print(f"  执行成功:   {state_update.get('execution_success')}")
            print(f"  验证通过:   {state_update.get('validation_passed')}")
            print(f"  指标结果:   {state_update.get('metrics_result')}")
        elif node_name == "repair":
            print(f"  修复轮次:   {state_update.get('code_version')}")
            history = state_update.get("repair_history", [])
            if history:
                print(f"  最近错误:   {history[-1].get('error', '')[:80]}")
        elif node_name == "sediment":
            report = state_update.get("final_report", {})
            print(f"  沉淀完成")
            print(f"  选定算法:   {report.get('selected_algorithm')}")
            print(f"  验证通过:   {report.get('validation_passed')}")
            print(f"  最终指标:   {report.get('metrics')}")
            print(f"  修复次数:   {report.get('repair_attempts')}")

    print("\n" + "=" * 55)
    print("流水线完成")
