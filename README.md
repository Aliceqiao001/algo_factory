# 🏭 Algo Factory — AI驱动的算法能力工厂

## 1. 项目背景和目标

### 背景

在数据科学和机器学习项目中，从需求分析到模型上线是一个高度重复且耗时的过程。数据科学家需要反复经历「理解需求 → 选择算法 → 编写代码 → 调试验证」的循环，而大量经验和模式难以系统沉淀复用。

### 目标

**Algo Factory** 是一个基于大语言模型（LLM）和知识图谱的算法自动生成系统，旨在实现：

- **需求理解**：将自然语言描述的业务需求自动解析为结构化任务规格
- **智能检索**：从预构建的算法能力知识库中语义匹配最合适的方案
- **代码生成**：基于模板和 LLM 自动生成可运行的 Python 机器学习脚本
- **自动验证**：在沙箱环境中执行代码并评估指标是否达标
- **自动修复**：验证失败时 LLM 自动诊断错误并修复代码
- **知识沉淀**：将成功/失败经验写回知识图谱，持续积累能力

---

## 2. 系统架构和模块设计

### ASCII 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户界面 (Streamlit)                          │
│                    streamlit run ui/app.py                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │ user_query
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph 工作流 (AlgoFactoryWorkflow)            │
│                                                                      │
│  ┌───────────┐   ┌──────────┐   ┌────────┐   ┌─────────┐           │
│  │understand │──▶│ retrieve │──▶│  plan  │──▶│ codegen │           │
│  └───────────┘   └──────────┘   └────────┘   └────┬────┘           │
│        ▲                                           │                 │
│        │          ┌──────────┐                     ▼                 │
│        │          │ sediment │◀──(pass/limit)──┌──────────┐         │
│        │          └──────────┘                 │ validate │         │
│        │               ▲                       └────┬─────┘         │
│        │               │                           │(fail)           │
│        │               │                           ▼                 │
│        │               └──────────────────────┌────────┐            │
│        │                                      │ repair │            │
│        │                                      └────────┘            │
└────────┼─────────────────────────────────────────────────────────── ┘
         │                         │
         ▼                         ▼
┌─────────────────┐     ┌──────────────────────┐
│  知识图谱层      │     │    执行器层            │
│                 │     │                       │
│ KnowledgeGraph  │     │  CodeSandbox          │
│ (NetworkX)      │     │  (subprocess)         │
│                 │     │                       │
│ VectorStore     │     │  MetricsEvaluator     │
│ (ChromaDB)      │     │                       │
└─────────────────┘     └──────────────────────┘
```

### 模块说明

| 模块 | 路径 | 职责 |
|------|------|------|
| 知识图谱 | `knowledge/` | NetworkX 有向图 + ChromaDB 向量检索，存储算法能力节点 |
| Agent 层 | `agents/` | 7 个 LangGraph 节点，每个节点调用 LLM 或确定性逻辑 |
| 执行器 | `executor/` | subprocess 沙箱执行生成代码，解析 stdout 中的 JSON 指标 |
| 界面 | `ui/` | Streamlit Web 界面，流式展示每个节点进度 |
| 数据 | `data/` | 合成客户流失数据集 (1000 行) |

---

## 3. 能力知识图谱 Schema 和示例

### Schema 定义

```python
@dataclass
class AlgorithmCapability:
    id: str                              # 全局唯一标识
    name: str                            # 中文名称
    category: str                        # classification / preprocessing / feature_engineering
    description: str                     # 算法描述
    applicable_conditions: List[str]     # 适用条件（用于条件检索）
    input_schema: InputSchema            # 输入数据要求
    output_schema: OutputSchema          # 输出格式
    metrics: List[str]                   # 评估指标列表
    dependencies: List[str]              # pip 依赖包
    code_template: str                   # 可运行代码模板（含占位符）
    hyperparameters: List[HyperParameter]
    validation_history: List[ValidationRecord]  # 历史验证记录（自动积累）
```

### 已注册的 6 个能力节点

| ID | 名称 | 类别 |
|----|------|------|
| `logistic_regression_churn` | 逻辑回归流失预测 | classification |
| `random_forest_churn` | 随机森林流失预测 | classification |
| `xgboost_churn` | XGBoost 流失预测 | classification |
| `smote_oversampling` | SMOTE 过采样 | preprocessing |
| `standard_scaler_preprocessing` | 标准化预处理 | preprocessing |
| `feature_selection_rfe` | RFE 递归特征消除 | feature_engineering |

### 知识图谱语义边

```
smote_oversampling      ──REQUIRES──▶  logistic_regression_churn
smote_oversampling      ──REQUIRES──▶  random_forest_churn
smote_oversampling      ──REQUIRES──▶  xgboost_churn
standard_scaler         ──REQUIRES──▶  logistic_regression_churn
xgboost_churn           ──SIMILAR_TO─▶ random_forest_churn
feature_selection_rfe   ──REQUIRES──▶  xgboost_churn
feature_selection_rfe   ──REQUIRES──▶  random_forest_churn
```

---

## 4. Agent 工作流设计

### 各节点职责

| 节点 | Agent 类 | 是否调用 LLM | 核心逻辑 |
|------|----------|-------------|---------|
| understand | `UnderstandingAgent` | ✅ | 解析用户需求为 JSON 结构 |
| retrieve | `RetrievalAgent` | ❌ | 向量语义检索 + SMOTE 注入规则 |
| plan | `PlanningAgent` | ✅ | 选择最优算法，生成实施方案 |
| codegen | `CodeGenAgent` | ✅ | 基于模板和方案生成完整 Python 脚本 |
| validate | `ValidatorAgent` | ❌ | subprocess 沙箱执行，解析指标 |
| repair | `RepairAgent` | ✅ | 诊断错误/低指标，LLM 修复代码 |
| sediment | `SedimentAgent` | ❌ | 写回验证记录，持久化图谱和向量库 |

### 条件路由逻辑

```python
def route_after_validate(state):
    if state["validation_passed"]:
        return "sediment"          # 验证通过 → 沉淀
    if state["code_version"] >= state["max_repair_attempts"]:
        return "sediment"          # 超过修复上限 → 沉淀失败经验
    return "repair"                # 否则 → 继续修复
```

### LLM 适配器

所有 LLM 调用通过 `agents/llm_client.py` 统一路由，支持 OpenAI 兼容接口（默认配置为硅基流动）：

```python
# agents/llm_client.py
def chat(client, system, user, max_tokens=2048) -> str:
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
    )
    return resp.choices[0].message.content
```

---

## 5. 环境配置和运行方法

### 环境要求

- Python 3.10+
- Windows / macOS / Linux

### 安装步骤

```bash
# 1. 克隆项目
git clone <repo-url>
cd algo_factory

# 2. 安装依赖（国内镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install xgboost imbalanced-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 API Key 和模型配置：
# ANTHROPIC_API_KEY=sk-xxxxxxxxxx
# LLM_BASE_URL=https://api.siliconflow.cn/v1        # 硅基流动
# LLM_MODEL=Qwen/Qwen2.5-72B-Instruct

# 4. 生成示例数据
python data/generate_data.py

# 5. 运行命令行示例
python run_example.py

# 6. 启动 Web 界面
streamlit run ui/app.py
```

### 支持的 LLM 提供商

| 提供商 | LLM_BASE_URL | 推荐模型 |
|--------|-------------|---------|
| 硅基流动 | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-72B-Instruct` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| Anthropic | *(使用 anthropic SDK)* | `claude-sonnet-4-6` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5:72b` |

---

## 6. 示例数据和测试任务说明

### 数据集说明

`data/churn_sample.csv` 是程序自动生成的合成客户流失数据集：

| 列名 | 类型 | 说明 |
|------|------|------|
| CustomerID | string | 客户唯一标识（训练时自动过滤） |
| Age | int | 年龄 25–70 |
| Tenure | int | 在网月数 0–60 |
| MonthlyCharges | float | 月消费金额 20–100 |
| TotalCharges | float | 总消费金额 |
| NumProducts | int | 购买产品数 1–4 |
| HasCreditCard | int | 是否有信用卡 |
| IsActiveMember | int | 是否活跃用户 |
| EstimatedSalary | float | 预估年薪 |
| Geography | string | 地区（North/South/East/West） |
| Gender | string | 性别 |
| **Churn** | int | **目标列**：1=流失，0=留存（约 25% 流失率） |

**Churn 生成规则：**
- 基础流失率：5%
- Tenure<12 且 NumProducts==1：+60%（最强信号）
- IsActiveMember==0：+35%
- MonthlyCharges>70：+20%
- 不活跃 + 高消费交叉项：额外+20%

### 推荐测试查询

```
1. "客户流失预测，要求AUC>0.75，处理不平衡数据"
2. "用随机森林预测客户流失"
3. "用逻辑回归做分类，需要特征选择"
4. "对流失数据做SMOTE过采样后训练XGBoost"
```

### 运行测试

```bash
# 在 algo_factory/ 目录下
pytest tests/ -v
```

---

## 7. 生成算法代码示例

以下是系统对「XGBoost 流失预测」查询自动生成的完整代码示例：

```python
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

df = pd.read_csv('data/churn_sample.csv')
target_col = 'Churn'

X = df.drop(columns=[target_col])
y = df[target_col]

# 过滤高基数 ID 列，避免无意义特征爆炸
high_card = [c for c in X.columns if X[c].dtype == object and X[c].nunique() > len(X) * 0.5]
X = X.drop(columns=high_card, errors='ignore')
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
)
model.fit(X_train, y_train, verbose=False)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

results = {
    'accuracy': accuracy_score(y_test, y_pred),
    'roc_auc':  roc_auc_score(y_test, y_prob),
    'f1':       f1_score(y_test, y_pred),
}

import json
print(json.dumps({
    'accuracy': float(results['accuracy']),
    'auc':      float(results['roc_auc']),
    'f1':       float(results['f1']),
}))
```

**实际运行输出：**
```json
{"accuracy": 0.785, "auc": 0.839, "f1": 0.695}
```

---

## 8. 验证结果和报告样例

### 验证指标阈值

| 指标 | 阈值 | 说明 |
|------|------|------|
| accuracy | > 0.60 | 整体准确率 |
| auc | > 0.65 | ROC-AUC，对不平衡数据更可靠 |

### 实际运行结果（XGBoost，1000 条合成数据）

| 指标 | 值 | 是否达标 |
|------|-----|---------|
| Accuracy | 0.785 | ✅ |
| AUC | 0.839 | ✅ |
| F1 | 0.695 | — |

### 完整验证报告样例

```json
{
  "query": "客户流失预测，要求AUC>0.75，处理不平衡数据",
  "selected_algorithm": "xgboost_churn",
  "implementation_plan": "1.加载数据 2.预处理 3.训练XGBoost 4.评估指标",
  "validation_passed": true,
  "metrics": {"accuracy": 0.785, "auc": 0.839, "f1": 0.695},
  "repair_attempts": 1,
  "timestamp": "2026-06-23T10:42:51",
  "knowledge_sediment": {
    "capability_id": "xgboost_churn",
    "validation_record_appended": true
  }
}
```

---

## 9. 遇到的挑战和解决方案

### 挑战 1：ChromaDB 默认 Embedding 需要联网下载模型

**现象：** ChromaDB 1.5+ 首次使用时尝试从 HuggingFace 下载 ONNX 模型，在受限网络环境下超时报错。

**解决方案：** 实现 `_NgramHashEmbedding` 自定义 Embedding Function，使用字符 n-gram（1-3 gram）+ MD5 哈希映射到 256 维密向量，完全离线，无需任何下载。

```python
class _NgramHashEmbedding(EmbeddingFunction[Documents]):
    def _embed_one(self, text: str) -> List[float]:
        vec = np.zeros(256, dtype=np.float32)
        for n in (1, 2, 3):
            for i in range(len(text) - n + 1):
                gram = text[i:i+n]
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                vec[h % 256] += 1.0
        return vec.tolist()
```

### 挑战 2：高基数字符串列（CustomerID）污染 One-Hot 编码

**现象：** `CustomerID` 有 1000 个唯一值，进入 `pd.get_dummies` 后产生近 1000 个哑变量，导致模型 AUC 从 0.83 骤降至 0.58。

**解决方案：** 在所有代码模板中加入高基数列过滤：

```python
high_card = [c for c in X.columns if X[c].dtype == object and X[c].nunique() > len(X) * 0.5]
X = X.drop(columns=high_card, errors='ignore')
```

### 挑战 3：LangGraph 与 Anthropic SDK 的 API 不兼容

**现象：** 项目最初使用 Anthropic SDK，但用户使用硅基流动（OpenAI 兼容接口），两者 API 格式不同。

**解决方案：** 新建 `agents/llm_client.py` 统一适配层，通过环境变量 `LLM_BASE_URL` 和 `LLM_MODEL` 切换提供商，4 个 LLM Agent 无需关心底层实现。

### 挑战 4：XGBoost `scale_pos_weight` 反向降低 AUC

**现象：** 设置 `scale_pos_weight=3.0` 让模型过度追求少数类，AUC 从 0.60 下降至 0.58。

**解决方案：** 移除 `scale_pos_weight`，改用较小的 `max_depth=4` 和 `learning_rate=0.05` 防止过拟合，AUC 提升至 0.84。

---

## 10. 后续可扩展方向

### 短期优化

- **更多算法能力**：添加 LightGBM、CatBoost、SVM、神经网络等能力节点
- **回归/聚类任务**：当前仅支持分类，可扩展 `task_type` 路由逻辑
- **真实数据适配**：支持用户上传 CSV，自动推断列名和数据类型

### 中期架构

- **多轮对话**：支持用户对生成结果的反馈修正（"提高 Recall"、"换用线性模型"）
- **超参数自动调优**：在 validate 节点加入 Optuna 自动搜索
- **能力组合**：支持 SMOTE + XGBoost 的 Pipeline 自动组装
- **分布式执行**：沙箱改用 Docker 容器隔离，支持 GPU 加速

### 长期愿景

- **跨领域迁移**：从客户流失迁移到金融风控、医疗诊断等场景
- **自学习机制**：系统运行产生的数据自动用于微调 LLM 规划能力
- **多智能体协作**：引入专家 Agent（特征工程专家、调参专家）并行工作
- **知识图谱自增长**：新算法经验自动抽取为新节点，无需人工录入
