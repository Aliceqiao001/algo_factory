"""Tests for knowledge graph and vector store components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from knowledge.graph import KnowledgeGraph
from knowledge.schema import AlgorithmCapability, HyperParameter, InputSchema, OutputSchema
from knowledge.vector_store import VectorStore


def test_graph_load():
    """KnowledgeGraph loads exactly 6 capability nodes from capabilities.json."""
    kg = KnowledgeGraph()
    caps = kg.get_all_capabilities()
    assert len(caps) == 6, f"Expected 6 capabilities, got {len(caps)}"
    ids = {c.id for c in caps}
    assert "logistic_regression_churn" in ids
    assert "random_forest_churn" in ids
    assert "xgboost_churn" in ids
    assert "smote_oversampling" in ids
    assert "standard_scaler_preprocessing" in ids
    assert "feature_selection_rfe" in ids


def test_vector_search():
    """VectorStore.search returns at least one result for a relevant query."""
    kg = KnowledgeGraph()
    vs = VectorStore()
    vs.build_from_graph(kg)

    results = vs.search("处理不平衡数据的分类算法", n_results=3)
    assert len(results) >= 1, "Expected at least one search result"

    first = results[0]
    assert "id" in first
    assert "score" in first
    assert "name" in first
    assert isinstance(first["score"], float)
    assert 0.0 <= first["score"] <= 1.0


def test_schema_serialization():
    """AlgorithmCapability survives a round-trip through to_dict / from_dict."""
    cap = AlgorithmCapability(
        id="test_algo",
        name="测试算法",
        category="classification",
        description="用于测试的算法",
        applicable_conditions=["数据量 > 100", "目标变量为二分类"],
        input_schema=InputSchema(
            required_columns=["target"],
            optional_columns=["age"],
            min_rows=100,
            data_types={"target": "int", "age": "float"},
        ),
        output_schema=OutputSchema(
            format="dataframe",
            columns=["y_pred", "y_prob"],
            description="预测结果",
        ),
        metrics=["accuracy", "roc_auc"],
        dependencies=["scikit-learn"],
        code_template="# placeholder",
        hyperparameters=[
            HyperParameter(name="C", default=1.0, range=[0.1, 1.0, 10.0], description="正则化")
        ],
        validation_history=[],
    )

    d = cap.to_dict()
    restored = AlgorithmCapability.from_dict(d)

    assert restored.id == cap.id
    assert restored.name == cap.name
    assert restored.category == cap.category
    assert restored.applicable_conditions == cap.applicable_conditions
    assert restored.metrics == cap.metrics
    assert restored.dependencies == cap.dependencies
    assert restored.hyperparameters[0].name == cap.hyperparameters[0].name
    assert restored.hyperparameters[0].default == cap.hyperparameters[0].default
    assert restored.input_schema.required_columns == cap.input_schema.required_columns
    assert restored.output_schema.format == cap.output_schema.format
