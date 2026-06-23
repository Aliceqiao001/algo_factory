"""Tests for the CodeSandbox executor."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from executor.sandbox import CodeSandbox


def test_sandbox_success():
    """Sandbox executes valid Python and captures stdout correctly."""
    sb = CodeSandbox()
    code = "import json\nprint(json.dumps({'accuracy': 0.9, 'auc': 0.95, 'f1': 0.88}))"
    success, stdout, stderr = sb.execute(code)

    assert success is True, f"Expected success, stderr={stderr}"
    assert stdout.strip() != "", "Expected non-empty stdout"

    parsed = json.loads(stdout.strip())
    assert parsed["accuracy"] == pytest.approx(0.9)
    assert parsed["auc"] == pytest.approx(0.95)
    assert parsed["f1"] == pytest.approx(0.88)


def test_sandbox_failure():
    """Sandbox returns success=False and captures stderr for broken code."""
    sb = CodeSandbox()
    code = "this is not valid python !!!"
    success, stdout, stderr = sb.execute(code)

    assert success is False, "Expected failure for invalid Python"
    assert stderr != "", "Expected non-empty stderr with error details"
