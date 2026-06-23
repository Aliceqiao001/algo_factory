"""Validation Agent: runs generated code in a sandbox and checks metric thresholds."""

import json
import logging
from typing import TYPE_CHECKING, Dict

from agents.state import AgentState

if TYPE_CHECKING:
    from executor.sandbox import CodeSandbox

logger = logging.getLogger(__name__)

# Relaxed thresholds — tighten once baseline data quality is confirmed
_THRESHOLD_ACCURACY = 0.6
_THRESHOLD_AUC = 0.65


def _parse_metrics(stdout: str) -> Dict[str, float]:
    """Extract the last JSON object from stdout as the metrics dict."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for line in reversed(lines):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return {k: float(v) for k, v in obj.items()}
        except (json.JSONDecodeError, ValueError):
            continue
    return {}


class ValidatorAgent:
    """Runs generated code via CodeSandbox and evaluates the resulting metrics."""

    def __init__(self, sandbox: "CodeSandbox") -> None:
        self._sandbox = sandbox

    def run(self, state: AgentState) -> dict:
        """Execute the generated code and return execution + validation results.

        Returns
        -------
        dict with keys:
            execution_success, execution_output, execution_error,
            metrics_result, validation_passed
        """
        code = state.get("generated_code", "")
        if not code:
            logger.warning("ValidatorAgent: generated_code is empty")
            return {
                "execution_success": False,
                "execution_output": "",
                "execution_error": "No code to execute",
                "metrics_result": {},
                "validation_passed": False,
            }

        success, stdout, stderr = self._sandbox.execute(code)
        logger.debug(
            "Sandbox result: success=%s stdout_len=%d stderr_len=%d",
            success, len(stdout), len(stderr),
        )

        metrics: Dict[str, float] = {}
        validation_passed = False

        if success:
            metrics = _parse_metrics(stdout)
            if metrics:
                accuracy = metrics.get("accuracy", 0.0)
                auc = metrics.get("auc", metrics.get("roc_auc", 0.0))
                validation_passed = accuracy > _THRESHOLD_ACCURACY and auc > _THRESHOLD_AUC
                if not validation_passed:
                    logger.info(
                        "Metrics below threshold: accuracy=%.3f (>%.2f), auc=%.3f (>%.2f)",
                        accuracy, _THRESHOLD_ACCURACY, auc, _THRESHOLD_AUC,
                    )
            else:
                logger.warning("ValidatorAgent: could not parse metrics from stdout")
        else:
            logger.warning("ValidatorAgent: execution failed\n%s", stderr[:400])

        return {
            "execution_success": success,
            "execution_output": stdout,
            "execution_error": stderr,
            "metrics_result": metrics,
            "validation_passed": validation_passed,
        }
