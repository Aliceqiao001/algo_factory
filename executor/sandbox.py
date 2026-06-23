"""Subprocess-based sandbox for safely executing generated algorithm code."""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# algo_factory project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent


class CodeSandbox:
    """Runs arbitrary Python code in an isolated subprocess.

    The subprocess inherits the current environment (so installed packages are
    available) but has its working directory set to the algo_factory project
    root, which lets generated code access ``data/churn_sample.csv`` via
    relative paths.
    """

    def execute(self, code: str, timeout: int = 60) -> tuple[bool, str, str]:
        """Write *code* to a temp file, run it, capture output, then delete it.

        Parameters
        ----------
        code:
            Complete Python source to execute.
        timeout:
            Maximum wall-clock seconds before the subprocess is killed.

        Returns
        -------
        (success, stdout, stderr)
            ``success`` is ``True`` iff the process exited with code 0.
        """
        ts = int(time.time() * 1000)
        tmp_path = _PROJECT_ROOT / f"temp_algo_{ts}.py"
        try:
            tmp_path.write_text(code, encoding="utf-8")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(_PROJECT_ROOT),
                env=env,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            success = result.returncode == 0
            if not success:
                logger.warning("Sandbox exit code %d\nstderr: %s", result.returncode, stderr[:500])
            return success, stdout, stderr
        except subprocess.TimeoutExpired:
            logger.error("Sandbox timed out after %ds", timeout)
            return False, "", f"TimeoutExpired: execution exceeded {timeout}s"
        except Exception as exc:
            logger.error("Sandbox unexpected error: %s", exc)
            return False, "", str(exc)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
