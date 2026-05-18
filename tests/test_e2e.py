"""End-to-end test: install → train → verify metric thresholds."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Expected Spearman correlations from the comment in train.py.
# Thresholds are set ~1% below the expected values to allow for minor
# numerical variation without masking genuine regressions.
_THRESHOLDS = {
    "Linear GP": 0.91,
    "Tanimoto GP": 0.93,
    "LOCK GP": 0.97,
}


def test_train_end_to_end() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "demo.train", "--train-size", "256"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
    )
    assert result.returncode == 0, f"Train script exited with code {result.returncode}.\nstderr:\n{result.stderr}"

    output = result.stdout
    for name, threshold in _THRESHOLDS.items():
        match = re.search(rf"{re.escape(name)}:.*spearman:\s*([\d.]+)", output)
        assert match, f"Could not find '{name}' metrics in output:\n{output}"
        spearman = float(match.group(1))
        assert spearman >= threshold, (
            f"{name} Spearman {spearman:.3f} is below threshold {threshold}.\nFull output:\n{output}"
        )
