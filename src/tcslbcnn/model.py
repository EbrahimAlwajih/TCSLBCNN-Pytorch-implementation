"""
Thin wrapper so the project has an importable package.
Gradually move logic from repo root files into src/tcslbcnn/.
"""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LEGACY = _ROOT / "tcslbcnn_model.py"

spec = importlib.util.spec_from_file_location("legacy_tcslbcnn_model", _LEGACY)
legacy = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(legacy)

# Re-export anything you commonly use
TCSLBCNN = getattr(legacy, "TCSLBCNN", None)
