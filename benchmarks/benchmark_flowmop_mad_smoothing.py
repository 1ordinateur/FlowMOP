"""Compatibility wrapper for the paper-level MAD smoothing benchmark script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_flowmop_mad_smoothing.py"
_SPEC = importlib.util.spec_from_file_location("_flowmop_paper_benchmark_flowmop_mad_smoothing", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load benchmark implementation from {_SCRIPT_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

if __name__ == "__main__":
    raise SystemExit(_MODULE.main())

sys.modules[__name__] = _MODULE
