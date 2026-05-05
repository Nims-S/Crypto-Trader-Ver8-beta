"""Legacy compatibility layer for older imports."""

from research.scoring import ScoreDecision, score_metrics, promotion_status

# Backward-compatible fallback for older code that expects legacy exports.
import importlib as _il

try:
    _legacy = _il.import_module("legacy.evolution")
except ImportError:
    _legacy = None
else:
    for _k in dir(_legacy):
        if not _k.startswith("_") and _k not in globals():
            globals()[_k] = getattr(_legacy, _k)

