from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any
import random, time
from execution.backtest.data import fetch_ohlcv_full
from registry.store import compute_logic_hash, list_strategies, upsert_strategy

@dataclass(frozen=True)
class ScoreDecision:
    score: float
    passed: bool
    reasons: tuple[str, ...]
    def as_dict(self):
        return {"score": float(self.score), "passed": bool(self.passed), "reasons": list(self.reasons)}

def _safe(v, d=0.0):
    try: return float(v)
    except: return d

def score_metrics(m: dict) -> ScoreDecision:
    trades = int(m.get("trades", 0)); pf = _safe(m.get("profit_factor", 0)); wr = _safe(m.get("win_rate", 0)); dd = _safe(m.get("max_drawdown_pct", 0))
    reasons = []
    if trades < 20: reasons.append("trades<20")
    if pf < 1.1: reasons.append("pf<1.1")
    if wr < 0.45: reasons.append("wr<0.45")
    score = (0.4 * min(pf / 2.0, 1) + 0.3 * wr + 0.2 * max(0, 1 + dd / 20) + 0.1 * min(trades / 40, 1))
    return ScoreDecision(score, len(reasons) == 0 and score > 0.55, tuple(reasons))
