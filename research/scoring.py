from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoreDecision:
    score: float
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
        }


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _min_trades_for_timeframe(timeframe: str | None) -> int:
    tf = (timeframe or "").lower()
    base = {
        "1d": 4,
        "12h": 4,
        "8h": 4,
        "4h": 5,
        "2h": 5,
        "1h": 5,
        "30m": 6,
        "15m": 8,
    }
    return base.get(tf, 5)


def _compute_expectancy(m: dict[str, Any]) -> float:
    trades = m.get("trades_detail") or []
    if not trades:
        return _safe(m.get("avg_trade_pnl", 0.0))

    pnls = [float(t.get("pnl", 0.0)) for t in trades]
    if not pnls:
        return 0.0

    return sum(pnls) / len(pnls)


def score_metrics(m: dict[str, Any], timeframe: str | None = None, min_trades: int | None = None) -> ScoreDecision:
    trades = int(m.get("trades", 0) or 0)
    pf = _safe(m.get("profit_factor", 0))
    wr = _safe(m.get("win_rate", 0))
    dd = _safe(m.get("max_drawdown_pct", 0))
    ret = _safe(m.get("return_pct", 0))
    expectancy = _compute_expectancy(m)

    trade_floor = int(min_trades or _min_trades_for_timeframe(timeframe))
    min_pf = 0.85
    min_wr = 0.33

    reasons: list[str] = []
    if trades < trade_floor:
        reasons.append(f"trades<{trade_floor}")
    if pf < min_pf:
        reasons.append(f"pf<{min_pf}")
    if wr < min_wr:
        reasons.append(f"wr<{min_wr}")
    if ret < 0.0:
        reasons.append("return<0")
    if expectancy <= 0.0:
        reasons.append("expectancy<=0")

    return_score = _clamp(max(ret, 0.0) / 1.5)
    pf_score = _clamp(pf / 2.0)
    wr_score = _clamp(wr)
    dd_score = _clamp(1.0 + (dd / 20.0))
    trade_score = _clamp(trades / float(max(trade_floor * 2, 10)))

    score = (
        0.22 * return_score
        + 0.28 * pf_score
        + 0.20 * wr_score
        + 0.15 * dd_score
        + 0.15 * trade_score
    )

    # hard penalty for negative expectancy
    if expectancy <= 0.0:
        score *= 0.5

    score = _clamp(score)

    passed = len(reasons) == 0 and score > 0.35
    return ScoreDecision(score=score, passed=passed, reasons=tuple(reasons))


def promotion_status(decision: ScoreDecision) -> str:
    return "validated" if decision.passed else "rejected"
