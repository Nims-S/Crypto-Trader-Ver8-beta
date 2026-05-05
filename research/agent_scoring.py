from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentScore:
    score: float
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _normalize_range(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low))


def score_candidate(
    backtest: dict[str, Any],
    walk_forward: dict[str, Any],
    monte_carlo: dict[str, Any],
    *,
    goal_return_pct: float = 0.25,
    max_drawdown_pct: float = 15.0,
    min_profit_factor: float = 0.95,
    min_win_rate: float = 0.40,
    min_trades: int = 12,
) -> AgentScore:
    return_pct = _safe_float(backtest.get("return_pct", 0.0), 0.0)
    drawdown_pct = abs(_safe_float(backtest.get("max_drawdown_pct", 0.0), 0.0))
    profit_factor = _safe_float(backtest.get("profit_factor", 0.0), 0.0)
    win_rate = _safe_float(backtest.get("win_rate", 0.0), 0.0)
    trades = int(backtest.get("trades", 0) or 0)

    wf_score = _safe_float(walk_forward.get("score", 0.0), 0.0)
    wf_passed = bool(walk_forward.get("passed", False))
    wf_spread = abs(_safe_float(walk_forward.get("score_spread", 0.0), 0.0))

    mc_worst_dd = abs(_safe_float(monte_carlo.get("worst_drawdown_pct", monte_carlo.get("worst_drawdown", 0.0)), 0.0))
    mc_median_return = _safe_float(monte_carlo.get("median_return_pct", monte_carlo.get("median_final_return_pct", 0.0)), 0.0)

    reasons: list[str] = []

    # Hard quality gates (realistic scale)
    if trades < min_trades:
        reasons.append(f"trades<{min_trades}")
    if profit_factor < min_profit_factor:
        reasons.append(f"pf<{min_profit_factor}")
    if win_rate < min_win_rate:
        reasons.append(f"wr<{min_win_rate}")
    if return_pct < 0.0:
        reasons.append("return<0")
    if drawdown_pct > max_drawdown_pct:
        reasons.append(f"dd>{max_drawdown_pct}")
    if not wf_passed:
        reasons.append("walk_forward_failed")
    if wf_spread > 0.40:
        reasons.append("wf_spread>0.40")
    if mc_worst_dd > max_drawdown_pct:
        reasons.append(f"mc_dd>{max_drawdown_pct}")

    # Normalized scoring (return now properly scaled)
    return_score = _normalize_range(max(return_pct, 0.0), 0.0, max(goal_return_pct, 0.25))
    dd_score = _clamp(1.0 - (drawdown_pct / max_drawdown_pct if max_drawdown_pct > 0 else 1.0))
    pf_score = _normalize_range(profit_factor, min_profit_factor, max(min_profit_factor + 1.2, 2.2))
    wr_score = _normalize_range(win_rate, min_win_rate, min(0.70, min_win_rate + 0.25))
    wf_norm = _clamp(wf_score)
    mc_dd_score = _clamp(1.0 - (mc_worst_dd / max_drawdown_pct if max_drawdown_pct > 0 else 1.0))
    mc_ret_score = _normalize_range(max(mc_median_return, 0.0), 0.0, max(goal_return_pct, 0.25))

    score = (
        0.25 * return_score
        + 0.20 * dd_score
        + 0.15 * pf_score
        + 0.10 * wr_score
        + 0.15 * wf_norm
        + 0.10 * mc_dd_score
        + 0.05 * mc_ret_score
    )
    score = round(_clamp(score, 0.0, 1.0), 6)

    passed = len(reasons) == 0 and score >= 0.55
    return AgentScore(score=score, passed=passed, reasons=tuple(reasons))
