from __future__ import annotations

from typing import Any, Dict


def _safe(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return d


def compare_performance(expected: dict[str, Any], live: dict[str, Any]) -> Dict[str, Any]:
    exp_pf = _safe(expected.get("profit_factor", expected.get("pf", 0)))
    exp_wr = _safe(expected.get("win_rate", 0))
    exp_dd = abs(_safe(expected.get("max_drawdown_pct", 0)))

    live_present = any(
        key in live and live.get(key) not in (None, "")
        for key in ("profit_factor", "pf", "win_rate", "max_drawdown_pct")
    )

    live_pf = _safe(live.get("profit_factor", live.get("pf", 0)))
    live_wr = _safe(live.get("win_rate", 0))
    live_dd = abs(_safe(live.get("max_drawdown_pct", 0)))

    if not live_present:
        return {
            "status": "unknown",
            "allocation_multiplier": 1.0,
            "live_pf": live_pf,
            "expected_pf": exp_pf,
            "live_wr": live_wr,
            "expected_wr": exp_wr,
            "live_dd": live_dd,
            "expected_dd": exp_dd,
            "reason": "no live metrics yet",
        }

    status = "ok"
    allocation_multiplier = 1.0

    if exp_pf > 0 and live_pf < exp_pf * 0.7:
        status = "throttle"
        allocation_multiplier *= 0.5

    if exp_wr > 0 and live_wr < exp_wr * 0.7:
        status = "throttle"
        allocation_multiplier *= 0.7

    if exp_dd > 0 and live_dd > exp_dd * 1.5:
        status = "disable"
        allocation_multiplier = 0.0

    return {
        "status": status,
        "allocation_multiplier": float(allocation_multiplier),
        "live_pf": live_pf,
        "expected_pf": exp_pf,
        "live_wr": live_wr,
        "expected_wr": exp_wr,
        "live_dd": live_dd,
        "expected_dd": exp_dd,
    }
