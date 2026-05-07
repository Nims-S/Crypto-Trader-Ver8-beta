from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research.monte_carlo import infer_regime_hint


REGIME_ORDER = ("trend", "breakout", "mean_reversion")


@dataclass(frozen=True)
class RegimePlan:
    regime: str
    objective: str
    directives: dict[str, Any]
    parent_ids: list[str]
    tags: list[str]


def _safe_str(value: Any, default: str = "") -> str:
    try:
        s = str(value or "").strip()
        return s or default
    except Exception:
        return default


def infer_parent_regime(parent: dict[str, Any]) -> str:
    strategy = {
        "tags": parent.get("tags") or [],
        "parameters": parent.get("parameters") or {},
        "entry_mode": (parent.get("parameters") or {}).get("entry_mode") or parent.get("entry_mode") or "",
    }
    backtest = (parent.get("metrics") or {}).get("backtest") or parent.get("backtest") or {}
    regime = infer_regime_hint(strategy, backtest)
    return regime if regime in REGIME_ORDER else "trend"


def regime_objective(regime: str, symbol: str) -> str:
    regime = _safe_str(regime, "trend")
    if regime == "mean_reversion":
        return "profit_factor"
    if regime == "breakout":
        return "density"
    if symbol.startswith("BTC"):
        return "stability"
    return "balanced"


def regime_directives(regime: str, symbol: str) -> dict[str, Any]:
    regime = _safe_str(regime, "trend")
    base = {
        "entry_mode": "trend_pullback" if symbol.startswith("BTC") else "breakout",
        "prefer_trend_pullback": symbol.startswith("BTC"),
    }

    if regime == "trend":
        base.update(
            {
                "entry_mode": "trend_pullback",
                "use_trend_filter": True,
                "use_htf_filter": True,
                "use_volume_filter": True,
                "use_structure_filter": True,
                "use_breakout_filter": False,
                "use_reclaim_filter": False,
                "cooldown_bars": 14,
                "max_bars_override": 84,
                "tp1_rr": 2.1,
                "tp2_rr": 3.8,
                "stop_atr_mult": 2.0,
            }
        )
    elif regime == "breakout":
        base.update(
            {
                "entry_mode": "breakout",
                "use_trend_filter": False,
                "use_htf_filter": True,
                "use_volume_filter": True,
                "use_structure_filter": False,
                "use_breakout_filter": True,
                "use_reclaim_filter": False,
                "cooldown_bars": 10,
                "max_bars_override": 48,
                "tp1_rr": 1.9,
                "tp2_rr": 3.2,
                "stop_atr_mult": 1.7,
            }
        )
    elif regime == "mean_reversion":
        base.update(
            {
                "entry_mode": "mean_reversion",
                "use_trend_filter": False,
                "use_htf_filter": True,
                "use_volume_filter": False,
                "use_structure_filter": False,
                "use_breakout_filter": False,
                "use_reclaim_filter": True,
                "cooldown_bars": 18,
                "max_bars_override": 36,
                "tp1_rr": 1.7,
                "tp2_rr": 2.6,
                "stop_atr_mult": 1.5,
            }
        )
    return base


def cluster_parents_by_regime(parents: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    clusters: dict[str, list[dict[str, Any]]] = {k: [] for k in REGIME_ORDER}
    clusters["unknown"] = []
    for parent in parents:
        regime = infer_parent_regime(parent)
        clusters.setdefault(regime, []).append(parent)
    return clusters


def _parent_budget(regime: str, parent_limits: dict[str, int] | None) -> int:
    defaults = {"trend": 2, "breakout": 3, "mean_reversion": 5}
    if parent_limits:
        defaults.update({str(k): int(v) for k, v in parent_limits.items() if int(v) > 0})
    return max(1, int(defaults.get(regime, 2)))


def build_regime_plans(
    parents: list[dict[str, Any]],
    *,
    symbol: str,
    timeframe: str,
    parent_limits: dict[str, int] | None = None,
) -> list[RegimePlan]:
    clusters = cluster_parents_by_regime(parents)
    plans: list[RegimePlan] = []

    for regime in REGIME_ORDER:
        bucket = clusters.get(regime) or []
        if not bucket:
            continue
        limit = _parent_budget(regime, parent_limits)
        parent_ids = [str(p.get("strategy_id") or p.get("id") or "") for p in bucket[:limit]]
        plans.append(
            RegimePlan(
                regime=regime,
                objective=regime_objective(regime, symbol),
                directives=regime_directives(regime, symbol),
                parent_ids=parent_ids,
                tags=[symbol, timeframe, regime],
            )
        )

    if not plans:
        plans.append(
            RegimePlan(
                regime="trend",
                objective=regime_objective("trend", symbol),
                directives=regime_directives("trend", symbol),
                parent_ids=[],
                tags=[symbol, timeframe, "trend"],
            )
        )

    return plans
