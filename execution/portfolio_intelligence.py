from __future__ import annotations

import os
from typing import Any, Iterable

import pandas as pd

from execution.lifecycle import lifecycle_multiplier

DEFAULT_MAX_LIVE_POSITIONS = int(os.getenv("MAX_LIVE_POSITIONS", "6"))
DEFAULT_MAX_SYMBOL_EXPOSURE_FRAC = float(os.getenv("MAX_SYMBOL_EXPOSURE_FRAC", "0.35"))
DEFAULT_MAX_CLUSTER_EXPOSURE_FRAC = float(os.getenv("MAX_CLUSTER_EXPOSURE_FRAC", "0.55"))
DEFAULT_MAX_CORRELATION_AVG = float(os.getenv("MAX_CORRELATION_AVG", "0.65"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _returns(df: pd.DataFrame | None) -> pd.Series:
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df["close"], errors="coerce").dropna()
    return s.pct_change().dropna()


def _avg_abs_corr(target: pd.Series, others: Iterable[pd.Series]) -> float:
    vals: list[float] = []
    for other in others:
        if target.empty or other.empty:
            continue
        joined = pd.concat([target, other], axis=1, join="inner").dropna()
        if joined.shape[0] < 8:
            continue
        corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if pd.notna(corr):
            vals.append(abs(float(corr)))
    return sum(vals) / len(vals) if vals else 0.0


def _match_regime(strategy_regime: str | None, route_regime: str | None) -> float:
    if not strategy_regime or not route_regime:
        return 1.0
    s = str(strategy_regime).lower()
    r = str(route_regime).lower()
    if s == r:
        return 1.15
    if s in r or r in s:
        return 1.05
    return 0.85


def build_portfolio_intelligence(
    routes: list[dict[str, Any]],
    portfolio,
    market_cache: dict[tuple[str, str], tuple[Any, Any, str]],
    *,
    max_symbol_exposure_frac: float = DEFAULT_MAX_SYMBOL_EXPOSURE_FRAC,
    max_cluster_exposure_frac: float = DEFAULT_MAX_CLUSTER_EXPOSURE_FRAC,
    max_correlation_avg: float = DEFAULT_MAX_CORRELATION_AVG,
    max_live_positions: int = DEFAULT_MAX_LIVE_POSITIONS,
) -> dict[str, dict[str, Any]]:
    """Build per-strategy allocation context for lifecycle, regime, and exposure control."""
    symbol_exposure: dict[str, float] = {}
    for pos in (portfolio.positions or {}).values():
        symbol = str(pos.get("symbol") or "")
        symbol_exposure[symbol] = symbol_exposure.get(symbol, 0.0) + _safe_float(pos.get("capital", 0.0), 0.0)

    active_returns: dict[tuple[str, str], pd.Series] = {}
    for key, bundle in market_cache.items():
        ltf = bundle[0] if bundle else None
        active_returns[key] = _returns(ltf)

    context: dict[str, dict[str, Any]] = {}
    open_positions = len(getattr(portfolio, "positions", {}) or {})

    for route in routes:
        sid = str(route.get("strategy_id") or "")
        symbol = str(route.get("symbol") or "")
        timeframe = str(route.get("timeframe") or "")
        regime = route.get("regime")
        row = route.get("strategy") or {}

        runtime = (getattr(portfolio, "strategy_runtime", {}) or {}).get(sid, {})
        lifecycle = lifecycle_multiplier(runtime, int(getattr(portfolio, "cycle", 0) or 0))
        regime_weight = _match_regime(row.get("regime_profile"), regime)

        target = active_returns.get((symbol, timeframe), pd.Series(dtype=float))
        peers = [series for key, series in active_returns.items() if key != (symbol, timeframe)]
        avg_corr = _avg_abs_corr(target, peers)
        corr_weight = max(0.25, 1.0 - min(avg_corr, 1.0) * 0.5)
        if avg_corr >= max_correlation_avg:
            corr_weight *= 0.8

        live_metrics = (getattr(portfolio, "live_metrics", {}) or {}).get(sid, {})
        live_pf = _safe_float(live_metrics.get("profit_factor", 0.0), 0.0)
        live_wr = _safe_float(live_metrics.get("win_rate", 0.0), 0.0)
        live_multiplier = 1.0
        if live_pf and live_pf < 1.0:
            live_multiplier *= 0.75
        if live_wr and live_wr < 0.45:
            live_multiplier *= 0.85

        current_symbol_exposure = _safe_float(symbol_exposure.get(symbol, 0.0), 0.0)
        symbol_cap = max(0.0, max_symbol_exposure_frac * float(getattr(portfolio, "total_capital", 0.0)) - current_symbol_exposure)
        cluster_cap = max(0.0, max_cluster_exposure_frac * float(getattr(portfolio, "total_capital", 0.0)) - current_symbol_exposure)
        remaining_cash = max(0.0, float(getattr(portfolio, "cash", 0.0)))

        enabled = True
        if open_positions >= max_live_positions and sid not in (getattr(portfolio, "positions", {}) or {}):
            enabled = False
        if symbol_cap <= 0.0 or remaining_cash <= 0.0:
            enabled = False

        max_capital = max(0.0, min(symbol_cap, cluster_cap, remaining_cash))
        multiplier = lifecycle * regime_weight * corr_weight * live_multiplier
        if not enabled:
            multiplier = 0.0

        context[sid] = {
            "enabled": enabled,
            "multiplier": multiplier,
            "max_capital": max_capital,
            "symbol": symbol,
            "timeframe": timeframe,
            "regime": regime,
            "avg_corr": avg_corr,
            "lifecycle": lifecycle,
            "regime_weight": regime_weight,
            "corr_weight": corr_weight,
            "live_multiplier": live_multiplier,
            "symbol_cap": symbol_cap,
            "cluster_cap": cluster_cap,
        }

    return context


def portfolio_snapshot(portfolio) -> dict[str, Any]:
    return {
        "cash": float(getattr(portfolio, "cash", 0.0) or 0.0),
        "total_capital": float(getattr(portfolio, "total_capital", 0.0) or 0.0),
        "open_positions": len(getattr(portfolio, "positions", {}) or {}),
        "active_strategies": len(getattr(portfolio, "allocations", {}) or {}),
        "cycle": int(getattr(portfolio, "cycle", 0) or 0),
    }
