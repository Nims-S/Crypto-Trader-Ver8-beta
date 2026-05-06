from __future__ import annotations

from typing import Any

from strategy.state import StrategyState
from strategy.regime_classifier import classify_market
from strategy.signals.trend import generate as trend_generate
from strategy.signals.mean_reversion import generate as mr_generate
from strategy.signals.breakout import generate as breakout_generate


def _normalize_mode(mode: Any) -> str:
    value = str(mode or "").strip().lower()
    if value in {"trend_pullback", "trend_following", "trend"}:
        return "trend"
    if value in {"breakout", "mean_reversion"}:
        return value
    return ""


def _override_mode(strategy_override: dict[str, Any] | None) -> str:
    if not isinstance(strategy_override, dict):
        return ""
    direct = _normalize_mode(strategy_override.get("entry_mode"))
    if direct:
        return direct
    nested = strategy_override.get("parameters")
    if isinstance(nested, dict):
        return _normalize_mode(nested.get("entry_mode"))
    return ""


def generate_signal(
    df,
    state: StrategyState,
    symbol: str,
    df_htf=None,
    strategy_override: dict[str, Any] | None = None,
):
    mode = _override_mode(strategy_override)
    if mode:
        if mode == "trend":
            return trend_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)
        if mode == "breakout":
            return breakout_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)
        if mode == "mean_reversion":
            return mr_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)

    regime = classify_market(df, df_htf)

    if regime == "trend":
        return trend_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)
    if regime == "breakout":
        return breakout_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)

    return mr_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)
