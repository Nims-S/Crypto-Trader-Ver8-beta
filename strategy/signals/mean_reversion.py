from __future__ import annotations

import pandas as pd

from strategy.state import StrategyState, Signal
from strategy.indicators import compute_indicators


def _cfg(strategy_override: dict | None) -> dict:
    if not strategy_override:
        return {}
    return dict(strategy_override.get("parameters") or strategy_override)


def generate(df: pd.DataFrame, symbol: str, state: StrategyState, df_htf: pd.DataFrame | None = None, strategy_override: dict | None = None):
    cfg = _cfg(strategy_override)

    if df is None or len(df) < 220:
        return None

    df = compute_indicators(df) if "atr" not in df.columns else df
    cur = df.iloc[-1]

    close = float(cur.get("close", 0))
    bb_lower = float(cur.get("bb_lower", 0))
    bb_rank = float(cur.get("bb_width_rank", 0))
    rsi = float(cur.get("rsi", 50))
    atr = float(cur.get("atr", 0))

    if close <= 0 or atr <= 0:
        return None

    bb_rank_max = float(cfg.get("bb_rank_max", 0.30))
    rsi_max = float(cfg.get("rsi_max", 32.0))
    band_buffer = float(cfg.get("band_buffer", 0.01))
    stop_atr_mult = float(cfg.get("stop_atr_mult", 1.6))
    tp1_rr = float(cfg.get("tp1_rr", 1.8))
    size_mult = float(cfg.get("size_multiplier", 0.6))
    cooldown = int(cfg.get("cooldown_bars", 18))
    max_bars = int(cfg.get("max_bars_override", 36))

    # Only in low-vol regimes
    if bb_rank > bb_rank_max:
        return None

    # Deep oversold only
    if rsi > rsi_max:
        return None

    # Must be near lower band
    if close > bb_lower * (1.0 + band_buffer):
        return None

    entry = close
    stop = entry - (stop_atr_mult * atr)
    if stop >= entry:
        return None

    risk = entry - stop
    tp1 = entry + (tp1_rr * risk)

    return Signal(
        "LONG",
        entry,
        stop,
        tp1,
        symbol,
        str(cfg.get("strategy_name", "mean_reversion_v4")),
        str(cfg.get("regime", "mean_reversion")),
        confidence=float(cfg.get("confidence", 0.65)),
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        size_multiplier=size_mult,
        cooldown_bars=cooldown,
        max_bars_override=max_bars,
    )
