from __future__ import annotations

import pandas as pd

from strategy.state import StrategyState, Signal
from strategy.indicators import compute_indicators


def _cfg(strategy_override: dict | None) -> dict:
    if not strategy_override:
        return {}
    return dict(strategy_override.get("parameters") or strategy_override)


def _htf_confirm(df_htf: pd.DataFrame | None, cfg: dict) -> bool:
    if df_htf is None or len(df_htf) < 200:
        return True
    h = df_htf.iloc[-1]
    return float(h.get("close", 0)) > float(h.get("ema200", 0))


def generate(df: pd.DataFrame, symbol: str, state: StrategyState, df_htf: pd.DataFrame | None = None, strategy_override: dict | None = None):
    cfg = _cfg(strategy_override)

    if df is None or len(df) < 220:
        return None

    df = compute_indicators(df) if "atr" not in df.columns else df
    if df_htf is not None and len(df_htf) > 0 and "atr" not in df_htf.columns:
        df_htf = compute_indicators(df_htf)

    cur = df.iloc[-1]

    close = float(cur.get("close", 0))
    high20 = float(cur.get("swing_high_20", 0))
    bb_rank = float(cur.get("bb_width_rank", 0))
    atr = float(cur.get("atr", 0))
    vol = float(cur.get("volume", 0))
    vol_sma = float(cur.get("volume_sma20", 0))
    bb_streak = int(cur.get("bbwp_low_streak", 0))

    if close <= 0 or atr <= 0:
        return None

    if not _htf_confirm(df_htf, cfg):
        return None

    if bb_rank > float(cfg.get("bb_rank_max", 0.40)):
        return None

    if bb_streak < int(cfg.get("bb_streak_min", 3)):
        return None

    if close <= high20:
        return None

    if vol_sma > 0 and vol < vol_sma * float(cfg.get("volume_multiplier", 1.2)):
        return None

    entry = close
    stop = entry - (float(cfg.get("stop_atr_mult", 1.7)) * atr)
    if stop >= entry:
        return None

    risk = entry - stop
    tp1 = entry + (float(cfg.get("tp1_rr", 2.1)) * risk)

    return Signal(
        "LONG",
        entry,
        stop,
        tp1,
        symbol,
        "breakout_v4",
        "breakout",
        confidence=float(cfg.get("confidence", 0.78)),
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        size_multiplier=float(cfg.get("size_multiplier", 0.65)),
        cooldown_bars=int(cfg.get("cooldown_bars", 26)),
        max_bars_override=int(cfg.get("max_bars_override", 60)),
    )
