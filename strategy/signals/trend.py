from __future__ import annotations

import pandas as pd

from strategy.state import StrategyState, Signal
from strategy.indicators import compute_indicators


def _cfg(strategy_override: dict | None) -> dict:
    if not strategy_override:
        return {}
    if isinstance(strategy_override, dict):
        return dict(strategy_override.get("parameters") or strategy_override)
    return {}


def _recent_cross_above(series_a: pd.Series, series_b: pd.Series, lookback: int = 3) -> bool:
    if len(series_a) < lookback + 2 or len(series_b) < lookback + 2:
        return False
    a = series_a.iloc[-(lookback + 1):]
    b = series_b.iloc[-(lookback + 1):]
    prev = (a.shift(1) <= b.shift(1))
    now = a > b
    return bool((prev & now).any())


def _htf_confirm(df_htf: pd.DataFrame | None, cfg: dict) -> bool:
    if df_htf is None or len(df_htf) < 220:
        return True
    h = df_htf.iloc[-1]
    close = float(h.get("close", 0))
    ema20 = float(h.get("ema20", 0))
    ema50 = float(h.get("ema50", 0))
    ema200 = float(h.get("ema200", 0))
    adx = float(h.get("adx", 0))
    bb_rank = float(h.get("bb_width_rank", 0))
    return bool(
        close > ema200
        and ema20 > ema50 > ema200
        and adx >= float(cfg.get("htf_adx_min", 18.0))
        and bb_rank >= float(cfg.get("htf_bb_rank_min", 0.30))
    )


def generate(df: pd.DataFrame, symbol: str, state: StrategyState, df_htf: pd.DataFrame | None = None, strategy_override: dict | None = None):
    cfg = _cfg(strategy_override)

    if df is None or len(df) < 260:
        return None

    df = compute_indicators(df) if "atr" not in df.columns else df
    if df_htf is not None and len(df_htf) > 0 and "atr" not in df_htf.columns:
        df_htf = compute_indicators(df_htf)

    cur = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(cur.get("close", 0))
    ema20 = float(cur.get("ema20", 0))
    ema50 = float(cur.get("ema50", 0))
    ema200 = float(cur.get("ema200", 0))
    adx = float(cur.get("adx", 0))
    bb_rank = float(cur.get("bb_width_rank", 0))
    atr_rank = float(cur.get("atr_pct_rank", 0))
    rsi = float(cur.get("rsi", 50))
    vol = float(cur.get("volume", 0))
    vol_sma = float(cur.get("volume_sma20", 0))

    if close <= 0 or ema20 <= 0 or ema50 <= 0 or ema200 <= 0:
        return None

    if not (close > ema200 and ema20 > ema50 > ema200):
        return None

    if not _htf_confirm(df_htf, cfg):
        return None

    min_adx = float(cfg.get("min_adx", max(state.min_adx + 5.0, 23.0)))
    min_bb_rank = float(cfg.get("min_bb_rank", 0.38))
    min_atr_rank = float(cfg.get("min_atr_rank", 0.32))
    rsi_min = float(cfg.get("rsi_min", 52.0))
    rsi_max = float(cfg.get("rsi_max", 70.0))
    vol_mult = float(cfg.get("volume_multiplier", 1.08))
    pullback_lookback = int(cfg.get("pullback_lookback", 3))
    pullback_bars = int(cfg.get("pullback_bars", 1))

    if adx < min_adx:
        return None

    if bb_rank < min_bb_rank or atr_rank < min_atr_rank:
        return None

    if rsi < rsi_min or rsi > rsi_max:
        return None

    had_pullback = bool(
        prev.get("close", 0) <= prev.get("ema20", 0)
        or prev.get("low", 0) <= prev.get("ema20", 0)
        or _recent_cross_above(df["close"], df["ema20"], lookback=max(2, pullback_lookback))
    )
    if pullback_bars > 1 and len(df) >= pullback_bars:
        recent = df.iloc[-pullback_bars:]
        had_pullback = had_pullback or bool((recent["low"] <= recent["ema20"]).any())
    if not had_pullback:
        return None

    if vol_sma > 0 and vol < vol_sma * vol_mult:
        return None

    entry = close
    atr = float(cur.get("atr", 0))
    if entry <= 0 or atr <= 0:
        return None

    stop_mult = float(cfg.get("stop_atr_mult", 2.0))
    stop = entry - (stop_mult * atr)
    if stop >= entry:
        return None

    risk = entry - stop
    tp1_rr = float(cfg.get("tp1_rr", 2.3))
    tp2_rr = float(cfg.get("tp2_rr", 4.0))
    tp1 = entry + (tp1_rr * risk)
    tp2 = entry + (tp2_rr * risk)

    tp1_frac = float(cfg.get("tp1_close_fraction", 0.50))
    tp2_frac = float(cfg.get("tp2_close_fraction", 0.50))
    size_mult = float(cfg.get("size_multiplier", 0.80))
    cooldown = int(cfg.get("cooldown_bars", 30))
    max_bars = int(cfg.get("max_bars_override", 84))

    return Signal(
        "LONG",
        entry,
        stop,
        tp1,
        symbol,
        str(cfg.get("strategy_name", "trend_following_v4")),
        str(cfg.get("regime", "trend")),
        confidence=float(cfg.get("confidence", 0.84)),
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        secondary_take_profit_pct=(tp2 - entry) / entry,
        tp1_close_fraction=tp1_frac,
        tp2_close_fraction=tp2_frac,
        size_multiplier=size_mult,
        cooldown_bars=cooldown,
        max_bars_override=max_bars,
        trail_atr_mult=float(cfg.get("trail_atr_mult", 1.5)),
        trail_ema20=bool(cfg.get("trail_ema20", False)),
    )
