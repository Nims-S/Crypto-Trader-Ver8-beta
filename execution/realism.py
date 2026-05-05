from __future__ import annotations

import hashlib
from typing import Any, Dict

import pandas as pd

from config.execution import (
    BASE_SLIPPAGE_BPS,
    ATR_SLIPPAGE_MULT,
    IMPACT_SLIPPAGE_MULT,
    MIN_FILL_PROBABILITY,
    LATENCY_MS_BASE,
    LATENCY_ATR_MULT,
    LATENCY_IMPACT_MULT,
)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 1:
        return 0.0
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    c = close.iloc[-1]
    if c <= 0:
        return 0.0
    return float(atr / c)


def _range_pct(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    last = df.iloc[-1]
    high = _safe_float(last.get("high"))
    low = _safe_float(last.get("low"))
    close = _safe_float(last.get("close"))
    if close <= 0:
        return 0.0
    return float(abs(high - low) / close)


def _avg_dollar_volume(df: pd.DataFrame, window: int = 20) -> float:
    if df is None or len(df) < 2:
        return 0.0
    sub = df.tail(window)
    vol = sub.get("volume")
    close = sub.get("close")
    if vol is None or close is None:
        return 0.0
    dv = (pd.to_numeric(vol, errors="coerce") * pd.to_numeric(close, errors="coerce")).dropna()
    if dv.empty:
        return 0.0
    return float(dv.mean())


def _rand01(key: str) -> float:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # take first 8 hex chars
    n = int(h[:8], 16)
    return (n % 10_000_000) / 10_000_000.0


def estimate_execution(
    df: pd.DataFrame,
    *,
    price: float,
    side: str,
    notional: float,
    symbol: str,
    timeframe: str,
    cycle: int,
    action: str,
    urgency: float = 1.0,
) -> Dict[str, Any]:
    atr = _atr_pct(df)
    rng = _range_pct(df)
    adv = _avg_dollar_volume(df)

    impact = 0.0
    if adv > 0:
        impact = min(1.0, max(0.0, notional / adv))

    slippage_bps = (
        BASE_SLIPPAGE_BPS
        + (atr * 10_000.0) * (ATR_SLIPPAGE_MULT / 100.0)
        + (rng * 10_000.0) * 0.08
        + impact * IMPACT_SLIPPAGE_MULT
        + urgency * 3.0
    )

    latency_ms = int(
        LATENCY_MS_BASE
        + atr * LATENCY_ATR_MULT
        + impact * LATENCY_IMPACT_MULT
    )

    fill_prob = max(MIN_FILL_PROBABILITY, 0.98 - (slippage_bps / 1200.0) - impact * 0.2)

    key = f"{symbol}|{timeframe}|{cycle}|{action}"
    rnd = _rand01(key)
    filled = rnd <= fill_prob

    slip_frac = slippage_bps / 10_000.0
    if side.upper() == "LONG":
        fill_price = price * (1.0 + slip_frac)
    else:
        fill_price = price * (1.0 - slip_frac)

    return {
        "expected_price": float(price),
        "fill_price": float(fill_price),
        "slippage_bps": float(slippage_bps),
        "latency_ms": int(latency_ms),
        "fill_probability": float(fill_prob),
        "filled": bool(filled),
        "impact": float(impact),
        "atr_pct": float(atr),
    }
