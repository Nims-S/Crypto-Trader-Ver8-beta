from __future__ import annotations

from typing import Tuple

import pandas as pd

from config.execution import EXCHANGE_ID, API_KEY, API_SECRET, API_PASSWORD, USE_SANDBOX, LIVE_LOOKBACK_BARS
from strategy import compute_indicators
from strategy.regime_classifier import classify_market


def _get_exchange():
    try:
        import ccxt  # type: ignore
    except Exception:
        return None
    try:
        exchange_cls = getattr(ccxt, EXCHANGE_ID)
    except AttributeError:
        return None
    params = {"enableRateLimit": True}
    if API_KEY:
        params["apiKey"] = API_KEY
    if API_SECRET:
        params["secret"] = API_SECRET
    if API_PASSWORD:
        params["password"] = API_PASSWORD
    ex = exchange_cls(params)
    try:
        if USE_SANDBOX and hasattr(ex, "set_sandbox_mode"):
            ex.set_sandbox_mode(True)
    except Exception:
        pass
    return ex


def _htf_timeframe(ltf: str) -> str:
    mapping = {
        "1m": "5m",
        "5m": "15m",
        "15m": "1h",
        "1h": "4h",
        "4h": "1d",
        "1d": "1w",
    }
    return mapping.get(ltf, "4h")


def fetch_recent_ohlcv(symbol: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
    ex = _get_exchange()
    if ex is None:
        raise RuntimeError("Exchange unavailable (ccxt not installed or misconfigured)")
    lim = int(limit or LIVE_LOOKBACK_BARS or 300)
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=max(100, lim))
    if not rows:
        raise RuntimeError(f"No OHLCV data for {symbol} {timeframe}")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return compute_indicators(df)


def load_market_bundle(symbol: str, timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    ltf = fetch_recent_ohlcv(symbol, timeframe)
    htf_tf = _htf_timeframe(timeframe)
    htf = fetch_recent_ohlcv(symbol, htf_tf)
    regime = classify_market(ltf, htf)
    return ltf, htf, regime
