from __future__ import annotations
from pathlib import Path
import time
import ccxt
import pandas as pd
from strategy.indicators import compute_indicators

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
CACHE_DIR = Path(".backtest_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REQUIRED_INDICATOR_COLS = {"atr","atr_pct","atr_pct_rank","bb_width","bb_width_rank","rolling_body","ema20","ema50","ema200","adx","rsi","macd_hist"}

def _cache_path(sym: str, tf: str, since: int | None, until: int | None) -> Path:
    safe_sym = sym.replace("/", "_")
    since_s = str(since) if since is not None else "none"
    until_s = str(until) if until is not None else "none"
    return CACHE_DIR / f"{safe_sym}_{tf}_{since_s}_{until_s}.csv"

def _normalize_cached_frame(cached: pd.DataFrame) -> pd.DataFrame:
    if cached.empty or "timestamp" not in cached.columns:
        return pd.DataFrame()
    cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce")
    cached = cached.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if not REQUIRED_INDICATOR_COLS.issubset(cached.columns):
        cached = compute_indicators(cached.reset_index())
        if "timestamp" in cached.columns:
            cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce")
            cached = cached.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return cached

def fetch_ohlcv_full(sym, tf, since=None, until=None, use_cache=True) -> pd.DataFrame:
    cache_file = _cache_path(sym, tf, since, until)
    if use_cache and cache_file.exists():
        try:
            cached = pd.read_csv(cache_file)
            cached = _normalize_cached_frame(cached)
            if not cached.empty: return cached
        except Exception:
            pass
    rows = []
    cur = since
    while True:
        chunk = exchange.fetch_ohlcv(sym, timeframe=tf, since=cur, limit=1000)
        if not chunk: break
        rows.extend(chunk)
        cur = chunk[-1][0] + 1
        if len(chunk) < 1000 or (until and cur >= until): break
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty: return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = compute_indicators(df)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    if use_cache:
        df.reset_index().to_csv(cache_file, index=False)
    return df
