from __future__ import annotations
import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _macd_hist(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd = ema(close, 12) - ema(close, 26)
    signal = ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist

def _percent_rank(series: pd.Series, window: int = 252) -> pd.Series:
    def _rank(x: pd.Series) -> float:
        last = x.iloc[-1]
        return float((x <= last).mean())
    return series.rolling(window, min_periods=max(20, window // 4)).apply(_rank, raw=False)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    high = df["high"]; low = df["low"]; close = df["close"]; prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=14).mean()
    df["atr_pct"] = df["atr"] / close.replace(0.0, np.nan)
    df["rolling_body"] = (df["close"] - df["open"]).abs().rolling(20, min_periods=20).mean()
    df["ema20"] = ema(close, 20); df["ema50"] = ema(close, 50); df["ema200"] = ema(close, 200)
    df["sma200"] = close.rolling(200, min_periods=200).mean()
    df["ema50_slope"] = df["ema50"].diff(3) / 3.0
    df["ema20_slope"] = df["ema20"].diff(3) / 3.0
    df["rsi"] = _rsi(close, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd_hist(close)
    up_move = high.diff(); down_move = -low.diff()
    plus_dm = pd.Series(0.0, index=df.index); minus_dm = pd.Series(0.0, index=df.index)
    plus_mask = (up_move > down_move) & (up_move > 0); minus_mask = (down_move > up_move) & (down_move > 0)
    plus_dm.loc[plus_mask] = up_move.loc[plus_mask]; minus_dm.loc[minus_mask] = down_move.loc[minus_mask]
    atr14 = tr.rolling(14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(14, min_periods=14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14, min_periods=14).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    df["adx"] = dx.rolling(14, min_periods=14).mean()
    bb_mid = close.rolling(20, min_periods=20).mean(); bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + (2 * bb_std); bb_lower = bb_mid - (2 * bb_std)
    df["bb_mid"] = bb_mid; df["bb_upper"] = bb_upper; df["bb_lower"] = bb_lower
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)
    df["atr_pct_rank"] = _percent_rank(df["atr_pct"].ffill(), 252)
    df["bb_width_rank"] = _percent_rank(df["bb_width"].ffill(), 252)
    df["bbwp"] = df["bb_width_rank"]
    df["swing_high_20"] = df["high"].rolling(20, min_periods=20).max()
    df["swing_low_20"] = df["low"].rolling(20, min_periods=20).min()
    df["range_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0.0, np.nan)
    df["volume_sma20"] = df["volume"].rolling(20, min_periods=20).mean()
    df["bbwp_low_streak"] = (df["bbwp"] < 0.10).astype(int).groupby((df["bbwp"] >= 0.10).astype(int).cumsum()).cumsum()
    return df
