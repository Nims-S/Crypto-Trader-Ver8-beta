from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pandas as pd

from registry.store import record_experiment, upsert_strategy
from research.scoring import score_metrics, promotion_status
from strategy import StrategyState, compute_indicators, generate_signal

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".backtest_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})

TAKER_FEE_BPS = 6.0
MAKER_FEE_BPS = 2.0
SLIPPAGE_BPS = 3.0
SLIPPAGE_ATR_MULT = 0.10
RISK_PER_TRADE = 0.01
MAX_NOTIONAL_FRAC = 0.25
DEFAULT_STARTUP_BARS = 260


@dataclass
class Position:
    side: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    qty: float
    tp1_qty: float
    tp2_qty: float
    bars: int = 0
    max_bars: int = 72
    tp1_hit: bool = False
    open_ts: str = ""
    strategy: str = ""


def _to_ms(value: str | None) -> int | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _cache_path(symbol: str, timeframe: str, since: int | None, until: int | None) -> Path:
    safe = symbol.replace("/", "_")
    return CACHE_DIR / f"{safe}_{timeframe}_{since or 'none'}_{until or 'none'}.csv"


def _load_cached(cache_file: Path) -> pd.DataFrame:
    if not cache_file.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(cache_file)
        if "timestamp" not in df.columns:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        return df
    except Exception:
        return pd.DataFrame()


def _store_cache(cache_file: Path, df: pd.DataFrame) -> None:
    try:
        out = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        out.to_csv(cache_file, index=False)
    except Exception:
        pass


def fetch_ohlcv_full(symbol: str, timeframe: str, since: int | None = None, until: int | None = None, use_cache: bool = True) -> pd.DataFrame:
    cache_file = _cache_path(symbol, timeframe, since, until)
    if use_cache:
        cached = _load_cached(cache_file)
        if not cached.empty:
            return cached

    rows: list[list[Any]] = []
    cursor = since
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not chunk:
            break
        rows.extend(chunk)
        cursor = chunk[-1][0] + 1
        if len(chunk) < 1000:
            break
        if until is not None and cursor >= until:
            break
        time.sleep(exchange.rateLimit / 1000)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    df = compute_indicators(df.reset_index())
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

    if use_cache:
        _store_cache(cache_file, df)
    return df


def _htf_timeframe_for_symbol(symbol: str, ltf: str) -> str:
    if symbol == "BTC/USDT":
        if ltf == "1d":
            return "1w"
        return "1d" if ltf in {"15m", "30m", "1h", "2h", "4h"} else "1h"
    return "4h" if ltf in {"15m", "30m", "1h"} else "1d"


def _slip(price: float, atr: float, close: float, side: str) -> float:
    atr_pct = (atr / close) if close else 0.0
    slip = (SLIPPAGE_BPS / 10000.0) + (atr_pct * SLIPPAGE_ATR_MULT)
    return price * (1 + slip) if side == "LONG" else price * (1 - slip)


def _risk_position_size(entry: float, stop: float, capital: float) -> float:
    if entry <= 0 or stop <= 0:
        return 0.0
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return 0.0
    risk_qty = (capital * RISK_PER_TRADE) / stop_dist
    max_qty = (capital * MAX_NOTIONAL_FRAC) / entry
    return max(0.0, min(risk_qty, max_qty))


def _build_entry_levels(signal, entry: float, atr: float) -> tuple[float, float, float, float, float]:
    stop_pct = _safe_float(getattr(signal, "stop_loss_pct", 0.0), 0.0)
    if stop_pct <= 0:
        stop_mult = 2.0
        stop = entry - (stop_mult * atr) if getattr(signal, "side", "LONG") == "LONG" else entry + (stop_mult * atr)
    else:
        stop = entry * (1 - stop_pct) if getattr(signal, "side", "LONG") == "LONG" else entry * (1 + stop_pct)

    risk = abs(entry - stop)
    tp1_rr = max(0.1, _safe_float(getattr(signal, "take_profit_pct", 0.0), 0.0) * entry / max(risk, 1e-9))
    tp2_rr = max(tp1_rr, _safe_float(getattr(signal, "secondary_take_profit_pct", 0.0), 0.0) * entry / max(risk, 1e-9))
    if tp1_rr <= 0:
        tp1_rr = 2.0
    if tp2_rr <= 0:
        tp2_rr = max(tp1_rr * 1.5, 3.0)

    tp1 = entry + (tp1_rr * risk) if getattr(signal, "side", "LONG") == "LONG" else entry - (tp1_rr * risk)
    tp2 = entry + (tp2_rr * risk) if getattr(signal, "side", "LONG") == "LONG" else entry - (tp2_rr * risk)
    be_trigger = entry + (1.0 * risk) if getattr(signal, "side", "LONG") == "LONG" else entry - (1.0 * risk)
    return stop, tp1, tp2, be_trigger, risk


def _fee(price: float, qty: float, bps: float) -> float:
    return price * qty * (bps / 10000.0)


def _close_leg(trades: list[dict[str, Any]], pos: Position, exit_price: float, qty: float, result: str) -> tuple[list[dict[str, Any]], float]:
    qty = min(qty, pos.qty)
    if qty <= 0:
        return trades, 0.0
    if pos.side == "LONG":
        pnl = (exit_price - pos.entry) * qty - _fee(exit_price, qty, MAKER_FEE_BPS)
    else:
        pnl = (pos.entry - exit_price) * qty - _fee(exit_price, qty, MAKER_FEE_BPS)
    trades.append(
        {
            "ts": pos.open_ts,
            "strategy": pos.strategy,
            "side": pos.side,
            "entry_price": round(pos.entry, 6),
            "exit_price": round(exit_price, 6),
            "qty": round(qty, 8),
            "pnl": round(pnl, 6),
            "result": result,
        }
    )
    pos.qty -= qty
    return trades, pnl


def _compute_metrics(trades: list[dict[str, Any]], equity_curve: list[float], cap: float) -> dict[str, Any]:
    pnls = [float(t["pnl"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)
    wr = (len(wins) / len(pnls)) if pnls else 0.0
    avg_trade = (sum(pnls) / len(pnls)) if pnls else 0.0
    eq = np.array(equity_curve if equity_curve else [cap], dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / np.maximum(peak, 1e-9)).min() * 100.0
    return {
        "trades": len(trades),
        "win_rate": round(wr, 3),
        "profit_factor": round(pf, 4),
        "final_equity": round(float(eq[-1]), 2),
        "return_pct": round((float(eq[-1]) / cap - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(float(dd), 4),
        "avg_trade_pnl": round(avg_trade, 6),
    }


def run_backtest(
    sym: str,
    tf: str,
    start: str | None = None,
    end: str | None = None,
    allow_shorts: bool = False,
    max_bars: int = 0,
    use_cache: bool = True,
    strategy_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    since = _to_ms(start)
    until = _to_ms(end)

    df = fetch_ohlcv_full(sym, tf, since, until, use_cache=use_cache)
    if df.empty:
        return {"error": f"no data returned for {sym} on {tf}"}

    htf_tf = _htf_timeframe_for_symbol(sym, tf)
    df_htf = fetch_ohlcv_full(sym, htf_tf, since, until, use_cache=use_cache)
    if df_htf.empty:
        return {"error": f"no HTF data returned for {sym} on {htf_tf}"}

    if max_bars and max_bars > 0:
        warmup = min(DEFAULT_STARTUP_BARS, len(df) - 1)
        df = df.iloc[-(max_bars + warmup):].copy()
        if not df_htf.empty:
            df_htf = df_htf[df_htf.index >= df.index.min()].copy()
            if df_htf.empty:
                return {"error": f"HTF data trimmed away for {sym} on {htf_tf}"}

    htf_pos = np.searchsorted(df_htf.index.values, df.index.values, side="right") - 1

    cash = 10_000.0
    trades: list[dict[str, Any]] = []
    equity_curve: list[float] = []
    pos: Position | None = None
    cooldown_until = -1
    state = StrategyState(allow_shorts=allow_shorts)

    params = (strategy_override or {}).get("parameters") or strategy_override or {}
    params = dict(params or {})

    if params:
        state.allow_shorts = bool(params.get("allow_shorts", state.allow_shorts))
        state.min_adx = float(params.get("min_adx", state.min_adx))
        state.min_atr_rank = float(params.get("min_atr_rank", state.min_atr_rank))
        state.min_bb_rank = float(params.get("min_bb_rank", state.min_bb_rank))

    start_idx = min(max(DEFAULT_STARTUP_BARS, 50), max(len(df) - 2, 1))

    for i in range(start_idx, len(df) - 1):
        bar = df.iloc[i + 1]
        atr = _safe_float(bar.get("atr", 0.0), 0.0)
        close = _safe_float(bar.get("close", 0.0), 0.0)
        high = _safe_float(bar.get("high", 0.0), 0.0)
        low = _safe_float(bar.get("low", 0.0), 0.0)
        open_px = _safe_float(bar.get("open", 0.0), 0.0)

        if pos is not None:
            pos.bars += 1
            if pos.bars >= pos.max_bars:
                exit_px = _slip(close, atr, close, pos.side)
                closed_qty = pos.qty
                trades, _ = _close_leg(trades, pos, exit_px, closed_qty, "MAX_BARS")
                cash += exit_px * closed_qty - _fee(exit_px, closed_qty, TAKER_FEE_BPS)
                pos = None
                cooldown_until = i + 1
                equity_curve.append(cash)
                continue

            if pos.side == "LONG":
                sl_hit = low <= pos.stop
                tp1_hit = high >= pos.tp1
                tp2_hit = high >= pos.tp2
            else:
                sl_hit = high >= pos.stop
                tp1_hit = low <= pos.tp1
                tp2_hit = low <= pos.tp2

            if sl_hit:
                exit_px = _slip(pos.stop, atr, close, pos.side)
                closed_qty = pos.qty
                trades, _ = _close_leg(trades, pos, exit_px, closed_qty, "SL")
                cash += exit_px * closed_qty - _fee(exit_px, closed_qty, TAKER_FEE_BPS)
                pos = None
                cooldown_until = i + 1
                equity_curve.append(cash)
                continue

            if pos and not pos.tp1_hit and tp1_hit:
                exit_px = _slip(pos.tp1, atr, close, pos.side)
                closed_qty = pos.tp1_qty
                trades, _ = _close_leg(trades, pos, exit_px, closed_qty, "TP1")
                cash += exit_px * closed_qty - _fee(exit_px, closed_qty, TAKER_FEE_BPS)
                pos.tp1_hit = True
                pos.stop = pos.entry

            if pos and pos.tp1_hit and tp2_hit:
                exit_px = _slip(pos.tp2, atr, close, pos.side)
                closed_qty = pos.qty
                trades, _ = _close_leg(trades, pos, exit_px, closed_qty, "TP2")
                cash += exit_px * closed_qty - _fee(exit_px, closed_qty, TAKER_FEE_BPS)
                pos = None
                cooldown_until = i + 1
                equity_curve.append(cash)
                continue

        if pos is None and i >= cooldown_until:
            window = df.iloc[: i + 1]
            htf_end = htf_pos[i + 1]
            htf_slice = df_htf.iloc[: htf_end + 1] if htf_end >= 0 else df_htf.iloc[:0]

            sig = generate_signal(window, state=state, symbol=sym, df_htf=htf_slice, strategy_override={"parameters": params})
            if sig is None:
                equity_curve.append(cash)
                continue

            if sig.side == "SHORT" and not allow_shorts:
                equity_curve.append(cash)
                continue

            fill_px = _slip(open_px, atr, close, sig.side)
            stop, tp1, tp2, _, risk = _build_entry_levels(sig, fill_px, atr)
            if risk <= 0:
                equity_curve.append(cash)
                continue

            qty = _risk_position_size(fill_px, stop, cash)
            if qty <= 0:
                equity_curve.append(cash)
                continue

            fee = _fee(fill_px, qty, TAKER_FEE_BPS)
            if qty * fill_px + fee > cash:
                equity_curve.append(cash)
                continue

            tp1_frac = _safe_float(getattr(sig, "tp1_close_fraction", 0.50), 0.50)
            tp1_frac = min(max(tp1_frac, 0.10), 0.90)
            tp1_qty = qty * tp1_frac
            tp2_qty = qty - tp1_qty

            pos = Position(
                side=sig.side,
                entry=fill_px,
                stop=stop,
                tp1=tp1,
                tp2=tp2,
                qty=qty,
                tp1_qty=tp1_qty,
                tp2_qty=tp2_qty,
                max_bars=int(_safe_float(getattr(sig, "max_bars_override", 72), 72)),
                strategy=str(getattr(sig, "strategy", "unknown")),
                open_ts=str(bar.name),
            )
            cash -= qty * fill_px + fee
            equity_curve.append(cash + qty * close)
            continue

        equity_curve.append(cash + (pos.qty * close if pos else 0.0))

    if pos is not None:
        last = df.iloc[-1]
        last_close = _safe_float(last.get("close", 0.0), 0.0)
        last_atr = _safe_float(last.get("atr", 0.0), 0.0)
        exit_px = _slip(last_close, last_atr, last_close, pos.side)
        closed_qty = pos.qty
        trades, _ = _close_leg(trades, pos, exit_px, closed_qty, "EOD_CLOSE")
        cash += exit_px * closed_qty - _fee(exit_px, closed_qty, TAKER_FEE_BPS)
        pos = None

    metrics = _compute_metrics(trades, equity_curve, 10_000.0)
    result = {
        "symbol": sym,
        "ltf_timeframe": tf,
        "htf_timeframe": htf_tf,
        **metrics,
        "trades_detail": trades,
    }
    return result


def _log_backtest_experiment(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    strategy_id = args.strategy_id or f"{args.symbol.replace('/', '_').lower()}_{args.timeframe}_{'short' if args.allow_shorts else 'long'}"
    decision = score_metrics(result)
    payload = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start": args.start,
        "end": args.end,
        "allow_shorts": bool(args.allow_shorts),
        "max_bars": int(args.max_bars or 0),
        "decision": decision.as_dict(),
    }
    experiment = record_experiment(
        strategy_id,
        symbol=args.symbol,
        timeframe=args.timeframe,
        run_type="backtest",
        parameters=payload,
        metrics={**result, "decision": decision.as_dict()},
        passed=decision.passed,
        notes="auto-logged from backtest.py",
    )
    upsert_strategy(
        strategy_id,
        base_strategy=args.base_strategy or strategy_id,
        version=int(args.version or 1),
        status=promotion_status(decision),
        parameters=payload,
        metrics={**result, "decision": decision.as_dict()},
        tags=[args.symbol, args.timeframe, "backtest"],
        source="backtest",
        notes=f"decision={'pass' if decision.passed else 'fail'}",
        active=decision.passed,
    )
    return {"decision": decision.as_dict(), "experiment": experiment}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-bars", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--allow-shorts", action="store_true")
    ap.add_argument("--strategy-id", default=None)
    ap.add_argument("--base-strategy", default=None)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--log-experiment", action="store_true")
    args = ap.parse_args()

    out = run_backtest(
        args.symbol,
        args.timeframe,
        args.start,
        args.end,
        allow_shorts=args.allow_shorts,
        max_bars=args.max_bars,
        use_cache=not args.no_cache,
    )
    if args.log_experiment and "error" not in out:
        out["registry"] = _log_backtest_experiment(args, out)
    print(json.dumps(out, indent=2))
