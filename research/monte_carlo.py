from __future__ import annotations

import math
import random
import statistics
from typing import Any, Iterable

from registry.store import export_trade_history


def _returns_from_trades(trades: Iterable[dict]) -> list[float]:
    out: list[float] = []
    for t in trades:
        pnl = t.get("pnl", None)
        if pnl is None:
            continue
        try:
            out.append(float(pnl))
        except Exception:
            continue
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _drawdown_stats(equity_path: list[float]) -> float:
    if not equity_path:
        return 0.0
    peak = equity_path[0]
    max_dd = 0.0
    for x in equity_path:
        peak = max(peak, x)
        max_dd = min(max_dd, x - peak)
    return max_dd


def _bootstrap_paths(returns: list[float], simulations: int, horizon: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    paths: list[list[float]] = []
    for _ in range(max(1, simulations)):
        equity = 0.0
        path = []
        for _ in range(max(1, horizon)):
            equity += rng.choice(returns)
            path.append(equity)
        paths.append(path)
    return paths


def run_monte_carlo_from_trades(
    trades: list[dict[str, Any]],
    *,
    simulations: int = 1000,
    horizon: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    returns = _returns_from_trades(trades)
    if not returns:
        return {"error": "no_trade_history"}
    horizon = int(horizon or len(returns) or 1)
    paths = _bootstrap_paths(returns, simulations=simulations, horizon=horizon, seed=seed)
    finals = [p[-1] for p in paths if p]
    dds = [_drawdown_stats(p) for p in paths if p]
    finals_sorted = sorted(finals)
    p5_idx = max(0, int(math.floor(0.05 * (len(finals_sorted) - 1))))
    p95_idx = max(0, int(math.floor(0.95 * (len(finals_sorted) - 1))))
    return {
        "simulations": int(simulations),
        "horizon": horizon,
        "mean_final": statistics.mean(finals),
        "median_final": statistics.median(finals),
        "p5": finals_sorted[p5_idx],
        "p95": finals_sorted[p95_idx],
        "avg_drawdown": statistics.mean(dds),
        "worst_drawdown": min(dds) if dds else 0.0,
        "return_sample_count": len(returns),
    }


def run_monte_carlo_from_summary(
    summary: dict[str, Any],
    *,
    simulations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    trades = int(summary.get("trades", 0) or 0)
    if trades <= 0:
        return {"error": "no_trade_history"}

    win_rate = min(0.99, max(0.01, _safe_float(summary.get("win_rate", 0.0), 0.0)))
    pf = max(0.1, _safe_float(summary.get("profit_factor", 1.0), 1.0))
    avg_trade = _safe_float(summary.get("avg_trade_pnl", 0.0), 0.0)
    return_pct = _safe_float(summary.get("return_pct", 0.0), 0.0)
    scale = max(abs(avg_trade), abs(return_pct) * 100.0 / max(trades, 1), 0.1)

    win_mean = scale * max(1.0, pf)
    loss_mean = -scale
    win_std = max(0.05, abs(win_mean) * 0.25)
    loss_std = max(0.05, abs(loss_mean) * 0.25)

    rng = random.Random(seed)
    synth_returns: list[float] = []
    for _ in range(trades):
        if rng.random() < win_rate:
            synth_returns.append(rng.normalvariate(win_mean, win_std))
        else:
            synth_returns.append(rng.normalvariate(loss_mean, loss_std))

    return run_monte_carlo_from_trades([{ "pnl": r } for r in synth_returns], simulations=simulations, horizon=trades, seed=seed)


def run_monte_carlo(
    strategy_id: str | None = None,
    *,
    simulations: int = 1000,
    horizon: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    trades = export_trade_history(strategy_id=strategy_id)
    if trades:
        return run_monte_carlo_from_trades(trades, simulations=simulations, horizon=horizon, seed=seed)
    return {"error": "no_trade_history"}
