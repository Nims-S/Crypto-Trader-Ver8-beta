from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any


@dataclass(frozen=True)
class MonteCarloConfig:
    iterations: int = 300
    seed: int = 42
    initial_equity: float = 10000.0
    slippage_bps_mean: float = 2.0
    slippage_bps_std: float = 1.0
    return_noise_std: float = 0.05
    min_p05_return_pct: float = 0.0
    max_p95_drawdown_pct: float = 25.0
    max_failure_rate: float = 0.30


def _trade_pnls(bt: dict[str, Any]) -> list[float]:
    trades = bt.get("trades_detail") or []
    pnls = [float(t.get("pnl", 0.0)) for t in trades if isinstance(t, dict)]
    if pnls:
        return pnls

    # fallback
    avg = float(bt.get("avg_trade_pnl", 0.0))
    n = int(bt.get("trades", 0))
    return [avg] * n if n > 0 else []


def _drawdown(curve: list[float]) -> float:
    peak = curve[0]
    worst = 0.0
    for v in curve:
        peak = max(peak, v)
        dd = (v - peak) / peak * 100 if peak > 0 else 0
        worst = min(worst, dd)
    return worst


def _profit_factor(pnls):
    win = sum(p for p in pnls if p > 0)
    loss = abs(sum(p for p in pnls if p < 0))
    return win / loss if loss > 0 else win


def _simulate(pnls, cfg, rng):
    sampled = [rng.choice(pnls) for _ in range(len(pnls))]

    eq = cfg.initial_equity
    curve = [eq]

    for p in sampled:
        noise = rng.gauss(0, cfg.return_noise_std)
        slip = abs(p) * (rng.gauss(cfg.slippage_bps_mean, cfg.slippage_bps_std) / 10000)

        pnl = p * (1 + noise) - slip
        eq += pnl
        curve.append(eq)

    return {
        "return_pct": (eq / cfg.initial_equity - 1) * 100,
        "max_dd": abs(_drawdown(curve)),
        "pf": _profit_factor(sampled),
        "wr": sum(1 for p in sampled if p > 0) / len(sampled),
    }


def run_monte_carlo(backtest: dict[str, Any]) -> dict:
    pnls = _trade_pnls(backtest)

    if not pnls:
        return {"passed": False, "score": 0, "reason": "no_trades"}

    cfg = MonteCarloConfig()
    rng = random.Random(cfg.seed)

    sims = [_simulate(pnls, cfg, rng) for _ in range(cfg.iterations)]

    returns = sorted(s["return_pct"] for s in sims)
    dds = sorted(s["max_dd"] for s in sims)

    p05 = returns[int(0.05 * len(returns))]
    p50 = returns[int(0.50 * len(returns))]
    p95_dd = dds[int(0.95 * len(dds))]

    failure = sum(1 for s in sims if s["return_pct"] <= 0) / len(sims)

    passed = (
        p05 >= cfg.min_p05_return_pct
        and p95_dd <= cfg.max_p95_drawdown_pct
        and failure <= cfg.max_failure_rate
    )

    score = (
        0.4 * max(0, p50 / 10)
        + 0.4 * max(0, p05 / 5)
        + 0.2 * max(0, 1 - p95_dd / cfg.max_p95_drawdown_pct)
    )

    return {
        "passed": passed,
        "score": round(score, 4),
        "summary": {
            "p05": p05,
            "p50": p50,
            "p95_dd": p95_dd,
            "failure_rate": failure,
        },
    }