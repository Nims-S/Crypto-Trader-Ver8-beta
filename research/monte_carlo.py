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


REGIME_DEFAULTS: dict[str, dict[str, float]] = {
    "trend": {"min_p05_return_pct": 0.0, "max_p95_drawdown_pct": 22.0, "max_failure_rate": 0.28, "return_noise_std": 0.045},
    "breakout": {"min_p05_return_pct": 0.0, "max_p95_drawdown_pct": 24.0, "max_failure_rate": 0.30, "return_noise_std": 0.05},
    "mean_reversion": {"min_p05_return_pct": 0.0, "max_p95_drawdown_pct": 18.0, "max_failure_rate": 0.22, "return_noise_std": 0.035},
    "unknown": {"min_p05_return_pct": 0.0, "max_p95_drawdown_pct": 25.0, "max_failure_rate": 0.30, "return_noise_std": 0.05},
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def infer_regime_hint(strategy: dict[str, Any] | None = None, backtest: dict[str, Any] | None = None) -> str:
    """Infer a coarse regime label from strategy metadata and backtest shape."""

    strategy = strategy or {}
    backtest = backtest or {}
    tags = {str(t).lower() for t in (strategy.get("tags") or [])}
    params = strategy.get("parameters") or {}
    entry_mode = str(params.get("entry_mode") or strategy.get("entry_mode") or "").lower()

    if "mean_reversion" in tags or entry_mode == "mean_reversion":
        return "mean_reversion"
    if "breakout" in tags or entry_mode == "breakout":
        return "breakout"
    if "trend" in tags or entry_mode == "trend_pullback" or bool(params.get("use_trend_filter", False)):
        return "trend"

    # Backtest-level hints when tags are absent
    trades = _safe_int(backtest.get("trades", 0), 0)
    pf = _safe_float(backtest.get("profit_factor", 0.0), 0.0)
    dd = abs(_safe_float(backtest.get("max_drawdown_pct", 0.0), 0.0))
    wr = _safe_float(backtest.get("win_rate", 0.0), 0.0)

    if trades >= 8 and pf >= 1.5 and wr >= 0.55 and dd <= 4.0:
        return "mean_reversion"
    if trades >= 4 and pf >= 1.2 and wr <= 0.65 and dd <= 5.0:
        return "trend"
    if trades <= 3 and pf >= 1.0:
        return "breakout"
    return "unknown"


def _trade_pnls(bt: dict[str, Any]) -> list[float]:
    trades = bt.get("trades_detail") or []
    pnls = [float(t.get("pnl", 0.0)) for t in trades if isinstance(t, dict)]
    if pnls:
        return pnls

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


def _profit_factor(pnls: list[float]) -> float:
    win = sum(p for p in pnls if p > 0)
    loss = abs(sum(p for p in pnls if p < 0))
    return win / loss if loss > 0 else win


def _simulate(pnls: list[float], cfg: MonteCarloConfig, rng: random.Random) -> dict[str, float]:
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


def _build_config(regime: str | None = None) -> MonteCarloConfig:
    regime_key = (regime or "unknown").strip().lower()
    defaults = REGIME_DEFAULTS.get(regime_key, REGIME_DEFAULTS["unknown"])
    return MonteCarloConfig(
        min_p05_return_pct=defaults["min_p05_return_pct"],
        max_p95_drawdown_pct=defaults["max_p95_drawdown_pct"],
        max_failure_rate=defaults["max_failure_rate"],
        return_noise_std=defaults["return_noise_std"],
    )


def run_monte_carlo(
    backtest: dict[str, Any],
    *,
    regime: str | None = None,
    iterations: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    pnls = _trade_pnls(backtest)
    if not pnls:
        return {"passed": False, "score": 0.0, "reason": "no_trades", "regime": regime or "unknown"}

    cfg = _build_config(regime)
    if iterations is not None:
        cfg = MonteCarloConfig(
            iterations=max(1, int(iterations)),
            seed=cfg.seed if seed is None else int(seed),
            initial_equity=cfg.initial_equity,
            slippage_bps_mean=cfg.slippage_bps_mean,
            slippage_bps_std=cfg.slippage_bps_std,
            return_noise_std=cfg.return_noise_std,
            min_p05_return_pct=cfg.min_p05_return_pct,
            max_p95_drawdown_pct=cfg.max_p95_drawdown_pct,
            max_failure_rate=cfg.max_failure_rate,
        )
    else:
        cfg = MonteCarloConfig(
            iterations=cfg.iterations,
            seed=cfg.seed if seed is None else int(seed),
            initial_equity=cfg.initial_equity,
            slippage_bps_mean=cfg.slippage_bps_mean,
            slippage_bps_std=cfg.slippage_bps_std,
            return_noise_std=cfg.return_noise_std,
            min_p05_return_pct=cfg.min_p05_return_pct,
            max_p95_drawdown_pct=cfg.max_p95_drawdown_pct,
            max_failure_rate=cfg.max_failure_rate,
        )

    rng = random.Random(cfg.seed)
    sims = [_simulate(pnls, cfg, rng) for _ in range(cfg.iterations)]

    returns = sorted(s["return_pct"] for s in sims)
    dds = sorted(s["max_dd"] for s in sims)

    p05 = returns[int(0.05 * (len(returns) - 1))]
    p50 = returns[int(0.50 * (len(returns) - 1))]
    p95_dd = dds[int(0.95 * (len(dds) - 1))]
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
        "regime": regime or "unknown",
        "config": {
            "iterations": cfg.iterations,
            "min_p05_return_pct": cfg.min_p05_return_pct,
            "max_p95_drawdown_pct": cfg.max_p95_drawdown_pct,
            "max_failure_rate": cfg.max_failure_rate,
            "return_noise_std": cfg.return_noise_std,
        },
        "summary": {
            "p05": p05,
            "p50": p50,
            "p95_dd": p95_dd,
            "failure_rate": failure,
        },
    }
