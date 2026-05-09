from __future__ import annotations

import random
from dataclasses import dataclass
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
    min_p50_return_pct: float = 0.0
    min_pf_floor: float = 1.0


REGIME_DEFAULTS: dict[str, dict[str, float]] = {
    "trend": {
        "min_p05_return_pct": 0.0,
        "min_p50_return_pct": 0.25,
        "max_p95_drawdown_pct": 16.0,
        "max_failure_rate": 0.18,
        "return_noise_std": 0.035,
        "min_pf_floor": 1.20,
    },
    "breakout": {
        "min_p05_return_pct": 0.0,
        "min_p50_return_pct": 0.10,
        "max_p95_drawdown_pct": 22.0,
        "max_failure_rate": 0.26,
        "return_noise_std": 0.045,
        "min_pf_floor": 1.05,
    },
    # Mean reversion is now tuned for basket survivability rather than forcing
    # every standalone candidate to maintain a fully positive left tail under
    # noisy resampling.
    "mean_reversion": {
        "min_p05_return_pct": -1.25,
        "min_p50_return_pct": 0.15,
        "max_p95_drawdown_pct": 18.0,
        "max_failure_rate": 0.24,
        "return_noise_std": 0.028,
        "min_pf_floor": 1.10,
    },
    "unknown": {
        "min_p05_return_pct": 0.0,
        "min_p50_return_pct": 0.10,
        "max_p95_drawdown_pct": 22.0,
        "max_failure_rate": 0.28,
        "return_noise_std": 0.045,
        "min_pf_floor": 1.0,
    },
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


def _build_config(regime: str | None = None, backtest: dict[str, Any] | None = None) -> MonteCarloConfig:
    regime_key = (regime or "unknown").strip().lower()
    defaults = REGIME_DEFAULTS.get(regime_key, REGIME_DEFAULTS["unknown"])

    bt = backtest or {}
    trades = _safe_int(bt.get("trades", 0), 0)
    pf = _safe_float(bt.get("profit_factor", 0.0), 0.0)
    wr = _safe_float(bt.get("win_rate", 0.0), 0.0)
    dd = abs(_safe_float(bt.get("max_drawdown_pct", 0.0), 0.0))

    quality = 0.0
    quality += 0.30 * max(0.0, min(1.0, (trades - 3) / 10.0))
    quality += 0.30 * max(0.0, min(1.0, (pf - 1.0) / 2.5))
    quality += 0.20 * max(0.0, min(1.0, (wr - 0.45) / 0.30))
    quality += 0.20 * max(0.0, min(1.0, 1.0 - dd / 10.0))
    quality = max(0.0, min(1.0, quality))

    if regime_key == "mean_reversion":
        if quality >= 0.75:
            failure_adjust = -0.03
            drawdown_adjust = -1.0
            p50_adjust = 0.06
            p05_adjust = 0.25
        elif quality >= 0.50:
            failure_adjust = -0.01
            drawdown_adjust = -0.5
            p50_adjust = 0.03
            p05_adjust = 0.15
        else:
            failure_adjust = 0.01
            drawdown_adjust = 0.75
            p50_adjust = 0.0
            p05_adjust = 0.0
    else:
        if quality >= 0.75:
            failure_adjust = -0.05
            drawdown_adjust = -2.0
            p50_adjust = 0.12
        elif quality >= 0.50:
            failure_adjust = -0.02
            drawdown_adjust = -0.75
            p50_adjust = 0.06
        else:
            failure_adjust = 0.01
            drawdown_adjust = 0.5
            p50_adjust = 0.0
        p05_adjust = 0.0

    return MonteCarloConfig(
        min_p05_return_pct=defaults["min_p05_return_pct"] + p05_adjust,
        min_p50_return_pct=defaults["min_p50_return_pct"] + p50_adjust,
        max_p95_drawdown_pct=max(5.0, defaults["max_p95_drawdown_pct"] + drawdown_adjust),
        max_failure_rate=max(0.05, min(0.40, defaults["max_failure_rate"] + failure_adjust)),
        return_noise_std=defaults["return_noise_std"],
        min_pf_floor=defaults["min_pf_floor"],
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

    cfg = _build_config(regime, backtest)
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
            min_p50_return_pct=cfg.min_p50_return_pct,
            min_pf_floor=cfg.min_pf_floor,
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
            min_p50_return_pct=cfg.min_p50_return_pct,
            min_pf_floor=cfg.min_pf_floor,
        )

    rng = random.Random(cfg.seed)
    sims = [_simulate(pnls, cfg, rng) for _ in range(cfg.iterations)]

    returns = sorted(s["return_pct"] for s in sims)
    dds = sorted(s["max_dd"] for s in sims)
    pfs = sorted(s["pf"] for s in sims)

    p05 = returns[int(0.05 * (len(returns) - 1))]
    p50 = returns[int(0.50 * (len(returns) - 1))]
    p95_dd = dds[int(0.95 * (len(dds) - 1))]
    failure = sum(1 for s in sims if s["return_pct"] <= 0) / len(sims)
    pf_p50 = pfs[int(0.50 * (len(pfs) - 1))]

    passed = (
        p05 >= cfg.min_p05_return_pct
        and p50 >= cfg.min_p50_return_pct
        and p95_dd <= cfg.max_p95_drawdown_pct
        and failure <= cfg.max_failure_rate
        and pf_p50 >= cfg.min_pf_floor
    )

    score = (
        0.35 * max(0, p50 / 10)
        + 0.25 * max(0, (p05 + 2.0) / 7)
        + 0.20 * max(0, 1 - p95_dd / cfg.max_p95_drawdown_pct)
        + 0.20 * max(0, min(1.0, (pf_p50 - 0.75) / 2.25))
    )

    return {
        "passed": passed,
        "score": round(score, 4),
        "regime": regime or "unknown",
        "config": {
            "iterations": cfg.iterations,
            "min_p05_return_pct": cfg.min_p05_return_pct,
            "min_p50_return_pct": cfg.min_p50_return_pct,
            "max_p95_drawdown_pct": cfg.max_p95_drawdown_pct,
            "max_failure_rate": cfg.max_failure_rate,
            "min_pf_floor": cfg.min_pf_floor,
            "return_noise_std": cfg.return_noise_std,
        },
        "summary": {
            "p05": p05,
            "p50": p50,
            "p95_dd": p95_dd,
            "pf_p50": pf_p50,
            "failure_rate": failure,
        },
    }
