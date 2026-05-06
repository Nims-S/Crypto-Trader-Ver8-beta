from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from execution.backtest.core import run_backtest
from research.scoring import score_metrics


@dataclass(frozen=True)
class PerturbationConfig:
    iterations: int = 12
    seed: int = 17
    min_pass_ratio: float = 0.34
    min_mean_score: float = 0.25
    max_score_spread: float = 0.80


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


def _copy_params(params: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(params or {})


def _tweak_numeric(rng: random.Random, value: Any, *, pct: float = 0.10, min_value: float | None = None, max_value: float | None = None) -> float:
    base = _safe_float(value, 0.0)
    delta = base * rng.uniform(-pct, pct)
    adjusted = base + delta
    if min_value is not None:
        adjusted = max(min_value, adjusted)
    if max_value is not None:
        adjusted = min(max_value, adjusted)
    return adjusted


def _perturb_parameters(params: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    p = _copy_params(params)

    # Core trend / breakout / mean-reversion knobs used across the repo.
    for key in (
        "stop_atr_mult",
        "tp1_rr",
        "tp2_rr",
        "trail_atr_mult",
        "confidence",
        "size_multiplier",
        "volume_multiplier",
        "min_atr_rank",
        "min_bb_rank",
        "htf_bb_rank_min",
        "htf_adx_min",
        "min_adx",
        "rsi_min",
        "rsi_max",
    ):
        if key in p and isinstance(p[key], (int, float)):
            if key in {"tp1_rr", "tp2_rr", "stop_atr_mult", "trail_atr_mult"}:
                p[key] = round(_tweak_numeric(rng, p[key], pct=0.12, min_value=0.1), 6)
            elif key == "confidence":
                p[key] = round(_tweak_numeric(rng, p[key], pct=0.08, min_value=0.1, max_value=0.99), 6)
            elif key == "size_multiplier":
                p[key] = round(_tweak_numeric(rng, p[key], pct=0.10, min_value=0.1, max_value=1.0), 6)
            elif key in {"volume_multiplier", "min_atr_rank", "min_bb_rank", "htf_bb_rank_min", "htf_adx_min", "min_adx"}:
                p[key] = round(max(0.0, _tweak_numeric(rng, p[key], pct=0.15)), 6)
            elif key == "rsi_min":
                p[key] = round(max(0.0, min(99.0, _tweak_numeric(rng, p[key], pct=0.10))), 6)
            elif key == "rsi_max":
                p[key] = round(max(0.0, min(99.0, _tweak_numeric(rng, p[key], pct=0.10))), 6)

    # Integer / discrete controls
    for key in ("cooldown_bars", "pullback_bars", "pullback_lookback", "max_bars_override"):
        if key in p:
            base = _safe_int(p[key], 0)
            delta = rng.choice([-2, -1, 0, 1, 2])
            p[key] = max(1, base + delta)

    # Boolean toggles: flip lightly, but not all at once.
    for key in (
        "use_htf_filter",
        "use_trend_filter",
        "use_breakout_filter",
        "use_structure_filter",
        "use_reclaim_filter",
        "use_volume_filter",
        "trail_ema20",
    ):
        if key in p and rng.random() < 0.15:
            p[key] = not bool(p[key])

    return p


def _evaluate(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    *,
    parameters: dict[str, Any],
    allow_shorts: bool,
    use_cache: bool,
) -> dict[str, Any]:
    result = run_backtest(
        symbol,
        timeframe,
        start=start,
        end=end,
        allow_shorts=allow_shorts,
        use_cache=use_cache,
        strategy_override={"parameters": parameters},
    )
    if "error" in result:
        return result
    decision = score_metrics(result, timeframe=timeframe)
    return {"backtest": result, "score": decision.as_dict()}


def run_perturbation(
    *,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    base_parameters: dict[str, Any],
    allow_shorts: bool,
    use_cache: bool,
    iterations: int = 12,
    seed: int = 17,
    min_pass_ratio: float = 0.34,
    min_mean_score: float = 0.25,
    max_score_spread: float = 0.80,
) -> dict[str, Any]:
    cfg = PerturbationConfig(
        iterations=max(1, int(iterations)),
        seed=int(seed),
        min_pass_ratio=float(min_pass_ratio),
        min_mean_score=float(min_mean_score),
        max_score_spread=float(max_score_spread),
    )

    rng = random.Random(cfg.seed)
    base_eval = _evaluate(
        symbol,
        timeframe,
        start,
        end,
        parameters=base_parameters,
        allow_shorts=allow_shorts,
        use_cache=use_cache,
    )
    if "error" in base_eval:
        return {
            "score": 0.0,
            "passed": False,
            "reasons": ["base_backtest_failed"],
            "summary": {},
            "samples": [],
        }

    samples: list[dict[str, Any]] = []
    scores: list[float] = []
    pass_count = 0
    for i in range(cfg.iterations):
        if i == 0:
            params = _copy_params(base_parameters)
        else:
            params = _perturb_parameters(base_parameters, rng)
        result = _evaluate(
            symbol,
            timeframe,
            start,
            end,
            parameters=params,
            allow_shorts=allow_shorts,
            use_cache=use_cache,
        )
        if "error" in result:
            continue
        score = float(result["score"]["score"])
        scores.append(score)
        passed = bool(result["score"].get("passed"))
        if passed:
            pass_count += 1
        samples.append(
            {
                "index": i,
                "score": round(score, 6),
                "passed": passed,
                "metrics": result["backtest"],
                "parameters": params,
            }
        )

    if not scores:
        return {
            "score": 0.0,
            "passed": False,
            "reasons": ["no_valid_samples"],
            "summary": {},
            "samples": samples,
        }

    sorted_scores = sorted(scores)
    p05_idx = max(0, int(round(0.05 * (len(sorted_scores) - 1))))
    p50_idx = max(0, int(round(0.50 * (len(sorted_scores) - 1))))
    p95_idx = max(0, int(round(0.95 * (len(sorted_scores) - 1))))
    p05 = float(sorted_scores[p05_idx])
    p50 = float(sorted_scores[p50_idx])
    p95 = float(sorted_scores[p95_idx])
    spread = float(max(scores) - min(scores)) if len(scores) > 1 else 0.0
    mean_score = float(mean(scores))
    std_score = float(pstdev(scores)) if len(scores) > 1 else 0.0
    pass_ratio = pass_count / max(1, len(scores))

    # Penalize instability and wide spread.
    score = max(0.0, min(1.0, (0.45 * mean_score) + (0.35 * p50) + (0.20 * p05) - (0.15 * std_score)))
    passed = (
        pass_ratio >= cfg.min_pass_ratio
        and mean_score >= cfg.min_mean_score
        and spread <= cfg.max_score_spread
        and p05 >= 0.0
    )

    reasons = []
    if pass_ratio < cfg.min_pass_ratio:
        reasons.append("pass_ratio_low")
    if mean_score < cfg.min_mean_score:
        reasons.append("mean_score_low")
    if spread > cfg.max_score_spread:
        reasons.append("score_spread_high")
    if p05 < 0.0:
        reasons.append("tail_score_negative")

    return {
        "score": round(score, 6),
        "passed": passed,
        "reasons": reasons,
        "summary": {
            "mean_score": round(mean_score, 6),
            "std_score": round(std_score, 6),
            "p05_score": round(p05, 6),
            "p50_score": round(p50, 6),
            "p95_score": round(p95, 6),
            "score_spread": round(spread, 6),
            "pass_ratio": round(pass_ratio, 6),
            "iterations": len(scores),
        },
        "samples": samples,
    }


def perturbation_signature(parameters: dict[str, Any]) -> str:
    return hashlib.sha256(str(sorted(parameters.items())).encode("utf-8")).hexdigest()[:16]
