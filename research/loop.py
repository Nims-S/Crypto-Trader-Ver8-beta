from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from typing import Any

import pandas as pd

from execution.backtest.core import run_backtest
from registry.store import classify_strategy_status, list_strategies, record_experiment, record_evolution_run, upsert_strategy
from research.candidate_generator import mutate_parent, seed_strategy
from research.feedback import build_feedback_summary
from research.monte_carlo import run_monte_carlo
from research.scoring import score_metrics
from research.validation import build_walk_forward_folds, summarize_walk_forward_reports


@dataclass(frozen=True)
class EvolutionConfig:
    symbols: tuple[str, ...] = ("BTC/USDT",)
    timeframes: tuple[str, ...] = ("1d",)
    start: str = "2024-01-01"
    end: str = "2025-01-01"
    folds: int = 3
    parents_per_pair: int = 3
    children_per_parent: int = 3
    use_cache: bool = True
    allow_shorts: bool = False
    retain_top_n: int = 10
    min_deployable_robustness: float = 0.65
    min_validated_robustness: float = 0.50
    perturbation_trials: int = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_seed(*parts: Any) -> int:
    blob = json.dumps([str(p) for p in parts], sort_keys=True).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:16], 16)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _strategy_parent_row(parent: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": parent.get("strategy_id") or "seed",
        "base_strategy": parent.get("base_strategy") or parent.get("strategy_id") or "seed",
        "version": int(parent.get("version", 1) or 1),
        "status": parent.get("status") or "candidate",
        "parameters": parent.get("parameters") or {},
        "metrics": parent.get("metrics") or {},
        "tags": parent.get("tags") or [],
        "logic_hash": parent.get("logic_hash"),
        "regime_profile": parent.get("regime_profile"),
        "robustness_score": _safe_float(parent.get("robustness_score", 0.0), 0.0),
        "active": bool(parent.get("active", False)),
        "parent_strategy_id": parent.get("parent_strategy_id"),
    }


def _evaluate_variant(*, symbol: str, timeframe: str, start: str, end: str, parameters: dict[str, Any], allow_shorts: bool, use_cache: bool) -> dict[str, Any]:
    result = run_backtest(symbol, timeframe, start=start, end=end, allow_shorts=allow_shorts, use_cache=use_cache, strategy_override={"parameters": parameters})
    if "error" in result:
        return result
    decision = score_metrics(result, timeframe=timeframe)
    return {"backtest": result, "score": decision.as_dict()}


def _robustness_score(*, symbol: str, timeframe: str, start: str, end: str, parameters: dict[str, Any], trials: int, allow_shorts: bool, use_cache: bool) -> dict[str, Any]:
    base = _evaluate_variant(symbol=symbol, timeframe=timeframe, start=start, end=end, parameters=parameters, allow_shorts=allow_shorts, use_cache=use_cache)
    if "error" in base:
        return {"score": 0.0, "passed": False}

    scores = [base["score"]["score"]]
    for i in range(max(0, trials - 1)):
        mutated = copy.deepcopy(parameters)
        mutated["stop_atr_mult"] = max(1.0, mutated.get("stop_atr_mult", 2.0) * (0.9 + 0.2 * i))
        trial = _evaluate_variant(symbol=symbol, timeframe=timeframe, start=start, end=end, parameters=mutated, allow_shorts=allow_shorts, use_cache=use_cache)
        if "error" not in trial:
            scores.append(trial["score"]["score"])

    series = pd.Series(scores)
    mean = float(series.mean())
    std = float(series.std()) if len(series) > 1 else 0.0
    score = max(0.0, mean - 0.5 * std)
    return {"score": score, "passed": score > 0.45}


def _promotion_status(base_status: str, robustness: float, full_score: float) -> str:
    if robustness >= 0.65 and full_score >= 0.4:
        return "deployable"
    if robustness >= 0.5 and full_score >= 0.3:
        return "validated"
    return "candidate"


def evaluate_candidate(*, candidate: Any, parent: dict[str, Any], symbol: str, timeframe: str, start: str, end: str, folds: int, allow_shorts: bool, use_cache: bool, perturbation_trials: int, min_deployable_robustness: float, min_validated_robustness: float) -> dict[str, Any]:
    parameters = dict(getattr(candidate, "parameters", {}) or {})

    full = _evaluate_variant(symbol=symbol, timeframe=timeframe, start=start, end=end, parameters=parameters, allow_shorts=allow_shorts, use_cache=use_cache)
    if "error" in full:
        return {"status": "candidate", "error": full["error"]}

    wf_reports = []
    for fold in build_walk_forward_folds(start, end, folds=max(1, folds)):
        train = _evaluate_variant(symbol=symbol, timeframe=timeframe, start=fold.start, end=fold.end, parameters=parameters, allow_shorts=allow_shorts, use_cache=use_cache)
        if "error" not in train:
            wf_reports.append(train["backtest"])

    wf_summary = summarize_walk_forward_reports(wf_reports, timeframe=timeframe)

    robustness = _robustness_score(symbol=symbol, timeframe=timeframe, start=start, end=end, parameters=parameters, trials=perturbation_trials, allow_shorts=allow_shorts, use_cache=use_cache)

    mc = run_monte_carlo(full["backtest"])

    agent_score = full["score"]

    base_status = classify_strategy_status(agent_score=agent_score, backtest=full["backtest"], walk_forward=wf_summary, timeframe=timeframe)

    final_status = _promotion_status(base_status["status"], robustness["score"], agent_score["score"])

    if not mc["passed"]:
        final_status = "candidate"

    if robustness["score"] < min_validated_robustness:
        final_status = "candidate"
    elif robustness["score"] < min_deployable_robustness and final_status == "deployable":
        final_status = "validated"

    return {
        "status": final_status,
        "metrics": {
            "backtest": full["backtest"],
            "agent_score": agent_score,
            "walk_forward": wf_summary,
            "robustness": robustness,
            "monte_carlo": mc,
        },
        "score": agent_score["score"],
        "robustness_score": robustness["score"],
        "passed": base_status["score_passed"] and wf_summary.get("passed") and robustness["passed"] and mc["passed"],
        "logic_hash": hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:16],
    }


def run_evolution_cycle(config: EvolutionConfig, *, cycle_id: str | None = None) -> dict[str, Any]:
    cycle_id = cycle_id or f"cycle_{uuid.uuid4().hex[:8]}"
    results = []

    for symbol, timeframe in product(config.symbols, config.timeframes):
        feedback = build_feedback_summary(symbol=symbol, timeframe=timeframe)
        parents = list_strategies(active_only=False)[: config.parents_per_pair]

        if not parents:
            seed = seed_strategy(symbol, timeframe)
            parents = [{"strategy_id": seed.strategy_id, "parameters": seed.parameters}]

        for parent in parents:
            candidates = mutate_parent(parent, symbol, timeframe, n_children=config.children_per_parent, seed=_stable_seed(symbol, timeframe), feedback=feedback, diversity_pool=parents)

            for c in candidates:
                report = evaluate_candidate(candidate=c, parent=parent, symbol=symbol, timeframe=timeframe, start=config.start, end=config.end, folds=config.folds, allow_shorts=config.allow_shorts, use_cache=config.use_cache, perturbation_trials=config.perturbation_trials, min_deployable_robustness=config.min_deployable_robustness, min_validated_robustness=config.min_validated_robustness)
                results.append(report)

    return {"cycle_id": cycle_id, "results": results}


def run_continuous_loop(config: EvolutionConfig, *, interval_seconds: int = 3600, cycles: int | None = None) -> list[dict[str, Any]]:
    out = []
    for i in range(cycles or 1):
        out.append(run_evolution_cycle(config))
    return out
