from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from typing import Any

from execution.backtest.core import run_backtest
from registry.store import classify_strategy_status, list_strategies
from research.candidate_generator import mutate_parent, seed_strategy
from research.feedback import build_feedback_summary
from research.monte_carlo import run_monte_carlo, infer_regime_hint
from research.perturbation import run_perturbation
from research.scoring import score_metrics
from research.validation import (
    build_walk_forward_folds,
    split_walk_forward_window,
    summarize_walk_forward_reports,
)
from research.regime_evolution import build_regime_plans


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
    mc_iterations: int = 300


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_seed(*parts: Any) -> int:
    blob = json.dumps([str(p) for p in parts], sort_keys=True).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:16], 16)


def _evaluate_variant(*, symbol: str, timeframe: str, start: str, end: str, parameters: dict[str, Any], allow_shorts: bool, use_cache: bool) -> dict[str, Any]:
    result = run_backtest(symbol, timeframe, start=start, end=end, allow_shorts=allow_shorts, use_cache=use_cache, strategy_override={"parameters": parameters})
    if "error" in result:
        return result
    decision = score_metrics(result, timeframe=timeframe)
    return {"backtest": result, "score": decision.as_dict()}


def evaluate_candidate(*, candidate: Any, parent: dict[str, Any], symbol: str, timeframe: str, start: str, end: str, folds: int, allow_shorts: bool, use_cache: bool, mc_iterations: int = 300) -> dict[str, Any]:
    parameters = dict(getattr(candidate, "parameters", {}) or {})

    full = _evaluate_variant(symbol=symbol, timeframe=timeframe, start=start, end=end, parameters=parameters, allow_shorts=allow_shorts, use_cache=use_cache)
    if "error" in full:
        return {"status": "candidate", "error": full["error"]}

    wf_reports = []
    for fold in build_walk_forward_folds(start, end, folds=max(1, folds)):
        splits = split_walk_forward_window(fold)

        fold_result = {}
        for split_name, window in splits.items():
            res = _evaluate_variant(
                symbol=symbol,
                timeframe=timeframe,
                start=window["start"],
                end=window["end"],
                parameters=parameters,
                allow_shorts=allow_shorts,
                use_cache=use_cache,
            )

            if "error" not in res:
                fold_result[split_name] = res["backtest"]
            else:
                fold_result[split_name] = {}

        if fold_result:
            wf_reports.append(fold_result)

    wf_summary = summarize_walk_forward_reports(wf_reports, timeframe=timeframe)

    perturb = run_perturbation(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        base_parameters=parameters,
        allow_shorts=allow_shorts,
        use_cache=use_cache,
    )

    regime = infer_regime_hint({"parameters": parameters, "tags": getattr(candidate, "tags", [])}, full["backtest"])
    mc = run_monte_carlo(full["backtest"], regime=regime, iterations=mc_iterations)

    agent_score = full["score"]

    passed = (
        bool(agent_score.get("passed"))
        and bool(wf_summary.get("passed"))
        and bool(mc.get("passed"))
        and bool(perturb.get("passed"))
    )

    if passed:
        final_status = "deployable"
    elif agent_score.get("passed"):
        final_status = "validated"
    else:
        final_status = "candidate"

    return {
        "status": final_status,
        "metrics": {
            "backtest": full["backtest"],
            "agent_score": agent_score,
            "walk_forward": wf_summary,
            "monte_carlo": mc,
            "perturbation": perturb,
        },
        "score": agent_score.get("score", 0.0),
        "regime": regime,
        "passed": passed,
        "logic_hash": hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:16],
    }


def run_evolution_cycle(config: EvolutionConfig, *, cycle_id: str | None = None) -> dict[str, Any]:
    cycle_id = cycle_id or f"cycle_{uuid.uuid4().hex[:8]}"
    results = []

    for symbol, timeframe in product(config.symbols, config.timeframes):
        feedback = build_feedback_summary(symbol=symbol, timeframe=timeframe)
        parents = list_strategies(active_only=False)[: config.parents_per_pair]

        plans = build_regime_plans(parents, symbol=symbol, timeframe=timeframe)

        for plan in plans:
            plan_feedback = dict(feedback or {})
            plan_feedback["mutation_directives"] = plan.directives

            selected_parents = [p for p in parents if (p.get("strategy_id") in plan.parent_ids)]

            if not selected_parents:
                seed = seed_strategy(symbol, timeframe)
                selected_parents = [{"strategy_id": seed.strategy_id, "parameters": seed.parameters}]

            for parent in selected_parents:
                candidates = mutate_parent(
                    parent,
                    symbol,
                    timeframe,
                    n_children=config.children_per_parent,
                    seed=_stable_seed(symbol, timeframe, plan.regime),
                    feedback=plan_feedback,
                    diversity_pool=parents,
                )

                for c in candidates:
                    report = evaluate_candidate(
                        candidate=c,
                        parent=parent,
                        symbol=symbol,
                        timeframe=timeframe,
                        start=config.start,
                        end=config.end,
                        folds=config.folds,
                        allow_shorts=config.allow_shorts,
                        use_cache=config.use_cache,
                        mc_iterations=config.mc_iterations,
                    )
                    results.append(report)

    return {"cycle_id": cycle_id, "results": results}


def run_continuous_loop(config: EvolutionConfig, *, interval_seconds: int = 3600, cycles: int | None = None) -> list[dict[str, Any]]:
    out = []
    for _ in range(cycles or 1):
        out.append(run_evolution_cycle(config))
    return out
