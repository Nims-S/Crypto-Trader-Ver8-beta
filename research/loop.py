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


def evaluate_candidate(*, candidate: Any, parent: dict[str, Any], symbol: str, timeframe: str, start: str, end: str, folds: int, allow_shorts: bool, use_cache: bool) -> dict[str, Any]:
    parameters = dict(getattr(candidate, "parameters", {}) or {})

    full = _evaluate_variant(symbol=symbol, timeframe=timeframe, start=start, end=end, parameters=parameters, allow_shorts=allow_shorts, use_cache=use_cache)
    if "error" in full:
        return {"status": "candidate", "error": full["error"]}

    # Walk-forward
    wf_reports = []
    for fold in build_walk_forward_folds(start, end, folds=max(1, folds)):
        train = _evaluate_variant(symbol=symbol, timeframe=timeframe, start=fold.start, end=fold.end, parameters=parameters, allow_shorts=allow_shorts, use_cache=use_cache)
        if "error" not in train:
            wf_reports.append(train["backtest"])
    wf_summary = summarize_walk_forward_reports(wf_reports, timeframe=timeframe)

    # Infer regime (key upgrade)
    regime = infer_regime_hint({"parameters": parameters, "tags": getattr(candidate, "tags", [])}, full["backtest"])

    # Regime-aware Monte Carlo
    mc = run_monte_carlo(full["backtest"], regime=regime)

    agent_score = full["score"]
    base_status = classify_strategy_status(agent_score=agent_score, backtest=full["backtest"], walk_forward=wf_summary, timeframe=timeframe)

    # Promotion logic (strict + regime-aware)
    passed = bool(agent_score.get("passed")) and bool(wf_summary.get("passed")) and bool(mc.get("passed"))

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

        if not parents:
            seed = seed_strategy(symbol, timeframe)
            parents = [{"strategy_id": seed.strategy_id, "parameters": seed.parameters}]

        for parent in parents:
            candidates = mutate_parent(parent, symbol, timeframe, n_children=config.children_per_parent, seed=_stable_seed(symbol, timeframe), feedback=feedback, diversity_pool=parents)

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
                )
                results.append(report)

    return {"cycle_id": cycle_id, "results": results}


def run_continuous_loop(config: EvolutionConfig, *, interval_seconds: int = 3600, cycles: int | None = None) -> list[dict[str, Any]]:
    out = []
    for _ in range(cycles or 1):
        out.append(run_evolution_cycle(config))
    return out
