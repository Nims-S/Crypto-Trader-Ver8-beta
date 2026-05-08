from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from typing import Any

from execution.backtest.core import run_backtest
from registry.store import (
    classify_strategy_status,
    list_strategies,
    record_evolution_run,
    record_experiment,
    upsert_strategy,
)
from research.candidate_generator import mutate_parent, seed_strategy
from research.feedback import build_feedback_summary
from research.monte_carlo import infer_regime_hint, run_monte_carlo
from research.portfolio import build_portfolio_summary
from research.perturbation import run_perturbation
from research.scoring import score_metrics
from research.validation import build_walk_forward_folds, split_walk_forward_window, summarize_walk_forward_reports
from research.regime_evolution import build_regime_plans


@dataclass(frozen=True)
class EvolutionConfig:
    symbols: tuple[str, ...] = ("BTC/USDT",)
    timeframes: tuple[str, ...] = ("1d",)
    validation_symbols: tuple[str, ...] = ("ETH/USDT",)
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


def _cross_symbol_validation(
    *,
    symbols: tuple[str, ...],
    timeframe: str,
    start: str,
    end: str,
    parameters: dict[str, Any],
    allow_shorts: bool,
    use_cache: bool,
) -> dict[str, Any]:
    reports = []
    scores = []

    for symbol in symbols:
        result = _evaluate_variant(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            parameters=parameters,
            allow_shorts=allow_shorts,
            use_cache=use_cache,
        )

        if "error" in result:
            continue

        score = float((result.get("score") or {}).get("score", 0.0))
        passed = bool((result.get("score") or {}).get("passed", False))
        scores.append(score)

        reports.append(
            {
                "symbol": symbol,
                "score": round(score, 6),
                "passed": passed,
                "backtest": result.get("backtest") or {},
            }
        )

    if not reports:
        return {
            "passed": False,
            "score": 0.0,
            "reports": [],
            "reason": "no_cross_symbol_reports",
        }

    mean_score = sum(scores) / len(scores)
    pass_ratio = sum(1 for r in reports if r["passed"]) / len(reports)

    passed = mean_score >= 0.35 and pass_ratio >= 0.5

    return {
        "passed": passed,
        "score": round(mean_score, 6),
        "pass_ratio": round(pass_ratio, 6),
        "reports": reports,
    }


def _persist_evaluation(
    *,
    candidate: Any,
    parent: dict[str, Any],
    report: dict[str, Any],
    symbol: str,
    timeframe: str,
    cycle_id: str,
) -> None:
    metrics = report.get("metrics") or {}
    backtest = metrics.get("backtest") or {}
    mc = metrics.get("monte_carlo") or {}
    perturb = metrics.get("perturbation") or {}
    cross_symbol = metrics.get("cross_symbol") or {}
    wf = metrics.get("walk_forward") or {}

    candidate_id = str(getattr(candidate, "strategy_id", "") or f"evo_{symbol.lower().replace('/', '_')}_{timeframe}_{_stable_seed(cycle_id, symbol, timeframe)}")
    parameters = dict(getattr(candidate, "parameters", {}) or {})
    status = str(report.get("status") or "candidate")
    regime = str(report.get("regime") or infer_regime_hint({"parameters": parameters, "tags": getattr(candidate, "tags", [])}, backtest) or "unknown")
    robustness_score = float(mc.get("score", perturb.get("score", report.get("score", 0.0))) or 0.0)

    upsert_strategy(
        candidate_id,
        base_strategy=str(getattr(candidate, "base_strategy", parent.get("strategy_id") or "seed")),
        version=int(getattr(candidate, "version", 1) or 1),
        status=status,
        parameters=parameters,
        metrics=metrics,
        tags=list(getattr(candidate, "tags", []) or []) + [symbol, timeframe, regime],
        source="evolution",
        notes=f"cycle={cycle_id}",
        active=bool(report.get("passed", False)),
        validated_at=_now(),
        regime_profile=regime,
        robustness_score=robustness_score,
        parent_strategy_id=str(parent.get("strategy_id") or "seed"),
    )

    record_experiment(
        candidate_id,
        symbol=symbol,
        timeframe=timeframe,
        run_type="evolution",
        parameters=parameters,
        metrics=metrics,
        passed=bool(report.get("passed", False)),
        notes=f"cycle={cycle_id}",
    )

    record_evolution_run(
        cycle_id=cycle_id,
        symbol=symbol,
        timeframe=timeframe,
        parent_strategy_id=str(parent.get("strategy_id") or "seed"),
        child_strategy_id=candidate_id,
        status=status,
        score=float(report.get("score", 0.0) or 0.0),
        passed=bool(report.get("passed", False)),
        parameters=parameters,
        metrics=metrics,
        notes=f"regime={regime}; mc_passed={bool(mc.get('passed', False))}; perturb_passed={bool(perturb.get('passed', False))}; cross_symbol_passed={bool(cross_symbol.get('passed', False))}; wf_passed={bool(wf.get('passed', False))}",
    )


def _portfolio_snapshot(*, regime: str = "mean_reversion", limit: int = 3, total_capital: float = 10000.0) -> dict[str, Any]:
    strategies = list_strategies(active_only=False)
    return build_portfolio_summary(
        strategies,
        regime=regime,
        limit=limit,
        unique_markets=True,
        total_capital=total_capital,
    )


def evaluate_candidate(*, candidate: Any, parent: dict[str, Any], symbol: str, timeframe: str, start: str, end: str, folds: int, allow_shorts: bool, use_cache: bool, mc_iterations: int = 300, validation_symbols: tuple[str, ...] = ("ETH/USDT",)) -> dict[str, Any]:
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

    cross_symbol = _cross_symbol_validation(
        symbols=tuple(s for s in validation_symbols if s != symbol),
        timeframe=timeframe,
        start=start,
        end=end,
        parameters=parameters,
        allow_shorts=allow_shorts,
        use_cache=use_cache,
    )

    agent_score = full["score"]

    trend_hardening = True
    if regime == "trend":
        trend_hardening = (
            bool(mc.get("passed"))
            and float((mc.get("summary") or {}).get("failure_rate", 1.0)) <= 0.25
            and float((mc.get("summary") or {}).get("p05", -999.0)) >= -3.0
        )

    passed = (
        bool(agent_score.get("passed"))
        and bool(wf_summary.get("passed"))
        and bool(mc.get("passed"))
        and bool(perturb.get("passed"))
        and bool(cross_symbol.get("passed"))
        and trend_hardening
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
            "cross_symbol": cross_symbol,
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

        plans = build_regime_plans(
            parents,
            symbol=symbol,
            timeframe=timeframe,
            parent_limits={"trend": 2, "breakout": 3, "mean_reversion": 5},
        )

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
                        validation_symbols=config.validation_symbols,
                    )
                    _persist_evaluation(candidate=c, parent=parent, report=report, symbol=symbol, timeframe=timeframe, cycle_id=cycle_id)
                    results.append(report)

    portfolio_summary = _portfolio_snapshot(regime="mean_reversion", limit=3, total_capital=10000.0)

    return {
        "cycle_id": cycle_id,
        "results": results,
        "portfolio_summary": portfolio_summary,
    }


def run_continuous_loop(config: EvolutionConfig, *, interval_seconds: int = 3600, cycles: int | None = None) -> list[dict[str, Any]]:
    out = []
    for _ in range(cycles or 1):
        out.append(run_evolution_cycle(config))
    return out
