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
    if not isinstance(parent, dict):
        return {}
    return {
        "strategy_id": parent.get("strategy_id") or parent.get("id") or "seed",
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


def _range_bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    if end_ts <= start_ts:
        raise ValueError("end must be greater than start")
    return start_ts, end_ts


def _split_ratio_window(start: str, end: str) -> dict[str, dict[str, str]]:
    start_ts, end_ts = _range_bounds(start, end)
    span = end_ts - start_ts
    train_end = start_ts + (span * 0.6)
    val_end = start_ts + (span * 0.8)
    if train_end >= val_end:
        val_end = train_end + (span * 0.1)
    if val_end >= end_ts:
        val_end = end_ts - pd.Timedelta(minutes=1)
    return {
        "train": {"start": start_ts.isoformat(), "end": train_end.isoformat()},
        "val": {"start": train_end.isoformat(), "end": val_end.isoformat()},
        "test": {"start": val_end.isoformat(), "end": end_ts.isoformat()},
    }


def _perturb_params(params: dict[str, Any], seed: int) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    p = copy.deepcopy(params or {})
    if "stop_atr_mult" in p:
        p["stop_atr_mult"] = max(1.0, _safe_float(p["stop_atr_mult"], 2.0) + rng.uniform(-0.15, 0.2))
    if "tp1_rr" in p:
        p["tp1_rr"] = max(1.0, _safe_float(p["tp1_rr"], 2.0) + rng.uniform(-0.2, 0.2))
    if "tp2_rr" in p:
        p["tp2_rr"] = max(_safe_float(p.get("tp1_rr", 2.0), 2.0) + 0.4, _safe_float(p["tp2_rr"], 3.0) + rng.uniform(-0.3, 0.3))
    if "cooldown_bars" in p:
        p["cooldown_bars"] = max(4, int(round(_safe_float(p["cooldown_bars"], 16) + rng.choice([-3, 0, 3]))))
    if "max_bars_override" in p:
        p["max_bars_override"] = max(8, int(round(_safe_float(p["max_bars_override"], 48) + rng.choice([-6, 0, 6]))))
    if "size_multiplier" in p:
        p["size_multiplier"] = max(0.2, min(1.0, _safe_float(p["size_multiplier"], 1.0) * rng.uniform(0.85, 1.05)))
    if "confidence" in p:
        p["confidence"] = max(0.3, min(0.99, _safe_float(p["confidence"], 0.6) + rng.uniform(-0.05, 0.05)))
    return p


def _evaluate_variant(
    *,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
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


def _robustness_score(
    *,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    parameters: dict[str, Any],
    trials: int,
    allow_shorts: bool,
    use_cache: bool,
) -> dict[str, Any]:
    base = _evaluate_variant(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        parameters=parameters,
        allow_shorts=allow_shorts,
        use_cache=use_cache,
    )
    if "error" in base:
        return {"score": 0.0, "passed": False, "reasons": ["base_backtest_failed"], "trials": []}

    scores = [base["score"]["score"]]
    trial_rows = [{"name": "base", "score": base["score"]["score"], "metrics": base["backtest"]}]
    for idx in range(max(0, trials - 1)):
        mutated = _perturb_params(parameters, _stable_seed(symbol, timeframe, start, end, idx))
        trial = _evaluate_variant(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            parameters=mutated,
            allow_shorts=allow_shorts,
            use_cache=use_cache,
        )
        if "error" in trial:
            continue
        scores.append(trial["score"]["score"])
        trial_rows.append({"name": f"mut_{idx + 1}", "score": trial["score"]["score"], "metrics": trial["backtest"]})

    if not scores:
        return {"score": 0.0, "passed": False, "reasons": ["no_scores"], "trials": trial_rows}

    series = pd.Series(scores, dtype="float64")
    mean = float(series.mean())
    std = float(series.std(ddof=0)) if len(series) > 1 else 0.0
    final = max(0.0, min(1.0, mean - (0.5 * std)))
    passed = final >= 0.45 and std <= 0.20
    reasons = []
    if final < 0.45:
        reasons.append("robustness_low")
    if std > 0.20:
        reasons.append("robustness_unstable")
    return {"score": round(final, 6), "passed": passed, "reasons": reasons, "mean": round(mean, 6), "std": round(std, 6), "trials": trial_rows}


def _promotion_status(base_status: str, robustness: float, full_score: float) -> str:
    status = str(base_status or "candidate")
    if status in {"live", "deployable"}:
        return status
    if robustness >= 0.65 and full_score >= 0.40:
        return "deployable"
    if robustness >= 0.50 and full_score >= 0.30:
        return "validated"
    return "candidate"


def evaluate_candidate(
    *,
    candidate: Any,
    parent: dict[str, Any],
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    folds: int,
    allow_shorts: bool,
    use_cache: bool,
    perturbation_trials: int,
    min_deployable_robustness: float,
    min_validated_robustness: float,
) -> dict[str, Any]:
    parameters = dict(getattr(candidate, "parameters", {}) or {})
    candidate_id = str(getattr(candidate, "strategy_id", "unknown"))
    parent_id = str(parent.get("strategy_id") or "seed")
    seed = _stable_seed(candidate_id, symbol, timeframe, start, end)

    full = _evaluate_variant(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        parameters=parameters,
        allow_shorts=allow_shorts,
        use_cache=use_cache,
    )
    if "error" in full:
        return {
            "candidate_id": candidate_id,
            "parent_id": parent_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "candidate",
            "error": full["error"],
        }

    fold_reports: list[dict[str, Any]] = []
    wf_folds = build_walk_forward_folds(start, end, folds=max(1, int(folds)))
    for fold in wf_folds:
        window = _split_ratio_window(fold.start, fold.end)
        fold_report = {
            "train": _evaluate_variant(
                symbol=symbol,
                timeframe=timeframe,
                start=window["train"]["start"],
                end=window["train"]["end"],
                parameters=parameters,
                allow_shorts=allow_shorts,
                use_cache=use_cache,
            ),
            "val": _evaluate_variant(
                symbol=symbol,
                timeframe=timeframe,
                start=window["val"]["start"],
                end=window["val"]["end"],
                parameters=parameters,
                allow_shorts=allow_shorts,
                use_cache=use_cache,
            ),
            "test": _evaluate_variant(
                symbol=symbol,
                timeframe=timeframe,
                start=window["test"]["start"],
                end=window["test"]["end"],
                parameters=parameters,
                allow_shorts=allow_shorts,
                use_cache=use_cache,
            ),
        }
        if all("error" not in split for split in fold_report.values()):
            fold_reports.append(
                {
                    "train": fold_report["train"]["backtest"],
                    "val": fold_report["val"]["backtest"],
                    "test": fold_report["test"]["backtest"],
                }
            )

    wf_summary = summarize_walk_forward_reports(fold_reports, timeframe=timeframe)
    robustness = _robustness_score(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        parameters=parameters,
        trials=perturbation_trials,
        allow_shorts=allow_shorts,
        use_cache=use_cache,
    )

    agent_score = full["score"]
    base_status = classify_strategy_status(
        agent_score=agent_score,
        backtest=full["backtest"],
        walk_forward=wf_summary,
        timeframe=timeframe,
    )
    final_status = _promotion_status(base_status["status"], robustness["score"], agent_score["score"])
    if robustness["score"] < min_validated_robustness:
        final_status = "candidate"
    elif robustness["score"] < min_deployable_robustness and final_status == "deployable":
        final_status = "validated"

    metrics = {
        "backtest": full["backtest"],
        "agent_score": agent_score,
        "walk_forward": wf_summary,
        "robustness": robustness,
        "selected_status": final_status,
        "seed": seed,
    }
    return {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "status": final_status,
        "metrics": metrics,
        "score": agent_score["score"],
        "robustness_score": robustness["score"],
        "passed": bool(base_status["score_passed"]) and bool(wf_summary.get("passed")) and bool(robustness["passed"]),
        "logic_hash": hashlib.sha256(json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16],
    }


def _candidate_tags(symbol: str, timeframe: str, status: str, parameters: dict[str, Any]) -> list[str]:
    tags = {symbol, timeframe, status}
    entry_mode = str(parameters.get("entry_mode") or "").strip()
    if entry_mode:
        tags.add(entry_mode)
    if symbol.startswith("BTC"):
        tags.add("btc")
    return sorted(tags)


def persist_candidate(
    *,
    candidate: Any,
    parent: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    parameters = dict(getattr(candidate, "parameters", {}) or {})
    candidate_id = str(getattr(candidate, "strategy_id", "unknown"))
    symbol = str(report.get("symbol") or getattr(candidate, "symbol", "BTC/USDT"))
    timeframe = str(report.get("timeframe") or getattr(candidate, "timeframe", "1d"))
    status = str(report.get("status") or "candidate")
    metrics = report.get("metrics") or {}

    upsert_strategy(
        candidate_id,
        base_strategy=str(getattr(candidate, "base_strategy", parent.get("strategy_id") or "seed")),
        version=int(getattr(candidate, "version", 1) or 1),
        status=status,
        parameters=parameters,
        metrics=metrics,
        tags=_candidate_tags(symbol, timeframe, status, parameters),
        source=str(getattr(candidate, "source", "evolution")),
        notes=str(getattr(candidate, "notes", "")),
        active=status in {"deployable", "live"},
        validated_at=_now(),
        regime_profile=str(report.get("metrics", {}).get("walk_forward", {}).get("regime", "") or ""),
        robustness_score=_safe_float(report.get("robustness_score", 0.0), 0.0),
        parent_strategy_id=str(parent.get("strategy_id") or "seed"),
    )

    record_experiment(
        candidate_id,
        symbol=symbol,
        timeframe=timeframe,
        run_type="autonomous_evolution",
        parameters=parameters,
        metrics=metrics,
        passed=bool(report.get("passed", False)),
        notes=f"status={status}",
    )

    record_evolution_run(
        cycle_id=str(report.get("cycle_id") or "cycle"),
        symbol=symbol,
        timeframe=timeframe,
        parent_strategy_id=str(parent.get("strategy_id") or "seed"),
        child_strategy_id=candidate_id,
        status=status,
        score=_safe_float(report.get("score", 0.0), 0.0),
        passed=bool(report.get("passed", False)),
        parameters=parameters,
        metrics=metrics,
        notes=f"robustness={report.get('robustness_score', 0.0)}",
    )
    return report


def run_evolution_cycle(
    config: EvolutionConfig,
    *,
    cycle_id: str | None = None,
) -> dict[str, Any]:
    cycle_id = cycle_id or f"cycle_{uuid.uuid4().hex[:12]}"
    started_at = _now()
    all_results: list[dict[str, Any]] = []

    for symbol, timeframe in product(config.symbols, config.timeframes):
        feedback = build_feedback_summary(symbol=symbol, timeframe=timeframe)
        parents = list_strategies(active_only=False)
        parents = [
            p for p in parents
            if symbol in {str(tag).lower() for tag in (p.get("tags") or [])}
            or timeframe in {str(tag).lower() for tag in (p.get("tags") or [])}
        ]
        parents = parents[: max(1, int(config.parents_per_pair))]

        if not parents:
            seed = seed_strategy(symbol, timeframe, family="autonomous")
            parents = [_strategy_parent_row({
                "strategy_id": seed.strategy_id,
                "base_strategy": seed.base_strategy,
                "version": seed.version,
                "parameters": seed.parameters,
                "tags": seed.tags,
                "status": "candidate",
            })]

        for parent in parents:
            parent_row = _strategy_parent_row(parent)
            candidates = mutate_parent(
                parent_row,
                symbol,
                timeframe,
                n_children=config.children_per_parent,
                seed=_stable_seed(cycle_id, symbol, timeframe, parent_row.get("strategy_id")),
                feedback=feedback,
                diversity_pool=parents,
            )

            for candidate in candidates:
                report = evaluate_candidate(
                    candidate=candidate,
                    parent=parent_row,
                    symbol=symbol,
                    timeframe=timeframe,
                    start=config.start,
                    end=config.end,
                    folds=config.folds,
                    allow_shorts=config.allow_shorts,
                    use_cache=config.use_cache,
                    perturbation_trials=config.perturbation_trials,
                    min_deployable_robustness=config.min_deployable_robustness,
                    min_validated_robustness=config.min_validated_robustness,
                )
                report["cycle_id"] = cycle_id
                persisted = persist_candidate(candidate=candidate, parent=parent_row, report=report)
                all_results.append(persisted)

    status_counts: dict[str, int] = {}
    for row in all_results:
        status = str(row.get("status") or "candidate")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "cycle_id": cycle_id,
        "started_at": started_at,
        "finished_at": _now(),
        "symbols": list(config.symbols),
        "timeframes": list(config.timeframes),
        "results": all_results,
        "status_counts": status_counts,
        "total_candidates": len(all_results),
    }


def run_continuous_loop(
    config: EvolutionConfig,
    *,
    interval_seconds: int = 3600,
    cycles: int | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    cycle_index = 0
    while True:
        cycle_index += 1
        result = run_evolution_cycle(config, cycle_id=f"cycle_{cycle_index:04d}")
        results.append(result)
        if cycles is not None and cycle_index >= cycles:
            break
        if interval_seconds > 0:
            import time
            time.sleep(interval_seconds)
    return results
