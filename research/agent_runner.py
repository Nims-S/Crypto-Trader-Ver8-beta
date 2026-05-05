from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from execution.backtest.core import run_backtest
from registry.store import (
    record_evolution_run,
    upsert_strategy,
    rank_strategies,
    classify_strategy_status,
)
from research.agent_scoring import AgentScore, score_candidate
from research.candidate_generator import StrategyCandidate, mutate_parent, seed_strategy
from research.feedback import build_feedback_summary
from research.validation import build_walk_forward_folds, summarize_walk_forward_reports
from research.llm_client import get_default_llm_client


@dataclass(frozen=True)
class AgentConfig:
    symbol: str
    timeframe: str
    start: str
    end: str
    goal_return: float = 0.25
    max_dd: float = 15.0
    iterations: int = 100
    candidates: int = 5
    folds: int = 3
    workers: int = 4
    continuous: bool = False
    sleep_seconds: float = 1.0


@dataclass(frozen=True)
class CandidateResult:
    candidate: StrategyCandidate
    backtest: dict[str, Any]
    walk_forward: dict[str, Any]
    monte_carlo: dict[str, Any]
    score: AgentScore
    parent_id: str
    iteration: int


ELITE_STATUSES = {"validated", "deployable", "live"}


def _normalize_parent(row: Any, symbol: str, timeframe: str) -> dict[str, Any]:
    if row is None:
        seed = seed_strategy(symbol, timeframe)
        return asdict(seed)
    if isinstance(row, dict):
        return row
    try:
        return asdict(row)
    except Exception:
        return {
            "strategy_id": getattr(row, "strategy_id", "seed"),
            "base_strategy": getattr(row, "base_strategy", "seed"),
            "version": int(getattr(row, "version", 1) or 1),
            "parameters": getattr(row, "parameters", {}) or {},
            "symbol": symbol,
            "timeframe": timeframe,
            "tags": [symbol, timeframe, "seed"],
            "source": "seed",
        }


def _composite_parent_score(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    agent = metrics.get("agent_score") or {}
    wf = metrics.get("walk_forward") or {}
    bt = metrics.get("backtest") or {}
    robustness = float(row.get("robustness_score", 0.0) or 0.0)
    return (
        0.45 * float(agent.get("score", 0.0) or 0.0)
        + 0.30 * float(wf.get("score", 0.0) or 0.0)
        + 0.15 * min(max(0.0, float(bt.get("return_pct", 0.0) or 0.0)) / 2.0, 1.0)
        + 0.10 * robustness
    )


def _elite_ranked(cfg: AgentConfig, limit: int = 20) -> list[dict[str, Any]]:
    ranked = rank_strategies(symbol=cfg.symbol, timeframe=cfg.timeframe, active_only=False, limit=limit)
    elite = [r for r in ranked if str(r.get("status") or "").lower() in ELITE_STATUSES]
    return elite or ranked


def _choose_parent(cfg: AgentConfig, prev_parent_id: str | None = None) -> dict[str, Any]:
    pool = _elite_ranked(cfg, limit=20)
    if not pool:
        return _normalize_parent(None, cfg.symbol, cfg.timeframe)

    if prev_parent_id:
        alt_pool = [r for r in pool if str(r.get("strategy_id")) != str(prev_parent_id)]
        if alt_pool:
            pool = alt_pool

    pool.sort(key=lambda r: (_composite_parent_score(r), float(r.get("robustness_score", 0.0) or 0.0)), reverse=True)
    return _normalize_parent(pool[0], cfg.symbol, cfg.timeframe)


def _split_window(start: str, end: str) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    span = e - s
    train_end = s + span * 0.6
    val_end = train_end + span * 0.2
    return (
        (s.isoformat(), train_end.isoformat()),
        (train_end.isoformat(), val_end.isoformat()),
        (val_end.isoformat(), e.isoformat()),
    )


def _extract_trade_pnls(trades: list[dict[str, Any]]) -> list[float]:
    pnls: list[float] = []
    for t in trades or []:
        if isinstance(t, dict):
            pnls.append(float(t.get("pnl", 0.0) or 0.0))
    return pnls


def _run_monte_carlo(trades: list[dict[str, Any]], simulations: int = 200, seed: int | None = None) -> dict[str, Any]:
    pnls = _extract_trade_pnls(trades)
    if not pnls:
        return {"median_return_pct": -100.0, "worst_drawdown_pct": 100.0}

    rng = random.Random(seed)
    returns: list[float] = []
    drawdowns: list[float] = []

    for _ in range(simulations):
        equity = 10_000.0
        peak = equity
        worst_dd = 0.0
        for _ in pnls:
            equity += rng.choice(pnls)
            peak = max(peak, equity)
            dd = (equity - peak) / max(peak, 1e-9) * 100.0
            worst_dd = min(worst_dd, dd)
        returns.append((equity / 10_000.0 - 1.0) * 100.0)
        drawdowns.append(abs(worst_dd))

    returns.sort()
    drawdowns.sort()
    median_return = returns[len(returns) // 2]
    worst_drawdown = max(drawdowns) if drawdowns else 100.0
    return {
        "median_return_pct": float(median_return),
        "worst_drawdown_pct": float(worst_drawdown),
    }


def _quick_proxy_score(backtest: dict[str, Any]) -> float:
    ret = float(backtest.get("return_pct", 0.0) or 0.0)
    dd = abs(float(backtest.get("max_drawdown_pct", 0.0) or 0.0))
    pf = float(backtest.get("profit_factor", 0.0) or 0.0)
    wr = float(backtest.get("win_rate", 0.0) or 0.0)
    trades = int(backtest.get("trades", 0) or 0)

    return (
        0.35 * max(-1.0, min(1.0, ret / 2.0))
        + 0.25 * max(0.0, min(1.0, pf / 2.0))
        + 0.15 * max(0.0, min(1.0, wr))
        + 0.15 * max(0.0, min(1.0, 1.0 - (dd / 20.0)))
        + 0.10 * max(0.0, min(1.0, trades / 20.0))
    )


def _quick_evaluate_candidate(candidate: StrategyCandidate, cfg: AgentConfig, iteration: int, parent_id: str) -> CandidateResult:
    params = dict(candidate.parameters or {})
    bt = run_backtest(
        cfg.symbol,
        cfg.timeframe,
        cfg.start,
        cfg.end,
        strategy_override={"parameters": params},
    )
    quick_score = _quick_proxy_score(bt)
    score = AgentScore(score=round(float(quick_score), 6), passed=False, reasons=tuple())
    return CandidateResult(candidate=candidate, backtest=bt, walk_forward={}, monte_carlo={}, score=score, parent_id=parent_id, iteration=iteration)


def _full_evaluate_candidate(result: CandidateResult, cfg: AgentConfig) -> CandidateResult:
    params = dict(result.candidate.parameters or {})
    bt = result.backtest

    wf_folds = build_walk_forward_folds(cfg.start, cfg.end, folds=max(1, int(cfg.folds)))
    fold_reports: list[dict[str, Any]] = []
    for fold in wf_folds:
        (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = _split_window(fold.start, fold.end)
        fold_reports.append(
            {
                "label": fold.label,
                "train": run_backtest(cfg.symbol, cfg.timeframe, tr_s, tr_e, strategy_override={"parameters": params}),
                "val": run_backtest(cfg.symbol, cfg.timeframe, va_s, va_e, strategy_override={"parameters": params}),
                "test": run_backtest(cfg.symbol, cfg.timeframe, te_s, te_e, strategy_override={"parameters": params}),
            }
        )

    wf = summarize_walk_forward_reports(fold_reports, timeframe=cfg.timeframe)
    mc = {"median_return_pct": -100.0, "worst_drawdown_pct": 100.0}
    if wf.get("passed") or float(wf.get("score", 0.0) or 0.0) >= 0.4:
        mc = _run_monte_carlo(bt.get("trades_detail", []), seed=result.iteration)

    score = score_candidate(bt, wf, mc, goal_return_pct=cfg.goal_return, max_drawdown_pct=cfg.max_dd)
    return CandidateResult(candidate=result.candidate, backtest=bt, walk_forward=wf, monte_carlo=mc, score=score, parent_id=result.parent_id, iteration=result.iteration)


def _persist_candidate(result: CandidateResult, cfg: AgentConfig) -> dict[str, Any]:
    candidate = result.candidate
    bt = result.backtest
    wf = result.walk_forward
    mc = result.monte_carlo
    params = dict(candidate.parameters or {})

    status_info = classify_strategy_status(
        agent_score=result.score.as_dict(),
        backtest=bt,
        walk_forward=wf,
        timeframe=cfg.timeframe,
    )

    record_evolution_run(
        cycle_id=f"iter_{result.iteration}",
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        parent_strategy_id=result.parent_id,
        child_strategy_id=candidate.strategy_id,
        status=status_info["status"],
        score=float(result.score.score),
        passed=bool(result.score.passed),
        parameters=params,
        metrics={"backtest": bt, "walk_forward": wf, "monte_carlo": mc, "agent_score": result.score.as_dict()},
        notes=", ".join(status_info.get("reasons") or []),
    )

    upsert_strategy(
        candidate.strategy_id,
        base_strategy=candidate.base_strategy,
        version=int(candidate.version or 1),
        status=status_info["status"],
        parameters=params,
        metrics={"backtest": bt, "walk_forward": wf, "monte_carlo": mc, "agent_score": result.score.as_dict()},
        tags=list(candidate.tags or []),
        source=candidate.source,
        notes=", ".join(status_info.get("reasons") or []),
        active=bool(status_info.get("active")),
        robustness_score=float(wf.get("score", 0.0) or 0.0),
        parent_strategy_id=result.parent_id,
    )
    return status_info


def run_agent(cfg: AgentConfig) -> dict[str, Any]:
    parent = _choose_parent(cfg)
    best_overall: CandidateResult | None = None
    elite_archive: list[dict[str, Any]] = []

    llm_client = get_default_llm_client()
    prev_parent_id: str | None = None

    iteration = 0
    while True:
        iteration += 1
        if not cfg.continuous and iteration > cfg.iterations:
            break

        feedback = build_feedback_summary(symbol=cfg.symbol, timeframe=cfg.timeframe)
        diversity_pool = _elite_ranked(cfg, limit=10)
        if elite_archive:
            # keep the archive separate from the live parent but available for diversity pressure
            diversity_pool = (elite_archive + diversity_pool)[:15]

        children = mutate_parent(
            parent,
            cfg.symbol,
            cfg.timeframe,
            n_children=max(1, int(cfg.candidates)),
            feedback=feedback,
            diversity_pool=diversity_pool,
            llm_client=llm_client,
        )

        if not children:
            raise RuntimeError("agent produced no candidates")

        quick_results: list[CandidateResult] = []
        parent_id = str(parent.get("strategy_id") if isinstance(parent, dict) else getattr(parent, "strategy_id", "seed"))

        if cfg.workers > 1 and len(children) > 1:
            with ThreadPoolExecutor(max_workers=int(cfg.workers)) as pool:
                futures = [pool.submit(_quick_evaluate_candidate, c, cfg, iteration, parent_id) for c in children]
                for fut in as_completed(futures):
                    quick_results.append(fut.result())
        else:
            for c in children:
                quick_results.append(_quick_evaluate_candidate(c, cfg, iteration, parent_id))

        quick_results.sort(key=lambda r: r.score.score, reverse=True)
        full_eval_count = min(len(quick_results), max(2, min(3, len(quick_results))))
        selected = quick_results[:full_eval_count]

        results: list[CandidateResult] = []
        if cfg.workers > 1 and len(selected) > 1:
            with ThreadPoolExecutor(max_workers=int(cfg.workers)) as pool:
                futures = [pool.submit(_full_evaluate_candidate, r, cfg) for r in selected]
                for fut in as_completed(futures):
                    results.append(fut.result())
        else:
            for r in selected:
                results.append(_full_evaluate_candidate(r, cfg))

        if not results:
            raise RuntimeError("agent produced no evaluable candidates")

        best = max(results, key=lambda r: r.score.score)
        status_info = _persist_candidate(best, cfg)

        if best_overall is None or best.score.score > best_overall.score.score:
            best_overall = best

        print(
            {
                "iteration": iteration,
                "best_strategy": best.candidate.strategy_id,
                "score": best.score.score,
                "passed": best.score.passed,
                "status": status_info.get("status"),
                "reasons": tuple(status_info.get("reasons") or []),
                "return_pct": best.backtest.get("return_pct"),
                "max_dd": best.backtest.get("max_drawdown_pct"),
                "pf": best.backtest.get("profit_factor"),
                "wr": best.backtest.get("win_rate"),
                "wf_passed": best.walk_forward.get("passed"),
            }
        )

        if status_info.get("status") in ELITE_STATUSES:
            elite_record = {
                "strategy_id": best.candidate.strategy_id,
                "base_strategy": best.candidate.base_strategy,
                "version": best.candidate.version,
                "parameters": best.candidate.parameters,
                "symbol": cfg.symbol,
                "timeframe": cfg.timeframe,
                "tags": best.candidate.tags,
                "source": best.candidate.source,
                "metrics": {"backtest": best.backtest, "walk_forward": best.walk_forward, "monte_carlo": best.monte_carlo, "agent_score": best.score.as_dict()},
                "robustness_score": float(best.walk_forward.get("score", 0.0) or 0.0),
                "status": status_info.get("status"),
                "active": bool(status_info.get("active")),
            }
            elite_archive = [e for e in elite_archive if e.get("strategy_id") != elite_record["strategy_id"]]
            elite_archive.append(elite_record)
            elite_archive.sort(key=lambda r: (_composite_parent_score(r), float(r.get("robustness_score", 0.0) or 0.0)), reverse=True)
            elite_archive = elite_archive[:10]

        # Only advance the live parent when the candidate itself is validated or better.
        if status_info.get("status") in ELITE_STATUSES:
            parent = _normalize_parent(
                {
                    "strategy_id": best.candidate.strategy_id,
                    "base_strategy": best.candidate.base_strategy,
                    "version": best.candidate.version,
                    "parameters": best.candidate.parameters,
                    "symbol": cfg.symbol,
                    "timeframe": cfg.timeframe,
                    "tags": best.candidate.tags,
                    "source": best.candidate.source,
                },
                cfg.symbol,
                cfg.timeframe,
            )
            prev_parent_id = parent_id
        else:
            # keep the current live parent unchanged
            pass

        if best.score.passed:
            return {
                "status": "target_achieved",
                "best_strategy": best.candidate.strategy_id,
                "score": best.score.as_dict(),
                "backtest": best.backtest,
                "walk_forward": best.walk_forward,
                "monte_carlo": best.monte_carlo,
            }

        if cfg.continuous:
            time.sleep(max(0.0, float(cfg.sleep_seconds or 0.0)))

    if best_overall is None:
        raise RuntimeError("agent did not evaluate any candidates")

    return {
        "status": "stopped",
        "best_strategy": best_overall.candidate.strategy_id,
        "score": best_overall.score.as_dict(),
        "backtest": best_overall.backtest,
        "walk_forward": best_overall.walk_forward,
        "monte_carlo": best_overall.monte_carlo,
    }
