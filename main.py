from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from execution.backtest.core import run_backtest
from execution.deployment import (
    build_deployment_plan,
    commit_deployment_plan,
    evaluate_drift,
    summarize_deployment_state,
    update_live_metric,
)
from execution.live_regime import (
    detect_live_regime,
    load_snapshot_file,
    route_live_strategies_from_snapshot,
)
from registry.store import get_strategy, list_evolution_runs, list_experiments, list_strategies, rank_strategies
from research.candidate_generator import seed_strategy
from research.feedback import build_feedback_summary
from research.loop import EvolutionConfig, run_continuous_loop, run_evolution_cycle
from research.portfolio import build_portfolio_summary


def _json_text(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=str)


def _print_json(data: Any) -> str:
    text = _json_text(data)
    print(text)
    return text


def _write_output_file(path: str | None, text: str) -> None:
    if not path:
        return
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def cmd_status(args: argparse.Namespace) -> int:
    active = list_strategies(active_only=True)
    all_strats = list_strategies(active_only=False)
    counts = Counter(str(row.get("status") or "candidate") for row in all_strats)
    summary = {
        "total_strategies": len(all_strats),
        "active_strategies": len(active),
        "status_counts": dict(sorted(counts.items())),
        "latest_strategies": all_strats[: min(5, len(all_strats))],
    }
    _print_json(summary)
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    ranked = rank_strategies(symbol=args.symbol, timeframe=args.timeframe, regime=args.regime, limit=args.limit)
    _print_json({"ranked": ranked})
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    strategies = list_strategies(active_only=False)
    summary = build_portfolio_summary(
        strategies,
        regime=args.regime,
        limit=args.limit,
        unique_markets=not args.allow_same_market,
        total_capital=args.capital,
    )
    _print_json(summary)
    return 0
