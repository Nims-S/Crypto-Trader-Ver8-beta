from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from execution.backtest.core import run_backtest
from registry.store import get_strategy, list_evolution_runs, list_experiments, list_strategies, rank_strategies
from research.candidate_generator import seed_strategy
from research.feedback import build_feedback_summary


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


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


def cmd_show(args: argparse.Namespace) -> int:
    strategy = get_strategy(args.strategy_id)
    experiments = list_experiments(strategy_id=args.strategy_id, limit=args.limit)
    evolutions = list_evolution_runs(strategy_id=args.strategy_id, limit=args.limit)
    _print_json({"strategy": strategy, "experiments": experiments, "evolutions": evolutions})
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    result = run_backtest(
        args.symbol,
        args.timeframe,
        start=args.start,
        end=args.end,
        allow_shorts=args.allow_shorts,
        max_bars=args.max_bars,
        use_cache=not args.no_cache,
        strategy_override={"parameters": {"entry_mode": args.entry_mode}} if args.entry_mode else None,
    )
    _print_json(result)
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    feedback = build_feedback_summary(strategy_id=args.strategy_id, symbol=args.symbol, timeframe=args.timeframe)
    _print_json(feedback)
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    candidate = seed_strategy(args.symbol, args.timeframe, family=args.family)
    _print_json(candidate.__dict__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crypto-Trader-Ver8-beta launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="Show registry status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("rank", help="Rank strategies")
    p.add_argument("--symbol")
    p.add_argument("--timeframe")
    p.add_argument("--regime")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("show", help="Show one strategy and its lineage")
    p.add_argument("strategy_id")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("backtest", help="Run a backtest")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--max-bars", type=int, default=0)
    p.add_argument("--allow-shorts", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--entry-mode", choices=["trend_pullback", "breakout", "mean_reversion"], default=None)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("feedback", help="Summarize strategy-store feedback")
    p.add_argument("--strategy-id")
    p.add_argument("--symbol")
    p.add_argument("--timeframe")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("seed", help="Create a seed candidate")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--family", default="evo")
    p.set_defaults(func=cmd_seed)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
