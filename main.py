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


def _build_evolution_config(args: argparse.Namespace) -> EvolutionConfig:
    symbols = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
    validation_symbols = tuple(s.strip() for s in args.validation_symbols.split(",") if s.strip())
    timeframes = tuple(t.strip() for t in args.timeframes.split(",") if t.strip())
    return EvolutionConfig(
        symbols=symbols or ("BTC/USDT",),
        validation_symbols=validation_symbols or ("ETH/USDT",),
        timeframes=timeframes or ("1d",),
        start=args.start,
        end=args.end,
        folds=max(1, int(args.folds)),
        parents_per_pair=max(1, int(args.parents_per_pair)),
        children_per_parent=max(1, int(args.children_per_parent)),
        use_cache=not args.no_cache,
        allow_shorts=args.allow_shorts,
        mc_iterations=max(10, int(args.mc_iterations)),
    )


def cmd_evolve(args: argparse.Namespace) -> int:
    config = _build_evolution_config(args)
    result = run_evolution_cycle(config)
    text = _print_json(result)
    _write_output_file(args.output_file, text)
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    config = _build_evolution_config(args)
    result = run_continuous_loop(config, interval_seconds=max(0, int(args.interval_seconds)), cycles=args.cycles)
    _print_json({"runs": result})
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    plan = build_deployment_plan(
        symbols=symbols,
        timeframes=timeframes,
        total_capital=args.capital,
        limit=args.limit,
        paper=not args.live,
    )
    committed = commit_deployment_plan(plan)
    _print_json({"plan": plan.__dict__, "committed": committed})
    return 0


def cmd_deploy_status(args: argparse.Namespace) -> int:
    summary = summarize_deployment_state()
    _print_json(summary)
    return 0


def cmd_update_live(args: argparse.Namespace) -> int:
    metrics = {
        "profit_factor": args.pf,
        "win_rate": args.wr,
        "max_drawdown_pct": args.dd,
    }
    updated = update_live_metric(args.strategy_id, metrics)
    _print_json(updated)
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    result = evaluate_drift(args.strategy_id)
    _print_json(result)
    return 0


def cmd_live_regime(args: argparse.Namespace) -> int:
    features = json.loads(args.features or "{}")
    htf = json.loads(args.htf_features or "{}") if args.htf_features else None
    snapshot = detect_live_regime(features, htf_features=htf, symbol=args.symbol, timeframe=args.timeframe)
    _print_json(snapshot.__dict__)
    return 0


def cmd_live_route(args: argparse.Namespace) -> int:
    snapshot_map = load_snapshot_file(args.snapshot_file)
    routed = route_live_strategies_from_snapshot(snapshot_map, limit=args.limit)
    _print_json({"routed": routed})
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

    p = sub.add_parser("evolve", help="Run one autonomous research cycle")
    p.add_argument("--symbols", default="BTC/USDT")
    p.add_argument("--validation-symbols", default="ETH/USDT,SOL/USDT")
    p.add_argument("--timeframes", default="1d")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--mc-iterations", type=int, default=300)
    p.add_argument("--parents-per-pair", type=int, default=3)
    p.add_argument("--children-per-parent", type=int, default=3)
    p.add_argument("--allow-shorts", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--output-file", default=None, help="Write evolve output JSON to a text file")
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("loop", help="Run repeated autonomous research cycles")
    p.add_argument("--symbols", default="BTC/USDT")
    p.add_argument("--validation-symbols", default="ETH/USDT,SOL/USDT")
    p.add_argument("--timeframes", default="1d")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--mc-iterations", type=int, default=300)
    p.add_argument("--parents-per-pair", type=int, default=3)
    p.add_argument("--children-per-parent", type=int, default=3)
    p.add_argument("--allow-shorts", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--interval-seconds", type=int, default=3600)
    p.add_argument("--cycles", type=int, default=1)
    p.set_defaults(func=cmd_loop)

    p = sub.add_parser("deploy", help="Build and commit a deployment plan")
    p.add_argument("--symbols", default="BTC/USDT")
    p.add_argument("--timeframes", default="1d")
    p.add_argument("--capital", type=float, default=1000)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--live", action="store_true")
    p.set_defaults(func=cmd_deploy)

    p = sub.add_parser("deploy-status", help="Show deployment state")
    p.set_defaults(func=cmd_deploy_status)

    p = sub.add_parser("update-live", help="Update live metrics for a strategy")
    p.add_argument("strategy_id")
    p.add_argument("--pf", type=float, default=0.0)
    p.add_argument("--wr", type=float, default=0.0)
    p.add_argument("--dd", type=float, default=0.0)
    p.set_defaults(func=cmd_update_live)

    p = sub.add_parser("drift", help="Evaluate drift for a strategy")
    p.add_argument("strategy_id")
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("live-regime", help="Detect live regime from feature snapshot")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--features", help="JSON string of features", default="{}")
    p.add_argument("--htf-features", help="JSON string of HTF features", default=None)
    p.set_defaults(func=cmd_live_regime)

    p = sub.add_parser("live-route", help="Route strategies from snapshot file")
    p.add_argument("--snapshot-file", required=True)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=cmd_live_route)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
