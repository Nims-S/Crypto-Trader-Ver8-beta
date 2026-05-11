from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from execution.backtest.core import run_backtest
from execution.deployment import (
    build_deployment_plan,
    commit_deployment_plan,
    evaluate_drift,
    summarize_deployment_state,
)
from execution.live_regime import (
    detect_live_regime,
    load_snapshot_file,
    route_live_strategy,
    route_live_strategies_from_snapshot,
)
from registry.store import get_strategy, list_experiments, list_strategies, rank_strategies, upsert_strategy
from research.loop import EvolutionConfig, run_continuous_loop, run_evolution_cycle
from research.portfolio import build_portfolio_summary
from research.registry_refresh import refresh_registry_from_snapshot_file

DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
DEFAULT_TIMEFRAMES = ("1d",)
DEFAULT_VALIDATION_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _dump(payload: Any, *, output_file: str | None = None, indent: int = 2) -> int:
    text = json.dumps(_jsonable(payload), indent=indent, sort_keys=True, default=str)
    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _split_csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    items = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(items) if items else default


def _parse_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON argument must be an object")
    return parsed


def _build_evolution_config(args: argparse.Namespace) -> EvolutionConfig:
    symbols = _split_csv(args.symbols, DEFAULT_SYMBOLS)
    timeframes = _split_csv(args.timeframes, DEFAULT_TIMEFRAMES)
    validation_symbols = _split_csv(args.validation_symbols, DEFAULT_VALIDATION_SYMBOLS)
    return EvolutionConfig(
        symbols=symbols,
        timeframes=timeframes,
        validation_symbols=validation_symbols,
        start=args.start,
        end=args.end,
        folds=args.folds,
        parents_per_pair=args.parents_per_pair,
        children_per_parent=args.children_per_parent,
        use_cache=not args.no_cache,
        allow_shorts=args.allow_shorts,
        mc_iterations=args.mc_iterations,
    )


def _add_output_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-file", default=None, help="Optional JSON output file")


def cmd_status(args: argparse.Namespace) -> int:
    payload = {
        "strategy_count": len(list_strategies(active_only=False)),
        "active_strategy_count": len(list_strategies(active_only=True)),
        "ranked_top": rank_strategies(limit=5, active_only=False),
        "deployment": summarize_deployment_state(),
    }
    return _dump(payload, output_file=args.output_file)


def cmd_rank(args: argparse.Namespace) -> int:
    payload = rank_strategies(
        symbol=args.symbol,
        timeframe=args.timeframe,
        regime=args.regime,
        active_only=not args.include_inactive,
        limit=args.limit,
    )
    return _dump(payload, output_file=args.output_file)


def cmd_show(args: argparse.Namespace) -> int:
    payload = {
        "strategy": get_strategy(args.strategy_id),
        "experiments": list_experiments(strategy_id=args.strategy_id, limit=args.limit),
    }
    return _dump(payload, output_file=args.output_file)


def cmd_backtest(args: argparse.Namespace) -> int:
    payload = run_backtest(
        args.symbol,
        args.timeframe,
        start=args.start,
        end=args.end,
        allow_shorts=args.allow_shorts,
        use_cache=not args.no_cache,
    )
    return _dump(payload, output_file=args.output_file)


def cmd_feedback(args: argparse.Namespace) -> int:
    rows = list_experiments(strategy_id=args.strategy_id, limit=args.limit)
    payload = {
        "strategy_id": args.strategy_id,
        "experiment_count": len(rows),
        "latest_experiments": rows[: min(len(rows), args.limit)],
    }
    return _dump(payload, output_file=args.output_file)


def cmd_seed(args: argparse.Namespace) -> int:
    parameters = _parse_json_arg(args.parameters)
    payload = upsert_strategy(
        args.strategy_id,
        base_strategy=args.base_strategy,
        version=args.version,
        status=args.status,
        parameters=parameters,
        metrics={},
        tags=[args.symbol, args.timeframe, args.regime, "seed"],
        source="manual_seed",
        notes=args.notes,
        active=False,
        regime_profile=args.regime,
        robustness_score=0.0,
        parent_strategy_id=args.parent_strategy_id,
    )
    return _dump(payload, output_file=args.output_file)


def cmd_evolve(args: argparse.Namespace) -> int:
    config = _build_evolution_config(args)
    if args.cycle_id:
        payload = run_evolution_cycle(config, cycle_id=args.cycle_id)
    else:
        payload = run_evolution_cycle(config)
    return _dump(payload, output_file=args.output_file)


def cmd_loop(args: argparse.Namespace) -> int:
    config = _build_evolution_config(args)
    payload = run_continuous_loop(config, interval_seconds=args.interval_seconds, cycles=args.cycles)
    return _dump(payload, output_file=args.output_file)


def cmd_portfolio(args: argparse.Namespace) -> int:
    payload = build_portfolio_summary(
        list_strategies(active_only=False),
        regime=args.regime,
        limit=args.limit,
        total_capital=args.capital,
        unique_markets=not args.allow_same_market,
        soft_fill=args.soft_fill,
        probationary_capital_fraction=args.probationary_capital_fraction,
    )
    return _dump(payload, output_file=args.output_file)


def cmd_deploy(args: argparse.Namespace) -> int:
    plan = build_deployment_plan(
        symbols=_split_csv(args.symbols, DEFAULT_SYMBOLS),
        timeframes=_split_csv(args.timeframes, DEFAULT_TIMEFRAMES),
        total_capital=args.capital,
        regimes={},
        limit=args.limit,
        temperature=args.temperature,
        paper=not args.live,
    )
    payload = commit_deployment_plan(plan)
    return _dump(payload, output_file=args.output_file)


def cmd_deploy_status(args: argparse.Namespace) -> int:
    return _dump(summarize_deployment_state(), output_file=args.output_file)


def cmd_drift(args: argparse.Namespace) -> int:
    return _dump(evaluate_drift(args.strategy_id), output_file=args.output_file)


def cmd_live_regime(args: argparse.Namespace) -> int:
    snapshot = load_snapshot_file(args.snapshot_file) if args.snapshot_file else {}
    if snapshot:
        payload = detect_live_regime(snapshot)
    else:
        payload = {"error": "snapshot_file required"}
    return _dump(payload, output_file=args.output_file)


def cmd_live_route(args: argparse.Namespace) -> int:
    if args.snapshot_file:
        snapshot = load_snapshot_file(args.snapshot_file)
        payload = route_live_strategies_from_snapshot(snapshot)
    else:
        payload = route_live_strategy()
    return _dump(payload, output_file=args.output_file)


def cmd_refresh_registry(args: argparse.Namespace) -> int:
    payload = refresh_registry_from_snapshot_file(args.snapshot_file)
    return _dump(payload, output_file=args.output_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="Show registry and deployment status")
    _add_output_arg(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("rank", help="Rank strategies")
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--regime", default=None)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--include-inactive", action="store_true")
    _add_output_arg(p)
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("show", help="Show a strategy")
    p.add_argument("strategy_id")
    p.add_argument("--limit", type=int, default=25)
    _add_output_arg(p)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("backtest", help="Run a backtest")
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--allow-shorts", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    _add_output_arg(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("feedback", help="Show recent experiments")
    p.add_argument("--strategy-id", default=None)
    p.add_argument("--limit", type=int, default=25)
    _add_output_arg(p)
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("seed", help="Create or update a seed strategy")
    p.add_argument("strategy_id")
    p.add_argument("--base-strategy", default="seed")
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--status", default="candidate")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--regime", default="mean_reversion")
    p.add_argument("--parent-strategy-id", default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--parameters", default="{}")
    _add_output_arg(p)
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("evolve", help="Run one evolution cycle")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    p.add_argument("--validation-symbols", default=",".join(DEFAULT_VALIDATION_SYMBOLS))
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-04-30")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--parents-per-pair", type=int, default=3)
    p.add_argument("--children-per-parent", type=int, default=3)
    p.add_argument("--mc-iterations", type=int, default=1000)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--allow-shorts", action="store_true")
    p.add_argument("--cycle-id", default=None)
    _add_output_arg(p)
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("loop", help="Run multiple evolution cycles")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    p.add_argument("--validation-symbols", default=",".join(DEFAULT_VALIDATION_SYMBOLS))
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-04-30")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--parents-per-pair", type=int, default=3)
    p.add_argument("--children-per-parent", type=int, default=3)
    p.add_argument("--mc-iterations", type=int, default=1000)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--allow-shorts", action="store_true")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--interval-seconds", type=int, default=5)
    _add_output_arg(p)
    p.set_defaults(func=cmd_loop)

    p = sub.add_parser("portfolio", help="Build portfolio summary")
    p.add_argument("--regime", default="mean_reversion")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--allow-same-market", action="store_true")
    p.add_argument("--soft-fill", action="store_true")
    p.add_argument("--probationary-capital-fraction", type=float, default=0.35)
    _add_output_arg(p)
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("deploy", help="Build and commit a deployment plan")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--live", action="store_true")
    _add_output_arg(p)
    p.set_defaults(func=cmd_deploy)

    p = sub.add_parser("deploy-status", help="Summarize deployment state")
    _add_output_arg(p)
    p.set_defaults(func=cmd_deploy_status)

    p = sub.add_parser("drift", help="Evaluate live drift for a strategy")
    p.add_argument("strategy_id")
    _add_output_arg(p)
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("live-regime", help="Detect live regime")
    p.add_argument("--snapshot-file", default=None)
    _add_output_arg(p)
    p.set_defaults(func=cmd_live_regime)

    p = sub.add_parser("live-route", help="Route live strategies")
    p.add_argument("--snapshot-file", default=None)
    _add_output_arg(p)
    p.set_defaults(func=cmd_live_route)

    p = sub.add_parser("refresh-registry", help="Refresh registry from an evolve snapshot")
    p.add_argument("snapshot_file")
    _add_output_arg(p)
    p.set_defaults(func=cmd_refresh_registry)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        payload = {"error": str(exc), "command": getattr(args, "command", None)}
        print(json.dumps(payload, indent=2, sort_keys=True, default=str), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
