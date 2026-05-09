from __future__ import annotations

import argparse
import json
import sys
import time
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
from registry.store import get_strategy, list_experiments, list_strategies, rank_strategies
from research.loop import EvolutionConfig, run_continuous_loop, run_evolution_cycle
from research.portfolio import build_portfolio_summary

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
    except Exception as exc:  # pragma: no cover - CLI guard
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


def cmd_evolve(args: argparse.Namespace) -> int:
    config = _build_evolution_config(args)
    cycles = max(1, int(args.cycles))
    interval = max(0, int(args.interval_seconds))

    if cycles == 1:
        result = run_evolution_cycle(config)
    else:
        result = run_continuous_loop(config, interval_seconds=interval, cycles=cycles)

    return _dump(result, output_file=args.output_file)


def cmd_portfolio(args: argparse.Namespace) -> int:
    strategies = list_strategies(active_only=bool(args.active_only))
    summary = build_portfolio_summary(
        strategies,
        regime=args.regime,
        limit=args.limit,
        unique_markets=not args.allow_same_market,
        total_capital=args.capital,
        symbols=_split_csv(args.symbols, DEFAULT_SYMBOLS) if args.symbols else None,
        timeframes=_split_csv(args.timeframes, DEFAULT_TIMEFRAMES) if args.timeframes else None,
        soft_fill=bool(args.soft_fill),
        probationary_capital_fraction=args.probationary_capital_fraction,
    )
    return _dump(summary, output_file=args.output_file)


def cmd_deploy(args: argparse.Namespace) -> int:
    symbols = list(_split_csv(args.symbols, DEFAULT_SYMBOLS))
    timeframes = list(_split_csv(args.timeframes, DEFAULT_TIMEFRAMES))
    regimes: dict[tuple[str, str], str | None] = {}
    if args.regime:
        for symbol in symbols:
            for timeframe in timeframes:
                regimes[(symbol, timeframe)] = args.regime

    plan = build_deployment_plan(
        symbols=symbols,
        timeframes=timeframes,
        total_capital=args.capital,
        regimes=regimes or None,
        limit=args.limit,
        temperature=args.temperature,
        paper=not args.live,
    )
    committed = commit_deployment_plan(plan)
    return _dump({"plan": plan.__dict__, "committed": committed}, output_file=args.output_file)


def cmd_deploy_status(args: argparse.Namespace) -> int:
    return _dump(summarize_deployment_state(), output_file=args.output_file)


def cmd_live_regime(args: argparse.Namespace) -> int:
    features = _parse_json_arg(args.features)
    htf_features = _parse_json_arg(args.htf_features) if args.htf_features else None
    snapshot = detect_live_regime(features, htf_features=htf_features, symbol=args.symbol, timeframe=args.timeframe)
    route = route_live_strategy(args.symbol, args.timeframe, features, htf_features=htf_features, limit=args.limit)
    payload = {
        "snapshot": snapshot.__dict__,
        "route": route.__dict__,
    }
    return _dump(payload, output_file=args.output_file)


def cmd_live_route(args: argparse.Namespace) -> int:
    if args.snapshot_file:
        snapshot_map = load_snapshot_file(args.snapshot_file)
    else:
        snapshot_map = {
            f"{args.symbol}|{args.timeframe}": {
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "features": _parse_json_arg(args.features),
                "htf_features": _parse_json_arg(args.htf_features) if args.htf_features else None,
            }
        }
    routed = route_live_strategies_from_snapshot(snapshot_map, limit=args.limit)
    return _dump({"routed": routed}, output_file=args.output_file)


def cmd_rank(args: argparse.Namespace) -> int:
    ranked = rank_strategies(
        symbol=args.symbol,
        timeframe=args.timeframe,
        regime=args.regime,
        active_only=not args.include_inactive,
        limit=args.limit,
    )
    return _dump({"ranked": ranked}, output_file=args.output_file)


def cmd_status(args: argparse.Namespace) -> int:
    payload = {
        "registry": {
            "total": len(list_strategies(active_only=False)),
            "active": len(list_strategies(active_only=True)),
            "top": rank_strategies(limit=min(5, args.limit)),
        },
        "deployment": summarize_deployment_state(),
        "experiments": list_experiments(limit=min(10, args.limit)),
    }
    return _dump(payload, output_file=args.output_file)


def cmd_backtest(args: argparse.Namespace) -> int:
    strategy_override = None
    if args.strategy_id:
        row = get_strategy(args.strategy_id)
        if not row:
            return _dump({"error": f"strategy not found: {args.strategy_id}"}, output_file=args.output_file)
        strategy_override = {"parameters": row.get("parameters") or {}}
        if args.allow_shorts is False and row.get("parameters"):
            strategy_override["parameters"]["allow_shorts"] = bool(row.get("parameters", {}).get("allow_shorts", False))

    if args.strategy_parameters:
        strategy_override = strategy_override or {"parameters": {}}
        strategy_override["parameters"].update(_parse_json_arg(args.strategy_parameters))

    result = run_backtest(
        args.symbol,
        args.timeframe,
        start=args.start,
        end=args.end,
        allow_shorts=args.allow_shorts,
        max_bars=args.max_bars,
        use_cache=not args.no_cache,
        strategy_override=strategy_override,
    )
    return _dump(result, output_file=args.output_file)


def cmd_feedback(args: argparse.Namespace) -> int:
    from research.feedback import build_feedback_summary

    summary = build_feedback_summary(strategy_id=args.strategy_id, symbol=args.symbol, timeframe=args.timeframe)
    return _dump(summary, output_file=args.output_file)


def cmd_loop(args: argparse.Namespace) -> int:
    config = _build_evolution_config(args)
    total_cycles = max(1, int(args.cycles))
    interval = max(0, int(args.interval_seconds))
    results = []
    for idx in range(total_cycles):
        results.append(run_evolution_cycle(config))
        if idx + 1 < total_cycles and interval > 0:
            time.sleep(interval)
    return _dump({"results": results}, output_file=args.output_file)


def _add_common_market_args(parser: argparse.ArgumentParser, *, with_regime: bool = True) -> None:
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-01-01")
    if with_regime:
        parser.add_argument("--regime", default="mean_reversion")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("evolve", help="Run the research/evolution loop")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    p.add_argument("--validation-symbols", default=",".join(DEFAULT_VALIDATION_SYMBOLS))
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--parents-per-pair", type=int, default=3)
    p.add_argument("--children-per-parent", type=int, default=3)
    p.add_argument("--mc-iterations", type=int, default=300)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--interval-seconds", type=int, default=0)
    p.add_argument("--allow-shorts", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--output-file")
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("portfolio", help="Build a portfolio basket")
    p.add_argument("--regime", default="mean_reversion")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--symbols")
    p.add_argument("--timeframes")
    p.add_argument("--allow-same-market", action="store_true")
    p.add_argument("--soft-fill", action="store_true")
    p.add_argument("--probationary-capital-fraction", type=float, default=0.35)
    p.add_argument("--active-only", action="store_true")
    p.add_argument("--output-file")
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("deploy", help="Build and commit a deployment plan")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--regime", default=None)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--live", action="store_true")
    p.add_argument("--output-file")
    p.set_defaults(func=cmd_deploy)

    p = sub.add_parser("deploy-status", help="Show deployment state")
    p.add_argument("--output-file")
    p.set_defaults(func=cmd_deploy_status)

    p = sub.add_parser("live-regime", help="Detect live regime and route a strategy")
    p.add_argument("--features", required=True)
    p.add_argument("--htf-features")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--output-file")
    p.set_defaults(func=cmd_live_regime)

    p = sub.add_parser("live-route", help="Route from a snapshot file or feature snapshot")
    p.add_argument("--snapshot-file")
    p.add_argument("--features")
    p.add_argument("--htf-features")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--output-file")
    p.set_defaults(func=cmd_live_route)

    p = sub.add_parser("rank", help="Rank strategies")
    p.add_argument("--symbol")
    p.add_argument("--timeframe")
    p.add_argument("--regime")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--include-inactive", action="store_true")
    p.add_argument("--output-file")
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("status", help="Show repository, registry, and deployment status")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--output-file")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("backtest", help="Run a backtest")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--max-bars", type=int, default=0)
    p.add_argument("--allow-shorts", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--strategy-id")
    p.add_argument("--strategy-parameters")
    p.add_argument("--output-file")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("feedback", help="Summarize failure patterns and mutation directives")
    p.add_argument("--strategy-id")
    p.add_argument("--symbol")
    p.add_argument("--timeframe")
    p.add_argument("--output-file")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("loop", help="Run multiple evolution cycles")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    p.add_argument("--validation-symbols", default=",".join(DEFAULT_VALIDATION_SYMBOLS))
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--parents-per-pair", type=int, default=3)
    p.add_argument("--children-per-parent", type=int, default=3)
    p.add_argument("--mc-iterations", type=int, default=300)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--interval-seconds", type=int, default=5)
    p.add_argument("--allow-shorts", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--output-file")
    p.set_defaults(func=cmd_loop)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2, sort_keys=True, default=str))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
