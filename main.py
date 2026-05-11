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


def cmd_refresh_registry(args: argparse.Namespace) -> int:
    payload = refresh_registry_from_snapshot_file(args.snapshot_file)
    return _dump(payload, output_file=args.output_file)

# existing commands unchanged below
