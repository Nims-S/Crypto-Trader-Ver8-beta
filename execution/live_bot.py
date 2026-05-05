from __future__ import annotations

import time
import asyncio
from typing import Any, Dict

from config.defaults import DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES
from config.execution import DEFAULT_TOTAL_CAPITAL, DEFAULT_LIVE_INTERVAL_SECONDS, PAPER_TRADING, LIVE_STATE_FILE
from execution.router import route_strategies
from execution.allocator import allocate_capital
from execution.drift_monitor import compare_performance
from execution.portfolio_state import PortfolioState
from execution.executor import TradeExecutor
from execution.market_data import load_market_bundle
from execution.async_market_data import load_market_cache_async
from execution.portfolio_intelligence import build_portfolio_intelligence, portfolio_snapshot
from execution.live_metrics import summarize_trades
from execution.state_store import load_portfolio_state, save_portfolio_state, ensure_parent_dir
from execution.lifecycle import update_runtime
from execution.realism import estimate_execution
from registry.store import record_experiment, upsert_strategy
from strategy import StrategyState, generate_signal


def _load_market_cache(symbols, timeframes):
    try:
        return asyncio.run(load_market_cache_async(symbols, timeframes))
    except Exception:
        cache: Dict[tuple[str, str], tuple[Any, Any, str]] = {}
        for symbol in symbols:
            for tf in timeframes:
                try:
                    cache[(symbol, tf)] = load_market_bundle(symbol, tf)
                except Exception:
                    pass
        return cache


def run_live_cycle(
    portfolio: PortfolioState | None = None,
    total_capital: float | None = None,
    state_file: str | None = None,
) -> Dict[str, Any]:
    state_file = state_file or LIVE_STATE_FILE

    if portfolio is None:
        loaded = load_portfolio_state(state_file)
        if loaded:
            portfolio = loaded
        else:
            capital = float(total_capital or DEFAULT_TOTAL_CAPITAL)
            portfolio = PortfolioState(total_capital=capital, cash=capital)

    portfolio.cycle += 1

    symbols = list(DEFAULT_SYMBOLS)
    timeframes = list(DEFAULT_TIMEFRAMES)

    market_cache = _load_market_cache(symbols, timeframes)

    regimes: Dict[tuple[str, str], str | None] = {}
    for key, bundle in market_cache.items():
        regimes[key] = bundle[2] if bundle else None

    routed = route_strategies(symbols, timeframes, regimes=regimes)
    strategy_rows = [r["strategy"] for r in routed]

    intelligence = build_portfolio_intelligence(routed, portfolio, market_cache)

    free_cash = max(float(portfolio.cash), 0.0)
    allocations = allocate_capital(strategy_rows, free_cash, context=intelligence)
    portfolio.apply_allocations(allocations)

    executor = TradeExecutor(paper_trading=PAPER_TRADING)

    reports = []
    closed_this_cycle = set()

    for route in routed:
        sid = route.get("strategy_id")
        symbol = route.get("symbol")
        tf = route.get("timeframe")
        row = route.get("strategy") or {}
        alloc = next((a for a in allocations if a.get("strategy_id") == sid), None)
        capital = float((alloc or {}).get("capital", 0.0))

        bundle = market_cache.get((symbol, tf))
        if not bundle:
            continue
        ltf, htf, regime = bundle

        current_price = float(ltf.iloc[-1]["close"])

        pos = portfolio.get_position(sid)

        if pos:
            stop = float(pos.get("stop_loss") or 0.0)
            tp = float(pos.get("take_profit") or 0.0)
            if stop > 0 and current_price <= stop:
                exec_ctx = estimate_execution(ltf, price=current_price, side=pos.get("side", "LONG"), notional=float(pos.get("capital", 0.0)), symbol=symbol, timeframe=tf, cycle=portfolio.cycle, action="close")
                if exec_ctx.get("filled"):
                    close_result = executor.close_position(pos, exit_price=exec_ctx.get("fill_price", current_price), reason="stop_loss")
                    trade = portfolio.close_position(sid, float(close_result.get("exit_price", current_price)), "stop_loss")
                    if trade:
                        trade.update(exec_ctx)
                        closed_this_cycle.add(sid)
                        record_experiment(sid, symbol=symbol, timeframe=tf, run_type="live_cycle", metrics={"trade": trade, "execution": exec_ctx}, passed=False)
                else:
                    record_experiment(sid, symbol=symbol, timeframe=tf, run_type="live_cycle", metrics={"skip": "close_missed_fill", "execution": exec_ctx}, passed=False)
            elif tp > 0 and current_price >= tp:
                exec_ctx = estimate_execution(ltf, price=current_price, side=pos.get("side", "LONG"), notional=float(pos.get("capital", 0.0)), symbol=symbol, timeframe=tf, cycle=portfolio.cycle, action="close")
                if exec_ctx.get("filled"):
                    close_result = executor.close_position(pos, exit_price=exec_ctx.get("fill_price", current_price), reason="take_profit")
                    trade = portfolio.close_position(sid, float(close_result.get("exit_price", current_price)), "take_profit")
                    if trade:
                        trade.update(exec_ctx)
                        closed_this_cycle.add(sid)
                        record_experiment(sid, symbol=symbol, timeframe=tf, run_type="live_cycle", metrics={"trade": trade, "execution": exec_ctx}, passed=True)
                else:
                    record_experiment(sid, symbol=symbol, timeframe=tf, run_type="live_cycle", metrics={"skip": "close_missed_fill", "execution": exec_ctx}, passed=False)

        params = row.get("parameters") or {}
        state = StrategyState(allow_shorts=bool(params.get("allow_shorts", False)))
        signal = None
        try:
            signal = generate_signal(ltf, state=state, symbol=symbol, df_htf=htf, strategy_override=params)
        except Exception as e:
            signal = None
            record_experiment(
                sid,
                symbol=symbol,
                timeframe=tf,
                run_type="live_cycle",
                parameters=row.get("parameters"),
                metrics={"error": str(e)},
                passed=False,
                notes="signal generation failed",
            )

        if sid not in closed_this_cycle and not portfolio.get_position(sid) and signal:
            exec_ctx = estimate_execution(ltf, price=current_price, side=getattr(signal, "side", "LONG"), notional=capital, symbol=symbol, timeframe=tf, cycle=portfolio.cycle, action="open")
            if exec_ctx.get("filled"):
                # override entry price with realistic fill
                try:
                    setattr(signal, "entry_price", exec_ctx.get("fill_price", current_price))
                except Exception:
                    pass
                result = executor.open_position(
                    strategy_id=sid,
                    symbol=symbol,
                    timeframe=tf,
                    signal=signal,
                    capital=capital,
                    current_price=exec_ctx.get("fill_price", current_price),
                )
                if result.get("status") == "opened":
                    pos_dict = result.get("position") or {}
                    pos_dict.update(exec_ctx)
                    portfolio.open_position(pos_dict)
            else:
                record_experiment(sid, symbol=symbol, timeframe=tf, run_type="live_cycle", metrics={"skip": "open_missed_fill", "execution": exec_ctx}, passed=False)

        trades = [t for t in portfolio.trade_history if t.get("strategy_id") == sid]
        live_stats = summarize_trades(trades)
        portfolio.update_live_metrics(sid, live_stats)

        expected = (row.get("metrics") or {}).get("walk_forward") or {}
        drift = compare_performance(expected, live_stats)

        runtime = portfolio.strategy_runtime.get(sid, {})
        portfolio.strategy_runtime[sid] = update_runtime(runtime, live=live_stats, drift=drift, cycle=portfolio.cycle)

        upsert_strategy(
            sid,
            base_strategy=row.get("base_strategy"),
            version=row.get("version"),
            status=("disabled" if drift.get("status") == "disable" else row.get("status")),
            parameters=row.get("parameters"),
            metrics={**(row.get("metrics") or {}), "live": live_stats},
            tags=row.get("tags"),
            source="live_execution",
            notes="live update",
            active=(drift.get("status") != "disable"),
        )

        record_experiment(
            sid,
            symbol=symbol,
            timeframe=tf,
            run_type="live_cycle",
            parameters=row.get("parameters"),
            metrics={"live": live_stats, "drift": drift, "intelligence": intelligence.get(sid)},
            passed=(drift.get("status") != "disable"),
        )

        reports.append({
            "strategy_id": sid,
            "allocation": alloc,
            "live": live_stats,
            "drift": drift,
            "intelligence": intelligence.get(sid),
        })

    ensure_parent_dir(state_file)
    save_portfolio_state(state_file, portfolio)

    return {
        "allocations": allocations,
        "reports": reports,
        "cash": portfolio.cash,
        "open_positions": list(portfolio.positions.values()),
        "cycle": portfolio.cycle,
        "snapshot": portfolio_snapshot(portfolio),
    }


def run_loop(interval_seconds: int | None = None, total_capital: float | None = None, state_file: str | None = None):
    interval = int(interval_seconds or DEFAULT_LIVE_INTERVAL_SECONDS)
    state_file = state_file or LIVE_STATE_FILE

    portfolio = load_portfolio_state(state_file)
    if not portfolio:
        capital = float(total_capital or DEFAULT_TOTAL_CAPITAL)
        portfolio = PortfolioState(total_capital=capital, cash=capital)

    while True:
        result = run_live_cycle(portfolio=portfolio, total_capital=total_capital, state_file=state_file)
        print(result)
        time.sleep(max(1, interval))
