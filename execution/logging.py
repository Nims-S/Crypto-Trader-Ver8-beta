from __future__ import annotations
from registry.store import record_experiment, upsert_strategy
from research.scoring import score_metrics

def log_backtest_result(strategy_id: str, symbol: str, timeframe: str, params: dict, result: dict, *, base_strategy: str | None = None, version: int = 1):
    decision = score_metrics(result)
    payload = {**result, "decision": decision.as_dict()}
    experiment = record_experiment(strategy_id, symbol=symbol, timeframe=timeframe, run_type="backtest", parameters=params, metrics=payload, passed=decision.passed, notes="logged from execution/logging.py")
    upsert_strategy(strategy_id, base_strategy=base_strategy or strategy_id, version=version, status="validated" if decision.passed else "rejected", parameters=params, metrics=payload, tags=[symbol, timeframe, "backtest"], source="backtest", notes=f"decision={'pass' if decision.passed else 'fail'}", active=decision.passed)
    return {"decision": decision.as_dict(), "experiment": experiment}
