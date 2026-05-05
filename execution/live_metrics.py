from __future__ import annotations

from typing import List, Dict, Any


def summarize_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "pnl": 0.0,
            "max_drawdown_pct": 0.0,
        }

    wins = 0
    losses = 0
    profit = 0.0
    loss = 0.0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for t in trades:
        pnl = float(t.get("pnl") or 0.0)
        equity += pnl
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

        if pnl >= 0:
            wins += 1
            profit += pnl
        else:
            losses += 1
            loss += abs(pnl)

    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0
    pf = profit / loss if loss > 0 else (profit if profit > 0 else 0.0)
    dd_base = max(abs(peak), 1.0)
    dd_pct = (max_dd / dd_base) * 100.0

    return {
        "trades": total,
        "win_rate": win_rate,
        "profit_factor": pf,
        "pnl": equity,
        "max_drawdown_pct": dd_pct,
    }
