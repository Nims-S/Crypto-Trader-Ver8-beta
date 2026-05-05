from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrategyState:
    trades_this_week: int = 0
    allow_shorts: bool = False
    min_adx: float = 16.0
    min_atr_rank: float = 0.15
    min_bb_rank: float = 0.15
    rsi_long: float = 53.0
    rsi_short: float = 47.0


@dataclass
class Signal:
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    symbol: str
    strategy: str
    regime: str
    confidence: float = 0.5
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    secondary_take_profit_pct: float = 0.0
    tp3_pct: float = 0.0
    tp3_close_fraction: float = 0.0
    trail_pct: float = 0.0
    trail_atr_mult: float = 0.0
    trail_ema20: bool = False
    tp1_close_fraction: float = 0.5
    tp2_close_fraction: float = 0.5
    be_trigger_rr: float = 0.0
    max_bars_override: int = 0
    cooldown_bars: int = 0
    size_multiplier: float = 1.0
