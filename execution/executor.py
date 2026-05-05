from __future__ import annotations

from typing import Any, Dict

from config.execution import PAPER_TRADING, LIVE_TRADING_ENABLED
from config.risk import RISK_PER_TRADE, MAX_NOTIONAL_FRAC


class TradeExecutor:
    def __init__(self, paper_trading: bool | None = None):
        self.paper = PAPER_TRADING if paper_trading is None else bool(paper_trading)
        self.live_enabled = bool(LIVE_TRADING_ENABLED) and not self.paper
        self.exchange = None
        if self.live_enabled:
            try:
                import ccxt  # type: ignore
                from config.execution import EXCHANGE_ID, API_KEY, API_SECRET, API_PASSWORD, USE_SANDBOX

                ex_cls = getattr(ccxt, EXCHANGE_ID)
                params = {"enableRateLimit": True}
                if API_KEY:
                    params["apiKey"] = API_KEY
                if API_SECRET:
                    params["secret"] = API_SECRET
                if API_PASSWORD:
                    params["password"] = API_PASSWORD
                self.exchange = ex_cls(params)
                if USE_SANDBOX and hasattr(self.exchange, "set_sandbox_mode"):
                    self.exchange.set_sandbox_mode(True)
            except Exception:
                self.exchange = None
                self.live_enabled = False

    def _position_size(self, entry: float, stop: float, capital: float) -> float:
        if entry <= 0 or stop <= 0:
            return 0.0
        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            return 0.0
        risk_amount = capital * RISK_PER_TRADE
        size_risk = risk_amount / stop_dist
        size_notional = (capital * MAX_NOTIONAL_FRAC) / entry
        return max(0.0, min(size_risk, size_notional))

    def open_position(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        signal: Any,
        capital: float,
        current_price: float,
    ) -> Dict[str, Any]:
        if signal is None:
            return {"status": "skipped", "reason": "no_signal"}

        side = getattr(signal, "side", "LONG")
        entry = float(getattr(signal, "entry_price", current_price) or current_price)
        stop = float(getattr(signal, "stop_loss", entry * 0.99) or entry * 0.99)
        tp = float(getattr(signal, "take_profit", entry * 1.02) or entry * 1.02)

        qty = self._position_size(entry, stop, capital)
        if qty <= 0:
            return {"status": "skipped", "reason": "size_zero"}

        if side == "SHORT":
            return {"status": "skipped", "reason": "short_not_supported_live"}

        if self.live_enabled and self.exchange is not None:
            try:
                order = self.exchange.create_market_buy_order(symbol, qty)
                fill_price = float(order.get("average") or entry)
                return {
                    "status": "opened",
                    "mode": "live",
                    "position": {
                        "strategy_id": strategy_id,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "side": side,
                        "qty": qty,
                        "entry_price": fill_price,
                        "stop_loss": stop,
                        "take_profit": tp,
                        "capital": capital,
                    },
                }
            except Exception as e:
                return {"status": "error", "reason": str(e)}

        return {
            "status": "opened",
            "mode": "paper",
            "position": {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "side": side,
                "qty": qty,
                "entry_price": entry,
                "stop_loss": stop,
                "take_profit": tp,
                "capital": capital,
            },
        }

    def close_position(
        self,
        position: Dict[str, Any],
        *,
        exit_price: float,
        reason: str,
    ) -> Dict[str, Any]:
        entry = float(position.get("entry_price") or 0.0)
        qty = float(position.get("qty") or 0.0)
        side = position.get("side", "LONG")

        pnl = 0.0
        if qty > 0 and entry > 0:
            if side == "LONG":
                pnl = (exit_price - entry) * qty
            else:
                pnl = (entry - exit_price) * qty

        return {
            "status": "closed",
            "strategy_id": position.get("strategy_id"),
            "symbol": position.get("symbol"),
            "timeframe": position.get("timeframe"),
            "entry_price": entry,
            "exit_price": exit_price,
            "qty": qty,
            "pnl": pnl,
            "reason": reason,
        }
