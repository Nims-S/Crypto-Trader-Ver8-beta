from __future__ import annotations

import os


PAPER_TRADING = os.getenv("PAPER_TRADING", "1").strip().lower() in {"1", "true", "yes", "on"}
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
API_PASSWORD = os.getenv("API_PASSWORD", "")
USE_SANDBOX = os.getenv("USE_SANDBOX", "0").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_LIVE_INTERVAL_SECONDS = int(os.getenv("LIVE_INTERVAL_SECONDS", "60"))
DEFAULT_TOTAL_CAPITAL = float(os.getenv("TOTAL_CAPITAL", "1000"))
MAX_LIVE_POSITIONS = int(os.getenv("MAX_LIVE_POSITIONS", "6"))
LIVE_LOOKBACK_BARS = int(os.getenv("LIVE_LOOKBACK_BARS", "300"))
LIVE_STATE_FILE = os.getenv("LIVE_STATE_FILE", ".live_state.json")
USE_ASYNC_MARKET_FETCH = os.getenv("USE_ASYNC_MARKET_FETCH", "1").strip().lower() in {"1", "true", "yes", "on"}
BASE_SLIPPAGE_BPS = float(os.getenv("BASE_SLIPPAGE_BPS", "4.0"))
ATR_SLIPPAGE_MULT = float(os.getenv("ATR_SLIPPAGE_MULT", "18.0"))
IMPACT_SLIPPAGE_MULT = float(os.getenv("IMPACT_SLIPPAGE_MULT", "28.0"))
MIN_FILL_PROBABILITY = float(os.getenv("MIN_FILL_PROBABILITY", "0.35"))
LATENCY_MS_BASE = int(os.getenv("LATENCY_MS_BASE", "45"))
LATENCY_ATR_MULT = float(os.getenv("LATENCY_ATR_MULT", "3500.0"))
LATENCY_IMPACT_MULT = float(os.getenv("LATENCY_IMPACT_MULT", "140.0"))
