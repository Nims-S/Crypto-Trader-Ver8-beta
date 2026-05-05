from __future__ import annotations

import asyncio
from typing import Any, Dict, Tuple

from execution.market_data import load_market_bundle


async def load_market_bundle_async(symbol: str, timeframe: str) -> Tuple[Any, Any, str]:
    return await asyncio.to_thread(load_market_bundle, symbol, timeframe)


async def load_market_cache_async(symbols: list[str], timeframes: list[str]) -> Dict[Tuple[str, str], Tuple[Any, Any, str]]:
    tasks = []
    keys = []
    for symbol in symbols:
        for tf in timeframes:
            keys.append((symbol, tf))
            tasks.append(load_market_bundle_async(symbol, tf))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    cache: Dict[Tuple[str, str], Tuple[Any, Any, str]] = {}
    for key, res in zip(keys, results):
        if isinstance(res, Exception):
            continue
        cache[key] = res
    return cache
