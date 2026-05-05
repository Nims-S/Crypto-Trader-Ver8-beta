from __future__ import annotations

import json
import os
from typing import Any

from execution.portfolio_state import PortfolioState


def load_portfolio_state(path: str) -> PortfolioState | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PortfolioState.from_dict(data)
    except Exception:
        return None


def save_portfolio_state(path: str, portfolio: PortfolioState) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(portfolio.to_dict(), f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def ensure_parent_dir(path: str) -> None:
    if not path:
        return
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
