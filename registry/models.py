from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class StrategyRecord:
    strategy_id: str
    base_strategy: str = "unknown"
    version: int = 1
    status: str = "candidate"
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source: str = "manual"
    notes: str = ""
    active: bool = False
    logic_hash: str | None = None
    regime_profile: str | None = None
    robustness_score: float = 0.0
    parent_strategy_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    validated_at: str | None = None
