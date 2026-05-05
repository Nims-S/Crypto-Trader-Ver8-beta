from __future__ import annotations

import argparse
import importlib
import inspect
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = ROOT / ".strategy_store.json"

EVALUATOR_MODULE_CANDIDATES = (
    "scripts.backtest",
    "scripts.evaluator",
    "scripts.engine",
    "backtest",
    "evaluator",
    "engine.backtest",
    "strategy_backtester",
    "strategy_engine",
)
EVALUATOR_FUNCTION_CANDIDATES = (
    "run_backtest",
    "backtest_strategy",
    "evaluate_strategy",
    "evaluate_candidate",
    "score_strategy",
    "run_evaluation",
)


@dataclass
class GateConfig:
    min_profit_factor: float = 0.95
    min_win_rate: float = 0.45
    min_return_pct: float = 0.0
    max_drawdown_pct: float = 15.0
    min_trades: int = 20
    max_mc_drawdown_pct: float = 15.0


@dataclass
class CandidateResult:
    strategy_id: str
    parent_strategy_id: Optional[str]
    child_strategy_id: str
    status: str
    passed: bool
    score: float
    wf_passed: bool
    reasons: Tuple[str, ...]
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    created_at: str
    cycle_id: str
    symbol: str
    timeframe: str


class StrategyRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {"counters": {"evolution_id": 0, "experiment_id": 0}, "evolution_runs": [], "experiments": []}

    def next_evolution_id(self) -> int:
        counters = self.data.setdefault("counters", {})
        counters["evolution_id"] = int(counters.get("evolution_id", 0)) + 1
        return counters["evolution_id"]

    def append_run(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.data.setdefault("evolution_runs", []).append(payload)
            self._save_locked()

    def best_parent(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        runs = [
            r for r in self.data.get("evolution_runs", [])
            if r.get("symbol") == symbol and r.get("timeframe") == timeframe
        ]
        if not runs:
            return None
        runs.sort(
            key=lambda r: (
                float(r.get("score", 0.0)),
                float(r.get("metrics", {}).get("backtest", {}).get("profit_factor", 0.0)),
            ),
            reverse=True,
        )
        return runs[0]

    def _save_locked(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    git_dir = ROOT / ".git"
    if not git_dir.exists():
        return "unknown"

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except Exception:
        return default


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def infer_regime(parameters: Dict[str, Any]) -> str:
    mode = str(parameters.get("entry_mode", "mean_reversion")).lower()
    if "trend" in mode:
        return "trend"
    if "hybrid" in mode:
        return "hybrid"
    return "mean_reversion"


def regime_weights(regime: str) -> Dict[str, float]:
    if regime == "trend":
        return {"trend": 0.65, "hybrid": 0.25, "mean_reversion": 0.10}
    if regime == "hybrid":
        return {"trend": 0.35, "hybrid": 0.40, "mean_reversion": 0.25}
    return {"trend": 0.20, "hybrid": 0.25, "mean_reversion": 0.55}


def choose_mutation_family(parent: Dict[str, Any], iteration: int, rng: random.Random) -> str:
    params = parent.get("parameters", {}) if parent else {}
    regime = infer_regime(params)
    weights = regime_weights(regime)
    notes = str(parent.get("notes", "") or "")
    if "val_weak" in notes or "test_weak" in notes:
        weights["trend"] += 0.10
        weights["hybrid"] += 0.10
    if iteration > 5:
        weights["hybrid"] += 0.05
    choices = list(weights.keys())
    probs = [max(0.01, weights[c]) for c in choices]
    return rng.choices(choices, weights=probs, k=1)[0]


def mutate_parameters(base: Dict[str, Any], family: str, rng: random.Random) -> Dict[str, Any]:
    p = dict(base)

    def j(v: float, scale: float, lo: float, hi: float) -> float:
        return clamp(v + rng.gauss(0.0, scale), lo, hi)

    if family == "trend":
        p.update({
            "entry_mode": "trend",
            "use_trend_filter": True,
            "use_structure_filter": True,
            "use_htf_filter": True,
            "use_reclaim_filter": rng.random() < 0.35,
            "use_volume_filter": rng.random() < 0.50,
            "use_bb_filter": True,
            "min_adx": j(safe_float(p.get("min_adx", 12.0)), 3.0, 10.0, 35.0),
            "min_bb_rank": j(safe_float(p.get("min_bb_rank", 0.20)), 0.08, 0.05, 0.65),
            "min_atr_rank": j(safe_float(p.get("min_atr_rank", 0.18)), 0.07, 0.05, 0.60),
            "rsi_max": j(safe_float(p.get("rsi_max", 40.0)), 5.0, 25.0, 60.0),
            "stop_atr_mult": j(safe_float(p.get("stop_atr_mult", 2.8)), 0.5, 1.4, 5.0),
            "tp1_rr": j(safe_float(p.get("tp1_rr", 2.2)), 0.4, 1.2, 5.0),
        })
        p["max_bars_override"] = int(j(safe_float(p.get("max_bars_override", 24)), 4.0, 8.0, 80.0))
        p["cooldown_bars"] = int(j(safe_float(p.get("cooldown_bars", 10)), 3.0, 1.0, 60.0))
    elif family == "hybrid":
        p.update({
            "entry_mode": "hybrid",
            "use_trend_filter": rng.random() < 0.75,
            "use_structure_filter": rng.random() < 0.80,
            "use_htf_filter": rng.random() < 0.85,
            "use_reclaim_filter": rng.random() < 0.55,
            "use_volume_filter": rng.random() < 0.55,
            "use_bb_filter": True,
            "min_adx": j(safe_float(p.get("min_adx", 8.0)), 2.0, 5.0, 28.0),
            "min_bb_rank": j(safe_float(p.get("min_bb_rank", 0.12)), 0.05, 0.03, 0.55),
            "min_atr_rank": j(safe_float(p.get("min_atr_rank", 0.12)), 0.05, 0.03, 0.55),
            "rsi_max": j(safe_float(p.get("rsi_max", 34.0)), 4.0, 18.0, 55.0),
            "stop_atr_mult": j(safe_float(p.get("stop_atr_mult", 2.0)), 0.35, 1.1, 4.5),
            "tp1_rr": j(safe_float(p.get("tp1_rr", 2.0)), 0.3, 1.1, 4.5),
        })
        p["max_bars_override"] = int(j(safe_float(p.get("max_bars_override", 20)), 3.0, 6.0, 60.0))
        p["cooldown_bars"] = int(j(safe_float(p.get("cooldown_bars", 12)), 3.0, 1.0, 60.0))
    else:
        p.update({
            "entry_mode": "mean_reversion",
            "use_trend_filter": rng.random() < 0.20,
            "use_structure_filter": rng.random() < 0.35,
            "use_htf_filter": rng.random() < 0.30,
            "use_reclaim_filter": rng.random() < 0.70,
            "use_volume_filter": rng.random() < 0.45,
            "use_bb_filter": True,
            "min_adx": j(safe_float(p.get("min_adx", 5.0)), 1.5, 3.0, 18.0),
            "min_bb_rank": j(safe_float(p.get("min_bb_rank", 0.08)), 0.04, 0.02, 0.35),
            "min_atr_rank": j(safe_float(p.get("min_atr_rank", 0.08)), 0.04, 0.02, 0.35),
            "rsi_max": j(safe_float(p.get("rsi_max", 30.0)), 3.5, 15.0, 42.0),
            "stop_atr_mult": j(safe_float(p.get("stop_atr_mult", 1.6)), 0.20, 0.8, 3.0),
            "tp1_rr": j(safe_float(p.get("tp1_rr", 1.8)), 0.25, 1.0, 3.5),
        })
        p["max_bars_override"] = int(j(safe_float(p.get("max_bars_override", 18)), 2.0, 4.0, 40.0))
        p["cooldown_bars"] = int(j(safe_float(p.get("cooldown_bars", 16)), 3.0, 2.0, 80.0))

    p["confidence"] = clamp(j(safe_float(p.get("confidence", 0.6)), 0.10, 0.10, 0.95), 0.10, 0.95)
    p["size_multiplier"] = clamp(j(safe_float(p.get("size_multiplier", 0.85)), 0.08, 0.35, 1.35), 0.35, 1.35)
    p["tp1_close_fraction"] = clamp(j(safe_float(p.get("tp1_close_fraction", 0.2)), 0.08, 0.05, 0.70), 0.05, 0.70)
    p["tp2_close_fraction"] = clamp(j(safe_float(p.get("tp2_close_fraction", 0.3)), 0.08, 0.05, 0.85), 0.05, 0.85)
    p["mutation_family"] = family
    p["regime_profile"] = infer_regime(p)
    return p


def build_candidate_id(symbol: str, timeframe: str, evo_id: int, rng: random.Random) -> str:
    return f"evo_{symbol.lower().replace('/', '_')}_{timeframe}_{evo_id}_{rng.randint(10000, 999999)}"


def metric_score_component(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp((value - lo) / (hi - lo), 0.0, 1.0)


def compute_ranking_score(metrics: Dict[str, Any]) -> float:
    backtest = metrics.get("backtest", {}) or {}
    wf = metrics.get("walk_forward", {}) or {}
    mc = metrics.get("monte_carlo", {}) or {}
    pf = safe_float(backtest.get("profit_factor"), 0.0)
    wr = safe_float(backtest.get("win_rate"), 0.0)
    ret = safe_float(backtest.get("return_pct"), 0.0)
    dd = abs(safe_float(backtest.get("max_drawdown_pct"), 0.0))
    trades = safe_float(backtest.get("trades"), 0.0)
    wf_score = safe_float(wf.get("score"), safe_float(wf.get("composite"), 0.0))
    wf_spread = safe_float(wf.get("score_spread"), 0.0)
    density = safe_float(wf.get("density_mean"), 0.0)
    mc_dd = abs(safe_float(mc.get("worst_drawdown_pct"), 0.0))

    score = (
        0.22 * metric_score_component(pf, 0.85, 2.0)
        + 0.18 * metric_score_component(wr, 0.35, 0.70)
        + 0.14 * metric_score_component(ret, -25.0, 25.0)
        + 0.16 * (1.0 - metric_score_component(dd, 0.0, 30.0))
        + 0.16 * clamp(wf_score, 0.0, 1.0)
        + 0.08 * clamp(density, 0.0, 1.0)
        + 0.06 * metric_score_component(trades, 10.0, 80.0)
        - 0.04 * metric_score_component(wf_spread, 0.0, 0.40)
        - 0.06 * metric_score_component(mc_dd, 0.0, 20.0)
    )
    return round(clamp(score, 0.0, 1.0), 6)


def extract_split_stat(split: Dict[str, Any]) -> Dict[str, float]:
    return {
        "pf": safe_float(split.get("profit_factor"), 0.0),
        "wr": safe_float(split.get("win_rate"), 0.0),
        "trades": safe_float(split.get("trades"), 0.0),
    }


def evaluate_gates(metrics: Dict[str, Any], walk_forward: Dict[str, Any], cfg: GateConfig) -> Tuple[bool, Tuple[str, ...], Dict[str, bool]]:
    reasons: List[str] = []
    gate_state: Dict[str, bool] = {}

    backtest = metrics.get("backtest", {}) or {}
    monte_carlo = metrics.get("monte_carlo", {}) or {}
    pf = safe_float(backtest.get("profit_factor"), 0.0)
    wr = safe_float(backtest.get("win_rate"), 0.0)
    ret = safe_float(backtest.get("return_pct"), 0.0)
    dd = abs(safe_float(backtest.get("max_drawdown_pct"), 0.0))
    trades = int(safe_float(backtest.get("trades"), 0.0))
    mc_dd = abs(safe_float(monte_carlo.get("worst_drawdown_pct"), 0.0))

    gate_state["pf_gate"] = pf >= cfg.min_profit_factor
    gate_state["return_gate"] = ret >= cfg.min_return_pct
    gate_state["dd_gate"] = dd <= cfg.max_drawdown_pct
    gate_state["walk_forward_gate"] = bool(walk_forward.get("passed", False))
    gate_state["monte_carlo_gate"] = mc_dd <= cfg.max_mc_drawdown_pct
    gate_state["density_gate"] = trades >= cfg.min_trades

    if not gate_state["pf_gate"]:
        reasons.append(f"pf<{cfg.min_profit_factor}")
    if not gate_state["return_gate"]:
        reasons.append(f"return<{cfg.min_return_pct}")
    if not gate_state["dd_gate"]:
        reasons.append(f"dd>{cfg.max_drawdown_pct}")
    if not gate_state["walk_forward_gate"]:
        reasons.append("walk_forward_failed")
    if not gate_state["monte_carlo_gate"]:
        reasons.append(f"mc_dd>{cfg.max_mc_drawdown_pct}")
    if not gate_state["density_gate"]:
        reasons.append("density_gate")

    split_ok = True
    split_reasons: List[str] = []
    for split_name in ("train", "val", "test"):
        for split in (walk_forward.get("split_results", {}) or {}).get(split_name, []):
            stat = extract_split_stat(split)
            local_reasons: List[str] = []
            if stat["trades"] < cfg.min_trades:
                local_reasons.append("trades<20")
            if stat["pf"] < cfg.min_profit_factor:
                local_reasons.append(f"pf<{cfg.min_profit_factor}")
            if stat["wr"] < cfg.min_win_rate:
                local_reasons.append(f"wr<{cfg.min_win_rate}")
            if local_reasons:
                split_ok = False
                split_reasons.extend([f"{split_name}:{reason}" for reason in local_reasons])
    gate_state["split_gate"] = split_ok
    if not split_ok:
        reasons.extend(split_reasons)

    passed = all(gate_state.values())
    return passed, tuple(reasons), gate_state


def normalize_evaluation_output(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        try:
            return result.to_dict()  # type: ignore[no-any-return]
        except Exception:
            pass
    raise TypeError(f"Unsupported evaluator return type: {type(result)!r}")


def describe_evaluator(evaluator: Callable[..., Any]) -> str:
    module = getattr(evaluator, "__module__", "unknown")
    name = getattr(evaluator, "__qualname__", getattr(evaluator, "__name__", repr(evaluator)))
    return f"{module}.{name}"


def resolve_evaluator() -> Callable[..., Any]:
    for module_name in EVALUATOR_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for fn_name in EVALUATOR_FUNCTION_CANDIDATES:
            fn = getattr(module, fn_name, None)
            if callable(fn):
                return fn
    raise RuntimeError("Could not locate an evaluator/backtest function. Update EVALUATOR_MODULE_CANDIDATES / EVALUATOR_FUNCTION_CANDIDATES.")


def call_evaluator(
    evaluator: Callable[..., Any],
    *,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    sig = inspect.signature(evaluator)
    names = list(sig.parameters.keys())

    # Prefer the repo's real backtest contract: run_backtest(sym, tf, start, end, strategy_override=...)
    attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = [
        ((), {"sym": symbol, "tf": timeframe, "start": start, "end": end, "strategy_override": {"parameters": parameters}}),
        ((), {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end, "strategy_override": {"parameters": parameters}}),
        ((), {"sym": symbol, "tf": timeframe, "start": start, "end": end, "parameters": parameters}),
        ((), {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end, "parameters": parameters}),
        ((), {"sym": symbol, "tf": timeframe, "start": start, "end": end}),
        ((), {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end}),
        ((symbol, timeframe, start, end), {"strategy_override": {"parameters": parameters}}),
        ((symbol, timeframe, start, end), {"params": parameters}),
        ((symbol, timeframe, start, end), {}),
        ((symbol, timeframe), {}),
        ((symbol,), {}),
    ]

    for args, kwargs in attempts:
        try:
            if kwargs:
                filtered = {k: v for k, v in kwargs.items() if k in names or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())}
                result = evaluator(*args, **filtered)
            else:
                result = evaluator(*args)
            out = normalize_evaluation_output(result)
            if isinstance(out, dict) and out.get("error"):
                out.setdefault("_evaluator", describe_evaluator(evaluator))
            return out
        except TypeError:
            continue

    raise TypeError("Unable to call evaluator with supported signatures.")


def strategy_family_name(parameters: Dict[str, Any]) -> str:
    return infer_regime(parameters)


def ensure_store_schema(store: StrategyRegistry) -> None:
    store.data.setdefault("counters", {}).setdefault("evolution_id", 0)
    store.data.setdefault("counters", {}).setdefault("experiment_id", 0)
    store.data.setdefault("evolution_runs", [])
    store.data.setdefault("experiments", [])


def select_parent(store: StrategyRegistry, symbol: str, timeframe: str) -> Dict[str, Any]:
    parent = store.best_parent(symbol, timeframe)
    if parent:
        return parent
    return {
        "strategy_id": "seed_mean_reversion",
        "parameters": {
            "entry_mode": "mean_reversion",
            "use_trend_filter": False,
            "use_structure_filter": False,
            "use_htf_filter": False,
            "use_reclaim_filter": True,
            "use_volume_filter": False,
            "min_adx": 5.0,
            "min_bb_rank": 0.08,
            "min_atr_rank": 0.08,
            "rsi_max": 30.0,
            "stop_atr_mult": 1.6,
            "tp1_rr": 1.8,
            "max_bars_override": 18,
            "cooldown_bars": 16,
            "confidence": 0.6,
            "size_multiplier": 0.85,
            "tp1_close_fraction": 0.2,
            "tp2_close_fraction": 0.3,
        },
        "score": 0.0,
        "notes": "",
    }


def make_status(passed: bool, wf_passed: bool, hard_reasons: Sequence[str]) -> str:
    if passed and wf_passed:
        return "validated"
    if hard_reasons:
        return "rejected"
    return "candidate"


def evaluate_one(
    evaluator: Callable[..., Any],
    store: StrategyRegistry,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    cfg: GateConfig,
    iteration: int,
    candidate_idx: int,
    rng_seed: int,
) -> Tuple[CandidateResult, Dict[str, Any]]:
    rng = random.Random(rng_seed)
    parent = select_parent(store, symbol, timeframe)
    family = choose_mutation_family(parent, iteration, rng)
    parameters = mutate_parameters(parent.get("parameters", {}), family, rng)
    child_id = build_candidate_id(symbol, timeframe, store.next_evolution_id(), rng)

    evaluation = call_evaluator(
        evaluator,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        parameters=parameters,
    )

    eval_error = evaluation.get("error") if isinstance(evaluation, dict) else None
    walk_forward = evaluation.get("walk_forward", {}) or {} if isinstance(evaluation, dict) else {}
    passed, hard_reasons, gate_state = evaluate_gates(evaluation if isinstance(evaluation, dict) else {}, walk_forward, cfg)
    score = compute_ranking_score(evaluation if isinstance(evaluation, dict) else {})
    status = "errored" if eval_error else make_status(passed, bool(walk_forward.get("passed", False)), hard_reasons)

    payload = {
        "id": store.next_evolution_id(),
        "cycle_id": f"iter_{iteration}",
        "child_strategy_id": child_id,
        "parent_strategy_id": parent.get("strategy_id"),
        "created_at": utc_now(),
        "symbol": symbol,
        "timeframe": timeframe,
        "status": status,
        "passed": passed,
        "score": score,
        "metrics": evaluation,
        "parameters": parameters,
        "notes": ", ".join(hard_reasons) if hard_reasons else "",
        "evaluation_error": eval_error,
    }

    result = CandidateResult(
        strategy_id=child_id,
        parent_strategy_id=parent.get("strategy_id"),
        child_strategy_id=child_id,
        status=status,
        passed=passed,
        score=score,
        wf_passed=bool(walk_forward.get("passed", False)),
        reasons=hard_reasons,
        metrics={**(evaluation if isinstance(evaluation, dict) else {}), "gate_state": gate_state, "evaluation_error": eval_error},
        parameters=parameters,
        created_at=utc_now(),
        cycle_id=f"iter_{iteration}",
        symbol=symbol,
        timeframe=timeframe,
    )
    return result, payload


def print_candidate(result: CandidateResult) -> None:
    backtest = result.metrics.get("backtest", {}) or {}
    walk_forward = result.metrics.get("walk_forward", {}) or {}
    monte_carlo = result.metrics.get("monte_carlo", {}) or {}
    print({
        "iteration": int(result.cycle_id.split("_")[-1]),
        "best_strategy": result.child_strategy_id,
        "score": result.score,
        "passed": result.passed,
        "status": result.status,
        "reasons": result.reasons,
        "return_pct": round(safe_float(backtest.get("return_pct"), 0.0), 4),
        "max_dd": round(safe_float(backtest.get("max_drawdown_pct"), 0.0), 4),
        "pf": round(safe_float(backtest.get("profit_factor"), 0.0), 4),
        "wr": round(safe_float(backtest.get("win_rate"), 0.0), 3),
        "wf_passed": result.wf_passed,
        "gate_state": result.metrics.get("gate_state", {}),
        "wf_score": round(safe_float(walk_forward.get("score"), safe_float(walk_forward.get("composite"), 0.0)), 6),
        "mc_dd": round(abs(safe_float(monte_carlo.get("worst_drawdown_pct"), 0.0)), 4),
        "evaluation_error": result.metrics.get("evaluation_error"),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evolution agent with explicit gate logging")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pf", type=float, default=0.95)
    parser.add_argument("--min-wr", type=float, default=0.45)
    parser.add_argument("--min-return-pct", type=float, default=0.0)
    parser.add_argument("--max-dd-pct", type=float, default=15.0)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--max-mc-dd-pct", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = GateConfig(
        min_profit_factor=args.min_pf,
        min_win_rate=args.min_wr,
        min_return_pct=args.min_return_pct,
        max_drawdown_pct=args.max_dd_pct,
        min_trades=args.min_trades,
        max_mc_drawdown_pct=args.max_mc_dd_pct,
    )
    evaluator = resolve_evaluator()
    store = StrategyRegistry(STORE_PATH)
    ensure_store_schema(store)

    print({
        "git_revision": git_revision(),
        "evaluator": describe_evaluator(evaluator),
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start": args.start,
        "end": args.end,
        "iterations": args.iterations,
        "candidates": args.candidates,
        "workers": args.workers,
        "gates": asdict(cfg),
    })

    rng = random.Random(args.seed)
    for iteration in range(1, args.iterations + 1):
        futures = []
        results: List[CandidateResult] = []
        payloads: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for candidate_idx in range(max(1, args.candidates)):
                seed = rng.randint(0, 10**9)
                futures.append(executor.submit(
                    evaluate_one,
                    evaluator,
                    store,
                    args.symbol,
                    args.timeframe,
                    args.start,
                    args.end,
                    cfg,
                    iteration,
                    candidate_idx,
                    seed,
                ))
            for future in as_completed(futures):
                result, payload = future.result()
                results.append(result)
                payloads.append(payload)

        results.sort(key=lambda r: (r.score, float(r.metrics.get("backtest", {}).get("profit_factor", 0.0))), reverse=True)
        payloads.sort(key=lambda p: (float(p.get("score", 0.0)), float(p.get("metrics", {}).get("backtest", {}).get("profit_factor", 0.0))), reverse=True)
        for payload in payloads:
            store.append_run(payload)
        print_candidate(results[0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
