from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULES = [
    "execution.backtest.core",
    "execution.deployment",
    "registry.store",
    "research.scoring",
    "research.candidate_generator",
    "research.feedback",
    "research.monte_carlo",
    "strategy",
]


def check_imports() -> list[str]:
    errors: list[str] = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:
            errors.append(f"import failed: {name}: {e}")
    return errors


def check_functions() -> list[str]:
    errors: list[str] = []
    try:
        from execution.backtest.core import run_backtest  # noqa: F401
    except Exception as e:
        errors.append(f"missing run_backtest: {e}")

    try:
        from execution.deployment import build_deployment_plan  # noqa: F401
    except Exception as e:
        errors.append(f"missing deployment: {e}")

    try:
        from research.monte_carlo import run_monte_carlo  # noqa: F401
    except Exception as e:
        errors.append(f"missing monte carlo: {e}")

    return errors


def main() -> int:
    errs = []
    errs.extend(check_imports())
    errs.extend(check_functions())

    if errs:
        print("SANITY CHECK FAILED")
        for e in errs:
            print(" -", e)
        return 1

    print("SANITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
