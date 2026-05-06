from __future__ import annotations

import importlib
import sys

MODULES = [
    "execution.backtest.core",
    "registry.store",
    "research.scoring",
    "research.candidate_generator",
    "research.feedback",
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
        from registry.store import list_strategies, rank_strategies  # noqa: F401
    except Exception as e:
        errors.append(f"missing registry functions: {e}")

    try:
        from strategy import generate_signal  # noqa: F401
    except Exception as e:
        errors.append(f"missing generate_signal: {e}")

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
