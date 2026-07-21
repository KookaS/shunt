#!/usr/bin/env python3
"""Validate benchmark benchmark.yaml against the model registry (``--config PATH`` to
override the default path). Exit 0 = valid, 1 = validation errors found.
"""

from __future__ import annotations

import argparse
import sys

from benchmark import config


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate benchmark benchmark.yaml")
    ap.add_argument("--config", default="benchmark/benchmark.yaml", help="Path to benchmark.yaml")
    ap.add_argument("--dry", action="store_true", help="Alias for running validation")
    args = ap.parse_args()

    errors = config.validate(args.config)
    if errors:
        print(f"❌ Config validation FAILED ({len(errors)} error(s)):")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)
    else:
        print("✅ Config validation PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
