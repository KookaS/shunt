"""CLI entry point for shunt-router."""

from __future__ import annotations

import argparse
import sys

from shunt import __version__


def _start(args: argparse.Namespace) -> None:
    from shunt.proxy.server import run

    run()


def _version(args: argparse.Namespace) -> None:
    print(f"shunt-router {__version__}")


def _explain(args: argparse.Namespace) -> None:
    print("shunt explain: not yet implemented (Step 2 of the roadmap)")
    sys.exit(1)


def _flag(args: argparse.Namespace) -> None:
    print("shunt flag: not yet implemented (Step 2 of the roadmap)")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shunt",
        description="Tool-agnostic, cache-safe LLM router.",
    )
    parser.set_defaults(func=_start)

    sub = parser.add_subparsers(title="commands")

    start = sub.add_parser("start", help="Start the proxy server (default)")
    start.set_defaults(func=_start)

    explain = sub.add_parser("explain", help="Explain a routing decision")
    explain.add_argument("session_id", help="Session ID to explain")
    explain.set_defaults(func=_explain)

    flag = sub.add_parser("flag", help="Flag a session outcome as good or bad")
    flag.add_argument("session_id", help="Session ID to flag")
    flag.add_argument("rating", choices=["good", "bad"], help="Outcome rating")
    flag.set_defaults(func=_flag)

    version = sub.add_parser("version", help="Print version")
    version.set_defaults(func=_version)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
