"""Command-line entry point for vramfit."""

import argparse
from collections.abc import Sequence

from vramfit import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vramfit",
        description="Estimate whether an LLM inference workload will fit in GPU memory.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
