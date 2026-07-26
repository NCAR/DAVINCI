#!/usr/bin/env python3
"""Run DAVINCI's public strict readiness validator without changing run state."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit one DAVINCI YAML control with strict schema and readiness checks.",
        epilog=(
            "Input: one config path. Output: human or JSON readiness evidence on stdout. "
            "Mutations: none. Safety: never runs the pipeline, allocates an attempt, or submits PBS."
        ),
    )
    parser.add_argument("config", type=Path, help="DAVINCI YAML control to audit.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only the machine-readable readiness report.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Audit resumption of an existing incomplete attempt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[3]
    davinci_cli = Path(sys.executable).with_name("davinci")
    if not davinci_cli.is_file():
        print(
            f"error: DAVINCI CLI not found beside the active Python: {davinci_cli}",
            file=sys.stderr,
        )
        return 2
    command = [
        str(davinci_cli),
        "validate",
        str(config_path),
        "--strict",
        "--readiness",
    ]
    if args.json:
        command.append("--json")
    if args.resume:
        command.append("--resume")
    return subprocess.run(command, cwd=repository_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
