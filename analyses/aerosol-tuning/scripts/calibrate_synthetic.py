"""Derive the frozen FABLE policy record from calibration-only synthetic runs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from davinci_monet.tests.synthetic.fable_calibration_runner import (
    CALIBRATION_SEED,
    NULL_SEED,
    run_frozen_calibration,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_root", type=Path, help="New directory for immutable run evidence")
    parser.add_argument("destination", type=Path, help="New calibration record path")
    parser.add_argument("--calibration-seed", type=int, default=CALIBRATION_SEED)
    parser.add_argument("--null-seed", type=int, default=NULL_SEED)
    args = parser.parse_args(argv)
    try:
        path = run_frozen_calibration(
            args.work_root,
            args.destination,
            calibration_seed=args.calibration_seed,
            null_seed=args.null_seed,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
