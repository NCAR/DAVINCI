"""Run the fixed multi-seed FABLE v2 synthetic calibration."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from davinci_monet.tests.synthetic.fable_v2_calibration_runner import run_v2_calibration
from davinci_monet.tests.synthetic.fable_v2_identity import load_current_v2_preregistration


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_root", type=Path, help="New locked calibration evidence root")
    parser.add_argument("destination", type=Path, help="New frozen calibration record")
    parser.add_argument("preregistration", type=Path, help="Frozen v2 preregistration")
    parser.add_argument("--dry-run", action="store_true", help="Lock and validate without arrays")
    args = parser.parse_args(argv)
    try:
        current, frozen = load_current_v2_preregistration(args.preregistration)
        path = run_v2_calibration(
            args.work_root,
            args.destination,
            current,
            frozen,
            dry_run=bool(args.dry_run),
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
