"""Run the exact protocol-owned FABLE v2 synthetic acceptance tuple once."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from davinci_monet.tests.synthetic.fable_v2_acceptance import (
    load_v2_acceptance_record,
    run_v2_acceptance,
    validate_v2_acceptance_record,
)
from davinci_monet.tests.synthetic.fable_v2_freeze import FrozenFileIdentity
from davinci_monet.tests.synthetic.fable_v2_identity import load_current_v2_preregistration


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="New locked acceptance evidence root")
    parser.add_argument("preregistration", type=Path, help="Frozen v2 preregistration")
    parser.add_argument("calibration", type=Path, help="Frozen v2 calibration record")
    parser.add_argument("preflight", type=Path, help="Frozen passing v2 preflight record")
    parser.add_argument("--dry-run", action="store_true", help="Lock and validate without arrays")
    args = parser.parse_args(argv)
    try:
        current, frozen = load_current_v2_preregistration(args.preregistration)
        path = run_v2_acceptance(
            args.root,
            current,
            frozen,
            args.calibration,
            args.preflight,
            dry_run=bool(args.dry_run),
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(path)
    if args.dry_run:
        status = "planned"
    else:
        record = validate_v2_acceptance_record(
            load_v2_acceptance_record(path),
            current,
            frozen,
            FrozenFileIdentity.capture(args.calibration),
            FrozenFileIdentity.capture(args.preflight),
        )
        status = record.status
    return 0 if status in {"planned", "passed_pending_user_review"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
