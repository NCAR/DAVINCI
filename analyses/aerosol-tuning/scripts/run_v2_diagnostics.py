"""Run the fixed FABLE v2 synthetic development diagnostic campaign."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from davinci_monet.tests.synthetic.fable_v2_development import run_v2_development


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="New locked development evidence root")
    parser.add_argument("--dry-run", action="store_true", help="Lock and validate without arrays")
    args = parser.parse_args(argv)
    try:
        path = run_v2_development(args.root, dry_run=bool(args.dry_run))
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(path)
    document = json.loads(path.read_text(encoding="ascii"))
    return 0 if document.get("status") in {"planned", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
