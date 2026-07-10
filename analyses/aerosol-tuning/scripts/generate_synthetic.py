"""Generate a deterministic FABLE synthetic development bundle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from davinci_monet.tests.synthetic.aerosol_tuning import (
    SyntheticTuningSpec,
    generate_aerosol_tuning_bundle,
    write_aerosol_tuning_bundle,
)

ScenarioFactory = Callable[[int], SyntheticTuningSpec]
SCENARIOS: dict[str, ScenarioFactory] = {
    "exact_micro": SyntheticTuningSpec.exact_micro,
    "masked_chain_ci": SyntheticTuningSpec.masked_chain_ci,
    "multi_sensor_ci": SyntheticTuningSpec.multi_sensor_ci,
    "writer_ci": SyntheticTuningSpec.writer_ci,
    "calibration_null": SyntheticTuningSpec.calibration_null,
    "null_ci": SyntheticTuningSpec.null_ci,
    "low_aod_ci": SyntheticTuningSpec.low_aod_ci,
    "synthetic_osse": SyntheticTuningSpec.synthetic_osse,
}
DEVELOPMENT_SEED = 20260712


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Destination root for inputs/ and oracle/")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="masked_chain_ci")
    parser.add_argument(
        "--seed",
        type=int,
        help="Root seed; required for synthetic_osse and never inferred for acceptance runs",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Print the normalized scenario contract without generating arrays",
    )
    args = parser.parse_args(argv)

    if args.scenario == "synthetic_osse" and args.seed is None:
        parser.error("--seed is required for synthetic_osse")
    seed = DEVELOPMENT_SEED if args.seed is None else args.seed
    spec = SCENARIOS[args.scenario](seed)
    if args.validate_only:
        print(json.dumps(spec.normalized(), sort_keys=True))
        return 0
    bundle = generate_aerosol_tuning_bundle(spec)
    manifest = write_aerosol_tuning_bundle(args.root, bundle, overwrite=bool(args.overwrite))
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
