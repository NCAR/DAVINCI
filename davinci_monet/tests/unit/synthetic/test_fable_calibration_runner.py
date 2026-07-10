"""Tests for the production-coupled FABLE calibration evidence runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from davinci_monet.pipeline.runner import PipelineResult
from davinci_monet.pipeline.stages.base import PipelineContext
from davinci_monet.tests.synthetic import fable_calibration_runner as runner
from davinci_monet.tests.synthetic._aerosol_policy import (
    apply_fitting_policy,
    fitting_policy_values,
)
from davinci_monet.tests.synthetic.aerosol_calibration import (
    EVIDENCE_HASH_KEYS,
    CalibrationCandidate,
    CalibrationEvidence,
    CandidateMetrics,
    ScientificPolicy,
    load_calibration_record,
    select_calibration_policy,
)


def _evidence() -> CalibrationEvidence:
    return CalibrationEvidence(
        calibration_seed=1,
        null_seed=2,
        calibration_scenario="writer_ci",
        null_scenario="calibration_null",
        calibration_split="calibration",
        hashes=tuple((name, "a" * 64) for name in EVIDENCE_HASH_KEYS),
    )


def _metrics(nrmse: float) -> CandidateMetrics:
    return CandidateMetrics(
        field_correlation=0.95,
        field_origin_slope=1.0,
        field_nrmse=nrmse,
        aod_rmse_ratio=0.5,
        full_target_aod_rmse_ratio=0.7,
        null_retained_energy_fraction=0.02,
        null_significant_fraction=0.03,
    )


def test_default_calibration_runs_and_selects_multiple_predeclared_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def evaluate(
        root: Path,
        policy: ScientificPolicy,
        *,
        calibration_seed: int,
        null_seed: int,
    ) -> CalibrationCandidate:
        del root, calibration_seed, null_seed
        seen.append(policy.policy_id)
        nrmse = 0.1 if policy.policy_id == "fable-v1-significant" else 0.2
        return CalibrationCandidate(policy, _metrics(nrmse), _evidence())

    monkeypatch.setattr(runner, "_evaluate_candidate", evaluate)

    path = runner.run_frozen_calibration(tmp_path / "work", tmp_path / "record.json")
    record = load_calibration_record(path)

    assert seen == [policy.policy_id for policy in runner.calibration_policy_candidates()]
    assert len(record.candidates) == 3
    assert record.selected_policy_id == "fable-v1-significant"


def test_every_candidate_round_trips_through_the_fitting_template() -> None:
    template_path = (
        Path(__file__).parents[4]
        / "analyses"
        / "aerosol-tuning"
        / "configs"
        / "fable-synthetic.example.yaml"
    )
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    for policy in runner.calibration_policy_candidates():
        rendered = deepcopy(template)
        apply_fitting_policy(rendered, policy)
        expected = policy.normalized()
        expected.pop("policy_id")
        expected.pop("simplicity_rank")
        assert fitting_policy_values(rendered) == expected


def test_null_gate_uses_equal_day_weights_exact_band_coi_and_matched_modes() -> None:
    time = pd.date_range("2005-01-01", periods=4, freq="1D")
    lat = [0.0]
    lon = [0.0, 180.0]
    estimate_patterns = xr.DataArray(
        [[[1.0, -1.0]], [[1.0, 1.0]]],
        dims=("mode", "lat", "lon"),
        coords={"mode": [1, 2], "lat": lat, "lon": lon},
    )
    truth_patterns = xr.DataArray(
        np.stack((estimate_patterns.values[1], estimate_patterns.values[0])),
        dims=("truth_mode", "mode_lat", "mode_lon"),
        coords={"truth_mode": [1, 2], "mode_lat": lat, "mode_lon": lon},
    )
    applied = np.zeros((4, 1, 2))
    applied[1] = 1.0
    support = np.ones_like(applied)
    support[1, 0, 1] = 0.0
    scaling = xr.Dataset(
        {
            "delta_log_applied": (("time", "lat", "lon"), applied),
            "spatial_support": (("time", "lat", "lon"), support),
            "coefficient_available": ("time", [True, True, True, False]),
            "eofs": estimate_patterns,
        },
        coords={"time": time, "lat": lat, "lon": lon, "mode": [1, 2]},
    ).assign_coords(month=("time", [1, 1, 1, 1]))
    filtered = xr.Dataset(
        {
            "valid_segment": (("time", "mode"), np.ones((4, 2), dtype=bool)),
            "coi": (
                ("time", "mode"),
                np.array([[1.0, 179.0], [1.0, 180.0], [1.0, 180.0], [1.0, 180.0]]),
            ),
            "power_significance": (
                ("time", "mode", "period"),
                np.zeros((4, 2, 3)),
            ),
        },
        coords={"time": time, "mode": [1, 2], "period": [4.0, 100.0, 180.0]},
    )
    truth = xr.Dataset(
        {
            "pattern_true": truth_patterns,
            "mode_observable_true": ("truth_mode", [True, False]),
            "innovation_noise_true": (("time", "mode_lat", "mode_lon"), np.ones((4, 1, 2))),
            "split": ("time", ["calibration"] * 4),
            "monthly_marker": ("month", [1.0]),
        },
        coords={
            "time": time,
            "truth_mode": [1, 2],
            "mode_lat": lat,
            "mode_lon": lon,
            "month": [1],
        },
    )

    result = runner.evaluate_null_control(
        scaling,
        filtered,
        truth,
        ScientificPolicy(policy_id="test"),
        split="calibration",
    )

    assert result["scored_day_count"] == 2.0
    assert result["null_retained_energy_fraction"] == pytest.approx(0.5)
    assert result["null_significant_fraction"] == 0.0


@pytest.mark.parametrize(
    "changed_name",
    [
        "pyproject.toml",
        "_aerosol_bundle.py",
        "fable-synthetic.example.yaml",
        "fable-synthetic-eval.example.yaml",
    ],
)
def test_calibration_code_identity_covers_environment_and_generator(
    changed_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = runner.calibration_code_sha256()
    original = Path.read_bytes

    def changed(path: Path) -> bytes:
        value = original(path)
        return value + b"\n# changed" if path.name == changed_name else value

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert runner.calibration_code_sha256() != baseline


def test_frozen_record_validation_binds_design_seeds_and_current_code() -> None:
    code_hash = runner.calibration_code_sha256()
    candidates = []
    for index, policy in enumerate(runner.calibration_policy_candidates()):
        hashes = {
            name: code_hash if name == "code_sha256" else "a" * 64 for name in EVIDENCE_HASH_KEYS
        }
        evidence = CalibrationEvidence(
            calibration_seed=runner.CALIBRATION_SEED,
            null_seed=runner.NULL_SEED,
            calibration_scenario="writer_ci",
            null_scenario="calibration_null",
            calibration_split=runner.CALIBRATION_SPLIT,
            hashes=tuple(hashes.items()),
        )
        candidates.append(CalibrationCandidate(policy, _metrics(0.1 + index * 0.01), evidence))
    record = select_calibration_policy(candidates)

    assert runner.validate_frozen_calibration_record(record) is record

    wrong_evidence = replace(record.candidates[0].evidence, calibration_seed=1)
    wrong_candidate = replace(record.candidates[0], evidence=wrong_evidence)
    wrong_record = select_calibration_policy((wrong_candidate, *record.candidates[1:]))
    with pytest.raises(ValueError, match="fixed design"):
        runner.validate_frozen_calibration_record(wrong_record)

    missing_record = select_calibration_policy(record.candidates[1:])
    with pytest.raises(ValueError, match="candidate set"):
        runner.validate_frozen_calibration_record(missing_record)


def test_candidate_cleanup_runs_when_null_scenario_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = False
    truth = xr.Dataset()

    def mark_closed() -> None:
        nonlocal closed
        closed = True

    truth.set_close(mark_closed)
    context = PipelineContext()
    completed = runner._ScenarioRun(
        root=tmp_path,
        spec_sha256="a" * 64,
        config_sha256="b" * 64,
        manifest_sha256="c" * 64,
        result=cast(PipelineResult, SimpleNamespace(context=context)),
        truth=truth,
    )
    calls = 0

    def scenario(*args: object, **kwargs: object) -> runner._ScenarioRun:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            return completed
        raise RuntimeError("null failed")

    monkeypatch.setattr(runner, "_run_scenario", scenario)

    with pytest.raises(RuntimeError, match="null failed"):
        runner._evaluate_candidate(
            tmp_path,
            ScientificPolicy(policy_id="test"),
            calibration_seed=1,
            null_seed=2,
        )
    assert closed is True
