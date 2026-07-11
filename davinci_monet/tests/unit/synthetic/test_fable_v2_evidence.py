"""Adversarial derivation tests for frozen FABLE v2 scenario evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr

import davinci_monet.tests.synthetic.fable_v2_artifact_evidence as artifact_evidence
import davinci_monet.tests.synthetic.fable_v2_evidence as scenario_evidence
import davinci_monet.tests.synthetic.fable_v2_runner as runner_module
from davinci_monet.analysis.artifacts import write_dataset_collection
from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec, spec_hash
from davinci_monet.tests.synthetic._aerosol_io import scientific_dataset_hash
from davinci_monet.tests.synthetic.fable_acceptance_gate import (
    RECOVERY_METRICS,
    REQUIRED_RECOVERY_DIAGNOSTICS,
    REQUIRED_RECOVERY_STRATA,
    resource_gate,
)
from davinci_monet.tests.synthetic.fable_v2_freeze import FrozenFileIdentity
from davinci_monet.tests.synthetic.fable_v2_policy import v2_calibration_policies
from davinci_monet.tests.synthetic.fable_v2_record_io import json_sha256
from davinci_monet.tests.synthetic.fable_v2_runner import V2ScenarioOutcome
from davinci_monet.tests.synthetic.fable_v2_scoring import (
    v2_null_score,
    v2_recovery_score,
)


def _identity(tmp_path: Path, name: str, text: str = "evidence\n") -> FrozenFileIdentity:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")
    return FrozenFileIdentity.capture(path)


def _recovery_report() -> xr.Dataset:
    strata = list(REQUIRED_RECOVERY_STRATA)
    values: dict[str, tuple[str, np.ndarray]] = {}
    defaults = {
        "field_correlation": 0.96,
        "field_origin_slope": 1.0,
        "field_nrmse": 0.2,
        "aod_rmse_ratio": 0.5,
        "full_target_aod_rmse_ratio": 0.7,
        "valid_count": 100.0,
        "candidate_count": 120.0,
        "excluded_fraction": 1.0 / 6.0,
        "best_representable_nrmse": 0.1,
        "off_basis_floor_nrmse": 0.05,
        "full_delta_nrmse": 0.2,
        "in_span_truth_rms": 0.4,
        "perpendicular_truth_rms": 0.02,
        "perpendicular_to_full_rms_ratio": 0.05,
        "clip_fraction": 0.0,
        "holdout_aod_rmse_ratio": 0.6,
        "subspace_projector_error": 0.01,
    }
    for name in (*RECOVERY_METRICS, *REQUIRED_RECOVERY_DIAGNOSTICS):
        values[name] = ("stratum", np.full(len(strata), defaults.get(name, 1.0)))
    return xr.Dataset(values, coords={"stratum": strata})


def _null_inputs() -> tuple[xr.Dataset, xr.Dataset]:
    time = np.arange("2006-01-01", "2006-01-05", dtype="datetime64[D]")
    mode = [1]
    lat = [-30.0, 30.0]
    lon = [0.0, 180.0]
    period = [20.0, 60.0]
    pattern = np.asarray([[[1.0, -1.0], [1.0, -1.0]]])
    scaling = xr.Dataset(
        {
            "eofs": (("mode", "lat", "lon"), pattern),
            "spatial_support": (
                ("time", "lat", "lon"),
                np.ones((len(time), len(lat), len(lon))),
            ),
            "delta_log_applied": (
                ("time", "lat", "lon"),
                np.full((len(time), len(lat), len(lon)), 0.005),
            ),
            "coefficient_available": ("time", np.ones(len(time), dtype=bool)),
            "valid_segment": (("time", "mode"), np.ones((len(time), 1), dtype=bool)),
            "coi": (("time", "mode"), np.full((len(time), 1), 180.0)),
            "power_significance": (
                ("time", "mode", "period"),
                np.full((len(time), 1, len(period)), 0.5),
            ),
        },
        coords={"time": time, "mode": mode, "lat": lat, "lon": lon, "period": period},
    )
    truth = xr.Dataset(
        {
            "split": ("time", ["calibration"] * len(time)),
            "pattern_true": (("truth_mode", "lat", "lon"), pattern),
            "mode_observable_true": ("truth_mode", [1]),
            "innovation_noise_true": (
                ("time", "lat", "lon"),
                np.full((len(time), len(lat), len(lon)), 0.1),
            ),
        },
        coords={"time": time, "truth_mode": mode, "lat": lat, "lon": lon},
    )
    return scaling, truth


def _outcome(tmp_path: Path) -> V2ScenarioOutcome:
    policy = v2_calibration_policies()[0]
    seed = 4720161833425845668
    spec = SyntheticTuningSpec.synthetic_osse(seed)
    identity = _identity(tmp_path, "manifest.json", '{"status":"completed"}\n')
    score = v2_recovery_score(_recovery_report())
    return V2ScenarioOutcome(
        seed=seed,
        scenario=spec.scenario,
        policy=policy,
        score_kind="recovery",
        evaluation_splits=("calibration",),
        spec_sha256=spec_hash(spec),
        scenario_manifest=identity,
        fitting_config=identity,
        fitting_manifest=identity,
        evaluation_config=identity,
        evaluation_manifest=identity,
        report_sha256="a" * 64,
        diagnostic_report_sha256="b" * 64,
        learned_basis_oracle_nrmse=0.03,
        score=score,
        evidence_passed=True,
        elapsed_seconds=12.5,
        process_peak_rss_bytes=4096,
        resources=resource_gate(12.5, 4096),
    )


def test_scenario_evidence_rejects_resource_and_derived_passed_tampering(tmp_path: Path) -> None:
    outcome = _outcome(tmp_path)
    spec = SyntheticTuningSpec.synthetic_osse(outcome.seed)
    value = outcome.normalized()

    changed = {**value, "passed": False}
    with pytest.raises(ValueError, match="passed state is not derived"):
        scenario_evidence.validate_v2_scenario_evidence(
            changed,
            spec,
            outcome.policy,
            score_kind="recovery",
            evaluation_splits=("calibration",),
            verify_files=False,
        )

    resources = dict(value["resources"])
    resources["elapsed_seconds"] = 1.0
    changed = {**value, "resources": resources}
    with pytest.raises(ValueError, match="resources do not match"):
        scenario_evidence.validate_v2_scenario_evidence(
            changed,
            spec,
            outcome.policy,
            score_kind="recovery",
            evaluation_splits=("calibration",),
            verify_files=False,
        )


def test_scenario_evidence_rejects_score_and_oracle_artifact_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = _outcome(tmp_path)
    spec = SyntheticTuningSpec.synthetic_osse(outcome.seed)
    value = outcome.normalized()
    derived = {
        "score": value["score"],
        "learned_basis_oracle_nrmse": value["learned_basis_oracle_nrmse"],
    }
    monkeypatch.setattr(scenario_evidence, "validate_v2_artifact_evidence", lambda *a, **k: derived)

    changed_score = {**value, "score": {**value["score"], "passed": False}, "passed": False}
    with pytest.raises(ValueError, match="score differs"):
        scenario_evidence.validate_v2_scenario_evidence(
            changed_score,
            spec,
            outcome.policy,
            score_kind="recovery",
            evaluation_splits=("calibration",),
            verify_files=True,
        )

    changed_oracle = {**value, "learned_basis_oracle_nrmse": 0.04}
    with pytest.raises(ValueError, match="oracle NRMSE differs"):
        scenario_evidence.validate_v2_scenario_evidence(
            changed_oracle,
            spec,
            outcome.policy,
            score_kind="recovery",
            evaluation_splits=("calibration",),
            verify_files=True,
        )


def test_deep_artifact_derivation_uses_exact_recovery_and_null_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = v2_calibration_policies()[0]
    identity = _identity(tmp_path, "bundle/scenario.json")
    report = _recovery_report()
    diagnostic = xr.Dataset({"learned_basis_oracle_nrmse": xr.DataArray(0.03)})
    scaling, truth = _null_inputs()
    truth_path = tmp_path / "bundle/oracle/truth.nc"
    truth_path.parent.mkdir(parents=True)
    truth.to_netcdf(truth_path)

    monkeypatch.setattr(artifact_evidence, "validate_v2_scenario_bundle", lambda *a, **k: None)
    monkeypatch.setattr(artifact_evidence, "_manifest_entries", lambda identity: ())
    monkeypatch.setattr(artifact_evidence, "_validate_mmr_entries", lambda *a, **k: None)

    entries = {
        "aod_basis": {"tag": "basis"},
        "obs_pcs": {"tag": "projection"},
        "scaling": {"tag": "scaling"},
        "recovery": {"tag": "recovery"},
        "v2_diagnostics": {"tag": "diagnostic"},
    }
    monkeypatch.setattr(
        artifact_evidence,
        "_validate_artifact_entry",
        lambda manifest, items, role, analysis: entries[analysis],
    )
    monkeypatch.setattr(
        artifact_evidence,
        "_load_artifact",
        lambda entry: {
            "scaling": scaling.copy(),
            "recovery": report.copy(),
            "diagnostic": diagnostic.copy(),
        }[entry["tag"]],
    )

    common: dict[str, Any] = {
        "scenario_manifest": identity.normalized(),
        "fitting_manifest": identity.normalized(),
        "evaluation_manifest": identity.normalized(),
        "report_sha256": scientific_dataset_hash(report),
        "diagnostic_report_sha256": scientific_dataset_hash(diagnostic),
    }
    recovery = artifact_evidence.validate_v2_artifact_evidence(
        common,
        SyntheticTuningSpec.synthetic_osse(1),
        policy,
        recovery=True,
    )
    assert recovery["score"] == v2_recovery_score(report)
    assert recovery["learned_basis_oracle_nrmse"] == 0.03

    null_score = v2_null_score(scaling, scaling, truth, policy)
    assert null_score == v2_null_score(scaling, scaling.copy(), truth, policy)
    null = artifact_evidence.validate_v2_artifact_evidence(
        {
            **common,
            "report_sha256": json_sha256(null_score),
        },
        SyntheticTuningSpec.synthetic_osse_null(1),
        policy,
        recovery=False,
    )
    assert null["score"] == null_score


def test_runner_hash_source_is_the_finalized_netcdf_artifact(tmp_path: Path) -> None:
    live = _recovery_report()
    live.attrs["derived"] = True
    collection = write_dataset_collection(tmp_path, "recovery", live)
    entry = {
        "analysis": "recovery",
        "artifact_dir": str(collection.root),
        "checksums": {"files": dict(collection.checksums)},
        "kind": "netcdf_collection",
        "role": "recovery_report",
    }
    result: Any = SimpleNamespace(context=SimpleNamespace(metadata={"analysis_artifacts": [entry]}))

    finalized = runner_module._loaded_finalized_artifact(result, "recovery", "recovery_report")

    assert finalized.attrs["derived"] == "True"
    assert scientific_dataset_hash(finalized) != scientific_dataset_hash(live)
    assert runner_module._recovery_artifact(result) == entry
