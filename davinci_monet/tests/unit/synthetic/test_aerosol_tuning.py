"""Contract tests for the coupled aerosol-tuning synthetic generator."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import davinci_monet.tests.synthetic as synthetic_module
import davinci_monet.tests.synthetic._aerosol_acceptance as acceptance_impl
import davinci_monet.tests.synthetic._aerosol_io as aerosol_io_impl
import davinci_monet.tests.synthetic.aerosol_tuning as tuning_module
from davinci_monet.tests.synthetic.aerosol_tuning import (
    DEFAULT_AEROSOL_SPECIES,
    NAMED_STREAMS,
    SyntheticTuningSpec,
    analytic_temporal_filter_target,
    area_weighted_regrid_oracle,
    build_synthetic_acceptance_plan,
    evaluate_synthetic_recovery_gate,
    generate_aerosol_tuning_bundle,
    local_overpass_oracle,
    named_rng,
    named_stream_id,
    optical_aod_oracle,
    periodic_bilinear_oracle,
    render_synthetic_acceptance_configs,
    run_synthetic_acceptance,
    scientific_dataset_hash,
    shifted_log_ratio_oracle,
    write_aerosol_tuning_bundle,
)


@pytest.fixture(scope="module")
def exact_bundle():
    """Generate the shared exact algebra case once."""
    return generate_aerosol_tuning_bundle(SyntheticTuningSpec.exact_micro())


@pytest.fixture(scope="module")
def masked_bundle():
    """Generate the shared masked case once."""
    return generate_aerosol_tuning_bundle(SyntheticTuningSpec.masked_chain_ci())


def _passing_recovery_report(
    *, field_correlation: float = 0.95, excluded_fraction: float = 0.75
) -> xr.Dataset:
    strata = list(acceptance_impl.REQUIRED_RECOVERY_STRATA)
    count = len(strata)

    def repeated(value: float) -> np.ndarray:
        return np.full(count, value, dtype=np.float64)

    return xr.Dataset(
        {
            "field_correlation": ("stratum", repeated(field_correlation)),
            "field_origin_slope": ("stratum", repeated(1.0)),
            "field_nrmse": ("stratum", repeated(0.2)),
            "aod_rmse_ratio": ("stratum", repeated(0.6)),
            "full_target_aod_rmse_ratio": ("stratum", repeated(0.9)),
            "valid_count": ("stratum", repeated(250.0)),
            "candidate_count": ("stratum", repeated(1000.0)),
            "excluded_fraction": ("stratum", repeated(excluded_fraction)),
            "best_representable_nrmse": ("stratum", repeated(0.15)),
            "off_basis_floor_nrmse": ("stratum", repeated(0.25)),
            "full_delta_nrmse": ("stratum", repeated(0.3)),
            "in_span_truth_rms": ("stratum", repeated(0.2)),
            "perpendicular_truth_rms": ("stratum", repeated(0.05)),
            "perpendicular_to_full_rms_ratio": ("stratum", repeated(0.2)),
            "clip_fraction": ("stratum", repeated(0.02)),
            "holdout_aod_rmse_ratio": ("stratum", repeated(0.65)),
            "subspace_projector_error": ("stratum", repeated(0.03)),
            "basis_mode_similarity": ("matched_mode", [0.99, 0.98, 0.97]),
            "matched_mode_observable": ("matched_mode", [True, True, False]),
            "coefficient_correlation": ("matched_mode", [0.98, 0.97, np.nan]),
            "coefficient_origin_slope": ("matched_mode", [0.95, 1.02, np.nan]),
            "coefficient_bias": ("matched_mode", [0.01, -0.01, np.nan]),
            "coefficient_nrmse": ("matched_mode", [0.2, 0.25, np.nan]),
        },
        coords={"stratum": strata, "matched_mode": [1, 2, 3]},
        attrs={"analysis_type": "known_truth"},
    )


def test_spec_is_frozen_validated_and_normalized() -> None:
    """The scenario contract is immutable and hashes only normalized scientific controls."""
    spec = SyntheticTuningSpec.exact_micro(master_seed=17)
    with pytest.raises(FrozenInstanceError):
        spec.master_seed = 18  # type: ignore[misc]
    normalized = spec.normalized()
    assert normalized["master_seed"] == 17
    assert [window[0] for window in normalized["split_windows"]] == [
        "basis_train",
        "bias_fit",
        "calibration",
        "development_test",
    ]
    with pytest.raises(ValueError, match="scenario"):
        SyntheticTuningSpec(scenario="real_data")
    with pytest.raises(ValueError, match="r_bounds"):
        SyntheticTuningSpec(r_bounds=(1.1, 5.0))


def test_public_facades_export_calibration_and_acceptance_contracts() -> None:
    required = {
        "CalibrationEvidence",
        "FROZEN_CALIBRATION_SHA256",
        "RECOVERY_THRESHOLDS",
        "calibration_code_sha256",
        "calibration_policy_candidates",
        "run_frozen_calibration",
        "run_synthetic_acceptance",
        "validate_frozen_calibration_record",
    }
    assert required <= set(tuning_module.__all__)
    assert required <= set(synthetic_module.__all__)
    for name in required:
        assert getattr(tuning_module, name) is getattr(synthetic_module, name)


def test_synthetic_osse_factory_locks_opt_in_stress_contract() -> None:
    """The large acceptance profile is validated without allocating its arrays."""
    spec = SyntheticTuningSpec.synthetic_osse(314159)

    assert spec.master_seed == 314159
    assert (spec.native_domain.n_lat, spec.native_domain.n_lon) == (36, 72)
    assert (spec.mode_domain.n_lat, spec.mode_domain.n_lon) == (18, 36)
    assert (
        pd.Timestamp(spec.time_config.end) - pd.Timestamp(spec.time_config.start)
    ).days + 1 == 2922
    assert spec.split_windows[-1] == (
        "development_test",
        "2007-01-01",
        "2008-12-31",
    )
    normalized = spec.normalized()
    for control in (
        "basis_drift_amplitude",
        "off_basis_amplitude",
        "out_of_band_amplitude",
        "sensor_bias_log",
        "heteroscedastic_strength",
        "error_temporal_correlation",
        "error_spatial_correlation",
        "mnar_cloud_strength",
    ):
        assert normalized[control]


def test_full_size_null_changes_only_scenario_and_zero_truth_role() -> None:
    """The v2 null keeps every full-stress generation control unchanged."""
    recovery = SyntheticTuningSpec.synthetic_osse(314159).normalized()
    null = SyntheticTuningSpec.synthetic_osse_null(314159).normalized()

    assert recovery.pop("scenario") == "synthetic_osse"
    assert null.pop("scenario") == "synthetic_osse_null"
    assert null == recovery


def test_full_osse_model_uses_bounded_dimension_aware_netcdf_chunks() -> None:
    """Chunk planning stays practical without allocating the eight-year model array."""
    spec = SyntheticTuningSpec.synthetic_osse(20260712)
    day_count = (pd.Timestamp(spec.time_config.end) - pd.Timestamp(spec.time_config.start)).days + 1
    shape = (
        (day_count + 2) * 24,
        spec.native_domain.n_lat,
        spec.native_domain.n_lon,
    )

    chunks = aerosol_io_impl._bounded_netcdf_chunks(
        ("time", "lat", "lon"), shape, np.dtype(np.float32).itemsize
    )
    chunk_bytes = math.prod(chunks) * np.dtype(np.float32).itemsize
    chunk_count = math.prod(math.ceil(size / chunk) for size, chunk in zip(shape, chunks))

    assert shape == (70176, 36, 72)
    assert chunks == (404, 36, 72)
    assert 3 * 1024**2 < chunk_bytes <= aerosol_io_impl.NETCDF_CHUNK_TARGET_BYTES
    assert chunk_count == 174


def test_analytic_temporal_oracle_applies_known_band_gaps_taper_and_observability() -> None:
    """The oracle filters known components without importing or estimating a CWT."""
    spec = SyntheticTuningSpec(
        scenario="masked_chain_ci",
        correction_periods_days=(10.0, 20.0, 30.0),
    )
    samples = 800
    in_band = np.ones((samples, spec.n_modes))
    out_of_band = np.full_like(in_band, 5.0)
    trend = np.full_like(in_band, 0.25)
    valid = np.ones((1, samples, 1, 1), dtype=bool)
    valid[:, 200:202] = False
    valid[:, 400:408] = False

    target = analytic_temporal_filter_target(
        spec,
        in_band,
        out_of_band,
        trend,
        valid,
        np.array([1, 1, 0], dtype=np.uint8),
    )

    assert target.bridged[200:202, :2].all()
    assert not target.bridged[400:408].any()
    assert target.valid_segment[200, :2].all()
    assert not target.valid_segment[400:408].any()
    np.testing.assert_allclose(target.coefficients[200, :2], 1.25)
    assert np.all(target.coefficients[:, 2] == 0.0)
    assert np.all(target.coefficients[400:408] == 0.0)
    assert np.all(target.edge_weight[0] == 0.0)


def test_acceptance_dry_run_records_and_locks_only_supplied_seeds(tmp_path: Path) -> None:
    """Dry validation records three user values without generating the large OSSE."""
    supplied = [101, 202, 303]
    record_path = run_synthetic_acceptance(tmp_path, supplied, dry_run=True)
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert record["scenario"] == "synthetic_osse"
    assert record["seeds"] == supplied
    assert record["seed_lock"]["status"] == "immutable"
    assert len(record["seed_lock"]["file"]["sha256"]) == 64
    seed_lock_path = tmp_path / "seed-lock.json"
    seed_lock = json.loads(seed_lock_path.read_text(encoding="utf-8"))
    assert seed_lock_path.stat().st_mode & 0o222 == 0
    assert seed_lock["seeds"] == supplied
    assert seed_lock["scientific_policy"]["policy_id"] == "fable-v1-all-band"
    assert record["recovery_thresholds"]["field_correlation_min"] == 0.90
    assert record["diagnostic_requirements"]["excluded_fraction_max"] == 0.80
    assert record["resource_limits"]["elapsed_seconds_max"] == 30 * 60
    assert record["resource_limits"]["process_peak_rss_bytes_max"] == 8 * 1024**3
    assert record["calibration"]["selected_policy_id"] == "fable-v1-all-band"
    assert record["calibration"]["record_sha256"] == acceptance_impl.FROZEN_CALIBRATION_SHA256
    assert record["status"] == "planned"
    assert record["runs"] == []
    assert list(tmp_path.glob("seed-[0-9]*")) == []
    assert list(tmp_path.glob(".*.tmp")) == []
    with pytest.raises(FileExistsError, match="already locked"):
        run_synthetic_acceptance(tmp_path, supplied, dry_run=True)
    with pytest.raises(ValueError, match="exactly three"):
        run_synthetic_acceptance(tmp_path / "short", supplied[:2], dry_run=True)
    with pytest.raises(ValueError, match="distinct"):
        run_synthetic_acceptance(tmp_path / "duplicate", [1, 1, 2], dry_run=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_synthetic_acceptance(tmp_path / "ambiguous", supplied, dry_run=True, generate_only=True)

    plan = build_synthetic_acceptance_plan(tmp_path / "rendered", supplied)
    fitting_path, evaluation_path = render_synthetic_acceptance_configs(
        plan, plan.root / "seed-101"
    )
    fitting = yaml.safe_load(fitting_path.read_text(encoding="utf-8"))
    evaluation = yaml.safe_load(evaluation_path.read_text(encoding="utf-8"))
    assert fitting["analysis"]["end_time"] == "2008-12-31 23:59:59"
    assert fitting["analyses"]["model_daily"]["target_grid"] == 10.0
    assert fitting["analyses"]["aod_basis"]["fit_window"]["end"].startswith("2003")
    assert fitting["analyses"]["obs_pcs"]["bias_fit_window"]["start"].startswith("2004")
    assert fitting["analyses"]["filtered_pcs"]["min_resolution"] == 0.3
    assert evaluation["analysis"]["start_time"] == "2007-01-01"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("analyses", "obs_pcs", "support_smoothing_passes"), 0),
        (("analyses", "model_daily", "log_epsilon"), 0.02),
        (("analyses", "obs_pcs", "delta_bounds"), [-1.0, 1.0]),
        (("analyses", "filtered_pcs", "omega0"), 8.0),
        (("analyses", "filtered_pcs", "dj"), 0.5),
        (("analyses", "scaling", "r_bounds"), [0.5, 2.0]),
    ],
)
def test_acceptance_rejects_fitting_template_policy_drift(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    plan = build_synthetic_acceptance_plan(tmp_path / "acceptance", [101, 202, 303])
    fitting = yaml.safe_load(plan.fitting_template.read_text(encoding="utf-8"))
    target = fitting
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    changed = tmp_path / "changed-fitting.yaml"
    changed.write_text(yaml.safe_dump(fitting, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen calibration policy"):
        render_synthetic_acceptance_configs(
            replace(plan, fitting_template=changed),
            plan.root / "seed-101",
        )


def test_recovery_gate_enforces_scientific_thresholds() -> None:
    passing = _passing_recovery_report()

    accepted = evaluate_synthetic_recovery_gate(passing)
    assert accepted["passed"] is True
    assert accepted["failures"] == []
    assert accepted["diagnostics"]["excluded_fraction"] == 0.75
    assert accepted["diagnostics"]["full_delta_nrmse"] == 0.3
    assert accepted["diagnostics"]["in_span_truth_rms"] == 0.2
    assert accepted["diagnostics"]["perpendicular_truth_rms"] == 0.05
    assert accepted["diagnostics"]["perpendicular_to_full_rms_ratio"] == 0.2
    assert accepted["diagnostic_requirements"]["excluded_fraction_max"] == 0.80
    assert evaluate_synthetic_recovery_gate(passing.chunk({"stratum": 1}))["passed"] is True

    failing = passing.copy(deep=True)
    failing["field_nrmse"][:] = 0.36
    rejected = evaluate_synthetic_recovery_gate(failing)
    assert rejected["passed"] is False
    assert "field_nrmse exceeds 0.35" in rejected["failures"]

    excessive_exclusion = evaluate_synthetic_recovery_gate(
        _passing_recovery_report(excluded_fraction=0.81)
    )
    assert excessive_exclusion["passed"] is False
    assert "excluded_fraction exceeds the frozen 0.80 ceiling" in excessive_exclusion["failures"]

    nonfinite = _passing_recovery_report()
    nonfinite["holdout_aod_rmse_ratio"].loc[{"stratum": "primary"}] = np.nan
    diagnostics = evaluate_synthetic_recovery_gate(nonfinite)
    assert diagnostics["passed"] is False
    assert "primary recovery diagnostics must all be finite" in diagnostics["failures"]

    nonfinite_decomposition = _passing_recovery_report()
    nonfinite_decomposition["perpendicular_truth_rms"].loc[{"stratum": "primary"}] = np.nan
    diagnostics = evaluate_synthetic_recovery_gate(nonfinite_decomposition)
    assert diagnostics["passed"] is False
    assert "primary recovery diagnostics must all be finite" in diagnostics["failures"]


def test_acceptance_resource_limits_are_hard_per_seed_gates() -> None:
    assert acceptance_impl._resource_gate(1799.0, 8 * 1024**3 - 1)["passed"] is True
    assert acceptance_impl._resource_gate(1800.0, 1)["passed"] is False
    assert acceptance_impl._resource_gate(1.0, 8 * 1024**3)["passed"] is False


def test_failed_recovery_gates_still_report_aggregate_statistics() -> None:
    seeds = [101, 202, 303]
    nrmse_values = [0.48, 0.50, 0.52]
    runs = []
    for seed, nrmse in zip(seeds, nrmse_values, strict=True):
        report = _passing_recovery_report()
        report["field_nrmse"][:] = nrmse
        runs.append(
            {
                "seed": seed,
                "status": "failed",
                "evaluation": {"recovery_gate": evaluate_synthetic_recovery_gate(report)},
            }
        )

    aggregate = acceptance_impl._aggregate_recovery(runs)

    assert aggregate["passed"] is False
    assert aggregate["metrics"]["field_nrmse"]["mean"] == pytest.approx(0.50)
    assert aggregate["metrics"]["field_nrmse"]["per_seed"] == [
        {"seed": seed, "value": pytest.approx(value)}
        for seed, value in zip(seeds, nrmse_values, strict=True)
    ]
    assert len(aggregate["metrics"]["field_nrmse"]["confidence_interval_95"]) == 2
    assert aggregate["failures"].count("equal-seed aggregate: field_nrmse exceeds 0.35") == 1
    assert (
        sum("did not pass its recovery gate" in failure for failure in aggregate["failures"]) == 3
    )


def test_full_acceptance_records_equal_seed_report_and_complete_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds = [101, 202, 303]

    def fake_generate(spec: SyntheticTuningSpec) -> object:
        return spec

    def fake_write(root: Path, bundle: object) -> Path:
        del bundle
        path = root / "scenario.json"
        path.write_text('{"synthetic":true}\n', encoding="utf-8")
        return path

    def identity(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def execute(config: Path) -> dict[str, Any]:
        manifest = config.with_name(f"{config.stem}-manifest.json")
        if "eval" not in config.stem:
            manifest.write_text('{"status":"completed"}\n', encoding="utf-8")
            return {"success": True, "manifest": identity(manifest)}
        seed = int(config.parent.name.removeprefix("seed-"))
        report = _passing_recovery_report(field_correlation=0.90 + seed / 10000.0)
        artifact_dir = config.parent / "recovery-artifact"
        artifact_dir.mkdir()
        artifact = artifact_dir / "chunk-00000.nc"
        artifact.write_bytes(b"synthetic recovery artifact")
        summary = artifact_dir / "summary.json"
        summary.write_text("{}\n", encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        summary_sha = hashlib.sha256(summary.read_bytes()).hexdigest()
        collection = hashlib.sha256()
        collection.update(artifact.name.encode("utf-8"))
        collection.update(artifact_sha.encode("ascii"))
        artifact_entry = {
            "analysis": "recovery",
            "role": "recovery_report",
            "kind": "netcdf_collection",
            "status": "finalized",
            "artifact_dir": str(artifact_dir),
            "summary_path": str(summary),
            "identity": {
                "source_hashes": {"truth": "a" * 64},
                "config_hashes": {"recovery": "b" * 64},
                "code_hashes": {"known_truth": "c" * 64},
            },
            "checksums": {
                "files": {artifact.name: artifact_sha},
                "summary_sha256": summary_sha,
                "collection_sha256": collection.hexdigest(),
            },
        }
        manifest.write_text(
            json.dumps({"status": "completed", "analysis_artifacts": [artifact_entry]}) + "\n",
            encoding="utf-8",
        )
        result: dict[str, Any] = {"success": True, "manifest": identity(manifest)}
        result.update(
            recovery_gate=evaluate_synthetic_recovery_gate(report),
            recovery_report=report.to_dict(data="list"),
            recovery_artifact=artifact_entry,
        )
        return result

    monkeypatch.setattr(acceptance_impl, "generate_aerosol_tuning_bundle", fake_generate)
    monkeypatch.setattr(acceptance_impl, "write_aerosol_tuning_bundle", fake_write)
    record_path = run_synthetic_acceptance(tmp_path, seeds, pipeline_executor=execute)
    record_text = record_path.read_text(encoding="utf-8")
    assert "NaN" not in record_text
    record = json.loads(record_text)

    assert record["status"] == "completed"
    assert record["aggregate"]["passed"] is True
    assert record["aggregate"]["weighting"] == "equal_seed"
    assert record["aggregate"]["seed_weights"] == {
        "101": pytest.approx(1.0 / 3.0),
        "202": pytest.approx(1.0 / 3.0),
        "303": pytest.approx(1.0 / 3.0),
    }
    correlations = [0.9101, 0.9202, 0.9303]
    assert record["aggregate"]["metrics"]["field_correlation"]["mean"] == pytest.approx(
        np.mean(correlations)
    )
    assert len(record["aggregate"]["metrics"]["field_correlation"]["confidence_interval_95"]) == 2
    for run in record["runs"]:
        assert run["status"] == "completed"
        assert run["evidence_gate"]["passed"] is True
        assert run["resource_gate"]["passed"] is True
        assert len(run["scenario_manifest"]["sha256"]) == 64
        assert run["fitting"]["manifest"]["sha256"]
        assert run["evaluation"]["manifest"]["sha256"]
        assert run["evaluation"]["recovery_artifact"]["identity"]
        assert run["evaluation"]["recovery_report"]["data_vars"]
    assert list(tmp_path.glob(".*.tmp")) == []

    first = record["runs"][0]
    mismatched_evaluation = dict(first["evaluation"])
    mismatched_artifact = json.loads(json.dumps(first["evaluation"]["recovery_artifact"]))
    mismatched_artifact["identity"]["source_hashes"]["truth"] = "d" * 64
    mismatched_evaluation["recovery_artifact"] = mismatched_artifact
    mismatched = acceptance_impl._evidence_gate(first["fitting"], mismatched_evaluation)
    assert mismatched["passed"] is False
    assert "recovery-artifact identity" in mismatched["failures"][0]

    artifact_path = Path(first["evaluation"]["recovery_artifact"]["artifact_dir"])
    artifact_path = artifact_path / "chunk-00000.nc"
    artifact_path.write_bytes(b"tampered recovery artifact")
    tampered = acceptance_impl._evidence_gate(first["fitting"], first["evaluation"])
    assert tampered["passed"] is False
    assert "recovery-artifact identity" in tampered["failures"][0]

    fitting_manifest = Path(first["fitting"]["manifest"]["path"])
    fitting_manifest.write_text('{"status":"failed"}\n', encoding="utf-8")
    first["fitting"]["manifest"] = identity(fitting_manifest)
    incomplete = acceptance_impl._evidence_gate(first["fitting"], first["evaluation"])
    assert incomplete["passed"] is False
    assert "completed manifest" in incomplete["failures"][0]

    changed = dict(record)
    changed["seeds"] = [101, 202, 404]
    plan = build_synthetic_acceptance_plan(tmp_path, seeds)
    with pytest.raises(ValueError, match="immutable seed lock"):
        acceptance_impl._write_record(plan, changed)


def test_osse_clis_require_supplied_seeds_and_support_dry_validation(tmp_path: Path) -> None:
    """OSSE CLI validation exposes the profile without materializing the large case."""
    script = (
        Path(__file__).parents[4]
        / "analyses"
        / "aerosol-tuning"
        / "scripts"
        / "generate_synthetic.py"
    )
    missing = subprocess.run(
        [sys.executable, str(script), "/unused", "--scenario", "synthetic_osse", "--validate-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "--seed is required" in missing.stderr

    validated = subprocess.run(
        [
            sys.executable,
            str(script),
            "/unused",
            "--scenario",
            "synthetic_osse",
            "--seed",
            "404",
            "--validate-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validated.returncode == 0, validated.stderr
    contract = json.loads(validated.stdout)
    assert contract["master_seed"] == 404
    assert contract["native_domain"]["n_lon"] == 72

    acceptance_script = script.with_name("run_acceptance.py")
    acceptance_root = tmp_path / "acceptance"
    acceptance = subprocess.run(
        [
            sys.executable,
            str(acceptance_script),
            str(acceptance_root),
            "--seed",
            "11",
            "--seed",
            "22",
            "--seed",
            "33",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert acceptance.returncode == 0, acceptance.stderr
    record = json.loads((acceptance_root / "acceptance.json").read_text(encoding="utf-8"))
    assert record["seeds"] == [11, 22, 33]


def test_named_streams_are_stable_and_invocation_order_independent() -> None:
    """A stream is derived from its name rather than the order streams are requested."""
    forward = {name: named_rng(91, name).normal(size=4) for name in NAMED_STREAMS}
    reverse = {name: named_rng(91, name).normal(size=4) for name in reversed(NAMED_STREAMS)}
    for name in NAMED_STREAMS:
        np.testing.assert_array_equal(forward[name], reverse[name])
    assert not np.array_equal(forward["common_error"], forward["sensor_a_noise"])
    assert not np.array_equal(forward["common_error"], named_rng(92, "common_error").normal(size=4))
    assert named_stream_id(91, "common_error") != named_stream_id(92, "common_error")
    with pytest.raises(KeyError, match="unknown synthetic random stream"):
        named_rng(91, "typo")


def test_local_overpass_oracle_handles_date_rollover() -> None:
    """Western dateline columns can select the next UTC calendar day."""
    time = pd.date_range("2001-01-01", "2001-01-03 23:00", freq="1h")
    lon = np.array([-180.0, 0.0, 150.0])
    values = np.broadcast_to(np.arange(time.size)[:, None, None], (time.size, 2, lon.size))
    hourly = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-30.0, 30.0], "lon": lon},
    )

    selected = local_overpass_oracle(hourly, pd.DatetimeIndex(["2001-01-02 12:00"]))

    # Exact half-hour ties select the earlier sample: 25.5, 13.5, and 3.5 become 25, 13, and 3.
    np.testing.assert_array_equal(selected.values[0, 0], [49, 37, 27])
    assert selected.time.values[0] == np.datetime64("2001-01-02T12:00:00")


def test_periodic_bilinear_oracle_crosses_longitude_seam() -> None:
    """Equivalent longitudes use the same wrapped interpolation bracket."""
    source_lat = np.array([-60.0, 60.0])
    source_lon = np.array([-180.0, -90.0, 0.0, 90.0])
    lon_values = np.array([-1.0, 0.0, 1.0, 0.0])
    values = np.stack((lon_values - 1.0, lon_values + 1.0))

    result = periodic_bilinear_oracle(
        values,
        source_lat,
        source_lon,
        target_lat=[0.0],
        target_lon=[135.0, -225.0],
    )

    np.testing.assert_allclose(result, [[-0.5, -0.5]], rtol=0.0, atol=1.0e-14)

    offset_grid = periodic_bilinear_oracle(
        np.broadcast_to(np.array([0.0, 1.0, 2.0, 3.0]), (2, 4)),
        source_lat,
        source_lon=[15.0, 105.0, 195.0, 285.0],
        target_lat=[0.0],
        target_lon=[7.5],
    )
    np.testing.assert_allclose(offset_grid, [[0.25]], rtol=0.0, atol=1.0e-14)


def test_area_weighted_regrid_oracle_handles_non_commensurate_cells() -> None:
    """Spherical overlap, rather than center ownership, handles 5x5 to 3x3 coarsening."""
    source_lat = np.array([-72.0, -36.0, 0.0, 36.0, 72.0])
    source_lon = np.array([-144.0, -72.0, 0.0, 72.0, 144.0])
    values = source_lat[:, None] ** 2 + np.zeros((source_lat.size, source_lon.size))

    result = area_weighted_regrid_oracle(
        values,
        source_lat,
        source_lon,
        target_lat=[-60.0, 0.0, 60.0],
        target_lon=[-120.0, 0.0, 120.0],
    )

    sin_30 = np.sin(np.deg2rad(30.0))
    sin_54 = np.sin(np.deg2rad(54.0))
    expected_north = (
        source_lat[3] ** 2 * (sin_54 - sin_30) + source_lat[4] ** 2 * (1.0 - sin_54)
    ) / (1.0 - sin_30)
    np.testing.assert_allclose(result[2], np.full(3, expected_north), rtol=0.0, atol=1.0e-12)

    longitude_field = np.broadcast_to([0.0, 6.0, 0.0], (2, 3))
    longitude_result = area_weighted_regrid_oracle(
        longitude_field,
        source_lat=[-45.0, 45.0],
        source_lon=[-120.0, 0.0, 120.0],
        target_lat=[-45.0, 45.0],
        target_lon=[-90.0, 90.0],
    )
    np.testing.assert_allclose(longitude_result, 2.0, rtol=0.0, atol=1.0e-14)


def test_shifted_log_oracle_uses_exact_ratio_bounds_and_identity_gates() -> None:
    """The epsilon shift makes the physical ratio differ from exp(delta)."""
    model = np.array([0.02, 0.02, 0.0, 0.2])
    delta = np.array([0.3, 1000.0, -0.2, -1000.0])
    support = np.array([1.0, 0.0, 1.0, 1.0])

    result = shifted_log_ratio_oracle(
        model,
        delta,
        epsilon=0.01,
        r_bounds=(0.4, 2.5),
        aod_floor=0.001,
        support=support,
    )

    expected = ((model[0] + 0.01) * np.exp(delta[0]) - 0.01) / model[0]
    assert result.applied_ratio[0] == pytest.approx(expected)
    assert result.applied_ratio[0] != pytest.approx(np.exp(delta[0]))
    assert result.applied_ratio[1] == 1.0
    assert result.applied_ratio[2] == 1.0
    assert result.applied_ratio[3] == pytest.approx(0.4)
    assert np.all(np.isfinite(result.applied_ratio))
    np.testing.assert_allclose(
        result.applied_delta,
        np.log((result.applied_aod + 0.01) / (model + 0.01)),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_exact_bundle_is_deterministic_and_inputs_hide_truth(exact_bundle) -> None:
    """Repeated generation is identical and fitting inputs expose no oracle fields."""
    repeated = generate_aerosol_tuning_bundle(SyntheticTuningSpec.exact_micro())
    assert scientific_dataset_hash(exact_bundle.model) == scientific_dataset_hash(repeated.model)
    assert scientific_dataset_hash(exact_bundle.truth) == scientific_dataset_hash(repeated.truth)
    for sensor in ("sensor_a", "sensor_b"):
        assert scientific_dataset_hash(
            exact_bundle.observations[sensor]
        ) == scientific_dataset_hash(repeated.observations[sensor])

    fitting_inputs = [
        exact_bundle.model,
        *exact_bundle.observations.values(),
        *exact_bundle.mmr.values(),
    ]
    for dataset in fitting_inputs:
        assert all(not name.endswith("_true") for name in dataset.data_vars)
        assert "oracle" not in json.dumps(dict(dataset.attrs)).lower()
        assert str(dataset.attrs["role"]).startswith("fitting_input")
    assert exact_bundle.truth.attrs["role"] == "evaluation_only:oracle"


def test_exact_bundle_couples_patterns_observations_and_shifted_log(exact_bundle) -> None:
    """The noiseless case closes independently from model through observation and ratio."""
    truth = exact_bundle.truth
    patterns = truth["pattern_true"].values
    weights = np.cos(np.deg2rad(truth["mode_lat"].values))[:, None]
    weights = np.broadcast_to(weights, patterns.shape[1:])
    gram = np.einsum("kij,lij,ij->kl", patterns, patterns, weights) / weights.sum()
    np.testing.assert_allclose(gram, np.eye(patterns.shape[0]), rtol=0.0, atol=2.0e-14)

    assert float(exact_bundle.model["TOTEXTTAU"].min()) > 0.0
    assert np.all(truth["valid_mask"].values == 1)
    assert np.all(truth["obs_error_log"].values == 0.0)
    assert np.all(truth["delta_perp_true"].values == 0.0)
    assert np.all(truth["clip_mask_true"].values == 0)
    model = truth["model_aod_overpass_true"].values
    delta = truth["delta_requested_true"].values
    observed_log = exact_bundle.observations["sensor_a"]["log_aod"].values
    np.testing.assert_allclose(
        observed_log - np.log(model + exact_bundle.spec.log_epsilon),
        delta,
        rtol=0.0,
        atol=3.0e-15,
    )
    ratio_from_aod = truth["aod_target_applied_true"].values / model
    np.testing.assert_allclose(ratio_from_aod, truth["r_applied_true"], rtol=1.0e-13)
    np.testing.assert_allclose(truth["delta_applied_true"], delta, rtol=0.0, atol=3.0e-15)
    assert tuple(np.unique(truth["split"].values)) == tuple(
        sorted(
            (
                "basis_train",
                "bias_fit",
                "calibration",
                "development_test",
            )
        )
    )


def test_model_overpass_truth_is_recomputed_from_hourly_input(exact_bundle) -> None:
    """Truth stores selection-then-linear-regrid output, not a parallel daily formula."""
    native = local_overpass_oracle(
        exact_bundle.model["TOTEXTTAU"],
        exact_bundle.truth["time"].values,
        exact_bundle.spec.local_overpass_hour,
    )
    expected = area_weighted_regrid_oracle(
        native.values,
        native["lat"].values,
        native["lon"].values,
        exact_bundle.truth["mode_lat"].values,
        exact_bundle.truth["mode_lon"].values,
    )
    np.testing.assert_array_equal(expected, exact_bundle.truth["model_aod_overpass_true"].values)


def test_masked_case_composes_masks_and_preserves_unsupported_identity(masked_bundle) -> None:
    """Every validity component is explicit, and unsupported cells remain exact identity."""
    truth = masked_bundle.truth
    component_names = (
        "footprint",
        "seasonal_visibility",
        "cloud",
        "day_available",
        "qa_pass",
    )
    expected_valid = np.logical_and.reduce(
        tuple(truth[name].values.astype(bool) for name in component_names)
    )
    np.testing.assert_array_equal(expected_valid, truth["valid_mask"].values.astype(bool))
    assert np.any(expected_valid)
    assert np.any(~expected_valid)
    assert np.all(truth["mask_reason"].values[expected_valid] == 0)
    assert np.all(truth["mask_reason"].values[~expected_valid] != 0)

    fitting_days = np.flatnonzero(truth["split"].values != "development_test")
    all_invalid_day = int(fitting_days[fitting_days.size // 2])
    assert not expected_valid[:, all_invalid_day].any()
    assert np.all(truth["day_available"].values[:, all_invalid_day] == 1)
    assert np.all(truth["day_available"].values[:, 1] == 0)
    for sensor_index, sensor in enumerate(("sensor_a", "sensor_b")):
        invalid = ~expected_valid[sensor_index]
        raw = masked_bundle.observations[sensor]["aod_550nm"].values
        qa = masked_bundle.observations[sensor]["qa_flag"].values
        assert np.all(np.isfinite(raw[invalid]))
        assert np.all(raw[invalid] >= 9.0)
        assert np.all(qa[invalid] == 0)

    month_index = truth["time"].dt.month.values - 1
    daily_support = truth["spatial_support_true"].values[month_index]
    unsupported = daily_support == 0.0
    assert np.any(unsupported)
    assert np.all(truth["r_applied_true"].values[unsupported] == 1.0)
    assert np.all(truth["delta_applied_true"].values[unsupported] == 0.0)


def test_masked_off_basis_term_is_weighted_orthogonal(masked_bundle) -> None:
    """The irreducible correction is generated outside the retained model subspace."""
    truth = masked_bundle.truth
    patterns = truth["pattern_true"].values
    perpendicular = truth["delta_perp_true"].values
    weights = np.cos(np.deg2rad(truth["mode_lat"].values))[:, None]
    projections = np.einsum("kij,tij,ij->tk", patterns, perpendicular, weights)
    assert np.max(np.abs(projections)) < 2.0e-15
    assert np.any(perpendicular != 0.0)


def test_masked_filter_target_is_an_independent_segment_policy_oracle(masked_bundle) -> None:
    """The primary target applies analytic segment/taper policy rather than aliasing a ceiling."""
    truth = masked_bundle.truth
    filtered = truth["delta_filter_target_true"].values
    representable = truth["delta_best_representable_true"].values

    assert np.any(filtered != representable)
    assert np.any(truth["filter_bridged_true"].values != 0)
    assert np.any(truth["filter_valid_segment_true"].values == 0)
    assert np.any(truth["filter_edge_weight_true"].values < 1.0)
    model = truth["model_aod_overpass_true"].values
    target = truth["aod_filter_target_true"].values
    expected_delta = np.log(
        (target + masked_bundle.spec.log_epsilon) / (model + masked_bundle.spec.log_epsilon)
    )
    np.testing.assert_allclose(filtered, expected_delta, rtol=0.0, atol=2.0e-15)


def test_partial_support_tapers_delta_before_shifted_log_scaling() -> None:
    """Fractional support attenuates both bias and anomaly before physical ratio bounds."""
    spec = SyntheticTuningSpec(
        scenario="masked_chain_ci",
        support_min_fraction=0.0,
        support_full_fraction=1.0,
    )
    truth = generate_aerosol_tuning_bundle(spec).truth
    month_index = truth["time"].dt.month.values - 1
    support = truth["spatial_support_true"].values[month_index]
    partial = (support > 0.0) & (support < 1.0)
    assert np.any(partial)
    np.testing.assert_allclose(
        truth["delta_supported_true"],
        truth["delta_requested_true"].values * support,
        rtol=0.0,
        atol=0.0,
    )
    expected = shifted_log_ratio_oracle(
        truth["model_aod_overpass_true"].values,
        truth["delta_supported_true"].values,
        epsilon=spec.log_epsilon,
        r_bounds=spec.r_bounds,
        aod_floor=spec.aod_floor,
        support=support,
    )
    np.testing.assert_allclose(
        truth["delta_applied_true"], expected.applied_delta, rtol=0.0, atol=0.0
    )


def test_writer_case_has_full_mmr_schema_and_independent_optical_closure() -> None:
    """All aerosol fields scale one optical column while gas and static inputs remain separate."""
    bundle = generate_aerosol_tuning_bundle(SyntheticTuningSpec.writer_ci())
    assert len(bundle.mmr) == 2
    for dataset in bundle.mmr.values():
        assert dataset.sizes["time"] == 8
        assert np.all(np.diff(dataset["lev"].values) > 0.0)
        assert dataset["lev"].values[-1] == dataset["lev"].values.max()
        assert set(DEFAULT_AEROSOL_SPECIES) <= set(dataset.data_vars)
        assert {"SO2", "DMS", "MSA", "RH", "DELP", "T", "ORO"} <= set(dataset.data_vars)
        assert dataset["DU001"].dtype == np.float32

    combined = xr.concat(list(bundle.mmr.values()), dim="time", data_vars="all")
    baseline = optical_aod_oracle(combined, DEFAULT_AEROSOL_SPECIES)
    np.testing.assert_allclose(
        baseline.values,
        bundle.truth["baseline_optical_aod"].values,
        rtol=0.0,
        atol=0.0,
    )
    mixing_ratio = np.stack([combined[name].values for name in DEFAULT_AEROSOL_SPECIES])
    serialized_sum = np.sum(
        bundle.truth["kappa"].values
        * mixing_ratio
        * bundle.truth["layer_weight"].values[None, ...],
        axis=(0, 2),
    )
    np.testing.assert_allclose(serialized_sum, baseline.values, rtol=0.0, atol=1.0e-15)
    scaled = combined.copy(deep=True)
    ratio = bundle.truth["r_3hour_true"].rename(
        {"mmr_time": "time", "native_lat": "lat", "native_lon": "lon"}
    )
    for species in DEFAULT_AEROSOL_SPECIES:
        scaled[species] = scaled[species] * ratio
    scaled_aod = optical_aod_oracle(scaled, DEFAULT_AEROSOL_SPECIES)
    np.testing.assert_allclose(
        scaled_aod.values,
        bundle.truth["scaled_optical_aod"].values,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        scaled_aod.values,
        baseline.values * bundle.truth["r_3hour_true"].values,
        rtol=5.0e-7,
        atol=2.0e-8,
    )
    xr.testing.assert_identical(scaled["SO2"], combined["SO2"])
    xr.testing.assert_identical(scaled["T"], combined["T"])


def test_serialization_records_reproducible_science_and_byte_hashes(tmp_path: Path) -> None:
    """The adapter separates fitting inputs from oracle data and hashes closed files."""
    bundle = generate_aerosol_tuning_bundle(tmp_path, SyntheticTuningSpec.writer_ci())
    manifest = json.loads((tmp_path / "scenario.json").read_text(encoding="utf-8"))

    assert manifest["spec_hash"] == bundle.truth.attrs["spec_hash"]
    assert set(manifest["stream_map"]) == set(NAMED_STREAMS)
    assert "wall_clock" not in json.dumps(manifest)
    assert len(manifest["files"]) == 6
    for relative, hashes in manifest["files"].items():
        path = tmp_path / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes["byte_sha256"]
        with xr.open_dataset(path) as opened:
            loaded = opened.load()
        assert scientific_dataset_hash(loaded) == hashes["scientific_sha256"]
        if relative.startswith("inputs/"):
            assert "oracle" not in relative
            assert hashes["role"].startswith("fitting_input")
        else:
            assert hashes["role"] == "evaluation_only:oracle"

    mmr_paths = sorted((tmp_path / "inputs/mmr").glob("*.nc4"))
    model_path = tmp_path / "inputs/model/MERRA2_SYNTH.tavg1_2d_aer_Nx.nc4"
    with xr.open_dataset(model_path) as reopened:
        model_encoding = reopened["TOTEXTTAU"].encoding
        assert model_encoding["zlib"] is True
        assert model_encoding["chunksizes"] == aerosol_io_impl._bounded_netcdf_chunks(
            reopened["TOTEXTTAU"].dims,
            reopened["TOTEXTTAU"].shape,
            reopened["TOTEXTTAU"].dtype.itemsize,
        )
    with xr.open_dataset(mmr_paths[0]) as reopened:
        assert reopened["DU001"].encoding["zlib"] is True
        assert reopened["DU001"].encoding["_FillValue"] == np.float32(-9.999e15)
        assert np.isnan(reopened["DU001"].values).any()
        assert np.isnan(reopened["SO2"].values).any()
    with pytest.raises(FileExistsError, match="already exists"):
        write_aerosol_tuning_bundle(tmp_path, bundle)


def test_generator_does_not_import_production_fable_helpers() -> None:
    """Generator identities remain independent of the implementation under test."""
    package = Path(tuning_module.__file__).parent
    imported_modules: set[str] = set()
    for path in package.glob("*aerosol*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
    forbidden_prefixes = (
        "davinci_monet.analysis",
        "davinci_monet.pipeline",
        "davinci_monet.pairing",
        "davinci_monet.util.logspace",
    )
    assert not any(
        module.startswith(prefix) for module in imported_modules for prefix in forbidden_prefixes
    )
