"""Compact real-pipeline integration coverage for the FABLE v2 chain."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
import xarray as xr
import yaml

from davinci_monet.pipeline.lifecycle import PipelineResourcePolicy
from davinci_monet.pipeline.runner import PipelineResult, PipelineRunner
from davinci_monet.tests.synthetic._aerosol_contracts import (
    SyntheticTuningBundle,
    SyntheticTuningSpec,
)
from davinci_monet.tests.synthetic._aerosol_io import generate_aerosol_tuning_bundle
from davinci_monet.tests.synthetic._aerosol_policy import ScientificPolicy
from davinci_monet.tests.synthetic.fable_calibration_runner import evaluate_null_control
from davinci_monet.tests.synthetic.fable_v2_policy import v2_calibration_policies
from davinci_monet.tests.synthetic.fable_v2_protocol import V1_EXPOSED_SEEDS, seed_roles
from davinci_monet.tests.synthetic.fable_v2_runner import render_v2_scenario_configs
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig

_LOCAL_NULL_FIXTURE_SEED = 424244


@dataclass(frozen=True)
class _CompactV2Run:
    root: Path
    spec: SyntheticTuningSpec
    fitting_path: Path
    evaluation_path: Path | None
    fitting_config: dict[str, Any]
    manifest: dict[str, Any]
    projection_attrs: dict[str, Any]
    projection_variables: frozenset[str]
    scaling: xr.Dataset
    filtered: xr.Dataset
    recovery: xr.Dataset | None = None
    diagnostics: xr.Dataset | None = None
    evaluation_artifacts: tuple[dict[str, Any], ...] = ()


def _compact_v2_spec(*, null: bool = False) -> SyntheticTuningSpec:
    roles = seed_roles()
    protocol_seeds = {seed for values in roles.values() for seed in values}
    assert _LOCAL_NULL_FIXTURE_SEED not in protocol_seeds
    assert _LOCAL_NULL_FIXTURE_SEED not in V1_EXPOSED_SEEDS
    base = (
        SyntheticTuningSpec.synthetic_osse_null(_LOCAL_NULL_FIXTURE_SEED)
        if null
        else SyntheticTuningSpec.synthetic_osse(roles["development"][0])
    )
    return replace(
        base,
        native_domain=Domain(-180.0, 180.0, -90.0, 90.0, 24 if null else 12, 12 if null else 6),
        mode_domain=Domain(-180.0, 180.0, -90.0, 90.0, 12 if null else 6, 6 if null else 3),
        time_config=TimeConfig("2001-01-01", "2002-12-31", "1h"),
        split_windows=(
            ("basis_train", "2001-01-01", "2001-06-30"),
            ("bias_fit", "2001-07-01", "2002-06-30"),
            ("calibration", "2002-07-01", "2002-09-30"),
            ("development_test", "2002-10-01", "2002-12-31"),
        ),
        correction_periods_days=(6.0, 9.0, 12.0),
        filter_band_days=(4.0, 16.0),
        filter_min_segment_days=32,
    )


def _close_bundle(bundle: SyntheticTuningBundle) -> None:
    datasets = [bundle.model, *bundle.observations.values(), *bundle.mmr.values(), bundle.truth]
    for dataset in datasets:
        dataset.close()


def _close_result(result: PipelineResult | None) -> None:
    if result is not None and result.context is not None:
        PipelineResourcePolicy(close_datasets_after_run=False).cleanup_context_datasets(
            result.context
        )


def _apply_compact_filter(path: Path, spec: SyntheticTuningSpec) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    wavelet = config["analyses"]["filtered_pcs"]
    wavelet["band"] = {
        "min": spec.filter_band_days[0],
        "max": spec.filter_band_days[1],
        "units": "days",
    }
    wavelet["max_bridge_days"] = spec.filter_max_bridge_days
    wavelet["min_segment_days"] = spec.filter_min_segment_days
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _fit_artifact(run: _CompactV2Run, role: str) -> tuple[Path, dict[str, Any]]:
    entries = [entry for entry in run.manifest["analysis_artifacts"] if entry.get("role") == role]
    assert len(entries) == 1
    entry = entries[0]
    filenames = sorted(entry["checksums"]["files"])
    assert filenames
    return Path(entry["artifact_dir"]) / filenames[0], entry


def _saved_fit_config(run: _CompactV2Run, output_name: str) -> dict[str, Any]:
    config = deepcopy(run.fitting_config)
    config["analysis"]["output_dir"] = run.root / output_name
    config["analysis"]["log_dir"] = run.root / f"{output_name}-logs"
    basis_path, _basis_entry = _fit_artifact(run, "basis_fit")
    _projection_path, projection_entry = _fit_artifact(run, "projection_fit")
    manifest_path = run.root / "output" / "manifest.json"
    projection_summary = json.loads(
        Path(projection_entry["summary_path"]).read_text(encoding="utf-8")
    )
    projection_variables = tuple(projection_summary["chunks"])
    config["sources"]["frozen_aod_basis"] = {
        "type": "generic",
        "files": str(basis_path),
        "artifact_manifest": str(manifest_path),
        "artifact_role": "basis_fit",
        "artifact_analysis": "aod_basis",
        "variables": {"eofs": {}},
    }
    config["sources"]["frozen_projection_fit"] = {
        "type": "generic",
        "files": str(Path(projection_entry["artifact_dir"]) / "chunk-*.nc"),
        "artifact_manifest": str(manifest_path),
        "artifact_role": "projection_fit",
        "artifact_analysis": "obs_pcs",
        "combine": "nested",
        "concat_dim": "time",
        "data_vars": "minimal",
        "coords": "minimal",
        "compat": "override",
        "join": "exact",
        "variables": {name: {} for name in projection_variables},
    }
    analyses = {
        name: value
        for name, value in config["analyses"].items()
        if name not in {"aod_basis", "corrected"}
    }
    projection = analyses["obs_pcs"]
    projection["basis"] = "frozen_aod_basis"
    projection.pop("bias_fit_window")
    projection["bias_fit_artifact"] = "frozen_projection_fit"
    analyses["scaling"]["basis"] = "frozen_aod_basis"
    config["analyses"] = analyses
    return config


@contextmanager
def _synthetic_root(root: Path) -> Iterator[None]:
    previous = os.environ.get("FABLE_SYNTH")
    os.environ["FABLE_SYNTH"] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FABLE_SYNTH", None)
        else:
            os.environ["FABLE_SYNTH"] = previous


@pytest.fixture(scope="module")
def compact_v2_fit_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[_CompactV2Run]:
    """Generate and fit the compact recovery chain once for all v2 assertions."""
    root = tmp_path_factory.mktemp("fable-v2-real-pipeline")
    spec = _compact_v2_spec()
    bundle = generate_aerosol_tuning_bundle(root, spec)
    _close_bundle(bundle)
    policy = v2_calibration_policies()[0]
    fitting_path, evaluation_path = render_v2_scenario_configs(
        root,
        spec,
        policy,
        evaluation_splits=("development_test",),
    )
    assert evaluation_path is not None
    _apply_compact_filter(fitting_path, spec)
    with _synthetic_root(root):
        fitting = PipelineRunner(
            show_progress=False,
            close_datasets_after_run=False,
        ).run_from_config(str(fitting_path))
    assert fitting.success, fitting.stage_errors
    manifest = json.loads((root / "output" / "manifest.json").read_text(encoding="utf-8"))
    assert fitting.context is not None
    projection = fitting.context.sources["obs_pcs"].data
    value = _CompactV2Run(
        root=root,
        spec=spec,
        fitting_path=fitting_path,
        evaluation_path=evaluation_path,
        fitting_config=deepcopy(fitting.context.config_dict()),
        manifest=manifest,
        projection_attrs=dict(projection.attrs),
        projection_variables=frozenset(projection.data_vars),
        scaling=fitting.context.sources["scaling"].data.compute(),
        filtered=fitting.context.sources["filtered_pcs"].data.compute(),
    )
    _close_result(fitting)
    yield value


@pytest.fixture(scope="module")
def compact_v2_run(compact_v2_fit_run: _CompactV2Run) -> Iterator[_CompactV2Run]:
    """Evaluate the compact fit once through the separate known-truth pipeline."""
    run = compact_v2_fit_run
    assert run.evaluation_path is not None
    with _synthetic_root(run.root):
        evaluation = PipelineRunner(
            show_progress=False,
            close_datasets_after_run=False,
        ).run_from_config(str(run.evaluation_path))
    assert evaluation.success, evaluation.stage_errors
    assert evaluation.context is not None
    value = replace(
        run,
        recovery=evaluation.context.sources["recovery"].data.compute(),
        diagnostics=evaluation.context.sources["v2_diagnostics"].data.compute(),
        evaluation_artifacts=tuple(deepcopy(evaluation.context.metadata["analysis_artifacts"])),
    )
    _close_result(evaluation)
    yield value


@pytest.fixture(scope="module")
def compact_v2_null_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_CompactV2Run]:
    """Run one compact full-policy null chain through a real PipelineRunner."""
    root = tmp_path_factory.mktemp("fable-v2-null-pipeline")
    spec = _compact_v2_spec(null=True)
    bundle = generate_aerosol_tuning_bundle(root, spec)
    _close_bundle(bundle)
    policy = v2_calibration_policies()[0]
    fitting_path, evaluation_path = render_v2_scenario_configs(
        root,
        spec,
        policy,
        evaluation_splits=(),
        include_evaluation=False,
    )
    assert evaluation_path is None
    _apply_compact_filter(fitting_path, spec)
    with _synthetic_root(root):
        fitting = PipelineRunner(
            show_progress=False,
            close_datasets_after_run=False,
        ).run_from_config(str(fitting_path))
    assert fitting.success, fitting.stage_errors
    manifest = json.loads((root / "output" / "manifest.json").read_text(encoding="utf-8"))
    assert fitting.context is not None
    projection = fitting.context.sources["obs_pcs"].data
    value = _CompactV2Run(
        root=root,
        spec=spec,
        fitting_path=fitting_path,
        evaluation_path=None,
        fitting_config=deepcopy(fitting.context.config_dict()),
        manifest=manifest,
        projection_attrs=dict(projection.attrs),
        projection_variables=frozenset(projection.data_vars),
        scaling=fitting.context.sources["scaling"].data.compute(),
        filtered=fitting.context.sources["filtered_pcs"].data.compute(),
    )
    _close_result(fitting)
    yield value


@pytest.mark.integration
def test_fable_v2_joint_fit_and_diagnostics_use_real_pipelines(
    compact_v2_run: _CompactV2Run,
) -> None:
    """Run the compact v2 fit and evaluation through fresh pipeline runners."""
    run = compact_v2_run
    root = run.root
    spec = run.spec
    fitting_path = run.fitting_path
    fitting_text = fitting_path.read_text(encoding="utf-8")
    assert "/oracle/" not in fitting_text
    assert "_true" not in fitting_text
    fitting_config = yaml.safe_load(fitting_text)
    assert fitting_config["analyses"]["filtered_pcs"]["band"] == {
        "min": 4.0,
        "max": 16.0,
        "units": "days",
    }
    assert fitting_config["analyses"]["filtered_pcs"]["min_segment_days"] == 32

    assert run.projection_attrs["projection_bias_fit_method"] == "joint_seasonal"
    assert run.projection_attrs["projection_joint_bias_converged"] == "true"
    assert {
        "clim_bias_perpendicular",
        "clim_bias_mode_coefficient",
        "sensor_offset",
        "sensor_offset_standard_error",
        "sensor_overlap_count",
        "joint_objective",
    } <= run.projection_variables
    assert {"delta_log_applied", "aod_target", "spatial_support"} <= set(run.scaling.data_vars)
    assert np.isfinite(run.scaling["delta_log_applied"]).all().item()

    projection_entries = [
        entry
        for entry in run.manifest["analysis_artifacts"]
        if entry.get("analysis") == "obs_pcs" and entry.get("role") == "projection_fit"
    ]
    assert len(projection_entries) == 1
    projection_entry = projection_entries[0]
    assert projection_entry["status"] == "finalized"
    artifact_files = sorted(
        Path(projection_entry["artifact_dir"]) / name
        for name in projection_entry["checksums"]["files"]
    )
    assert artifact_files
    with xr.open_dataset(artifact_files[0]) as saved_fit:
        assert saved_fit.attrs["projection_bias_fit_method"] == "joint_seasonal"
        assert saved_fit.attrs["projection_joint_bias_converged"] == "true"
        assert "clim_bias_perpendicular" in saved_fit
        assert "joint_objective" in saved_fit

    assert run.recovery is not None
    assert run.diagnostics is not None
    recovery = run.recovery
    diagnostics = run.diagnostics
    primary = recovery.sel(stratum="primary").compute()
    assert primary["valid_count"].item() > 0
    assert np.isfinite(primary["field_nrmse"]).item()
    assert diagnostics.attrs["diagnostic_only"] == "true"
    assert diagnostics.attrs["eligible_for_calibration"] == "false"
    assert diagnostics.attrs["sensor_offset_gauge"] == "zero_sensor_mean"
    assert diagnostics.sizes["stage"] == 7
    assert np.isfinite(diagnostics["learned_basis_oracle_nrmse"]).item()
    assert {
        "fitted_relative_sensor_offset",
        "true_relative_sensor_offset",
        "relative_sensor_offset_standard_error",
        "sensor_offset_overlap_count",
    } <= set(diagnostics.data_vars)
    assert diagnostics["fitted_relative_sensor_offset"].dims == ("sensor",)
    assert diagnostics["true_relative_sensor_offset"].dims == ("sensor",)
    assert {"recovery_report", "v2_diagnostic_report"} <= {
        entry["role"] for entry in run.evaluation_artifacts
    }

    assert list(root.rglob(".*.tmp")) == []


@pytest.mark.integration
def test_fable_v2_saved_fit_fresh_runner(compact_v2_fit_run: _CompactV2Run) -> None:
    """Load both finalized joint fits in a fresh runner without refitting."""
    run = compact_v2_fit_run
    config = _saved_fit_config(run, "saved-fit-output")
    fresh = PipelineRunner(
        show_progress=False,
        close_datasets_after_run=False,
    ).run_from_config(config)
    try:
        assert fresh.success, fresh.stage_errors
        assert fresh.context is not None
        assert "aod_basis" not in fresh.context.metadata["analysis_status"]
        projection = fresh.context.sources["obs_pcs"].data
        assert projection.attrs["projection_bias_fit_selection"] == "artifact"
        roles = {entry["role"] for entry in fresh.context.metadata["analysis_artifacts"]}
        assert "basis_fit" not in roles
        assert "projection_fit" not in roles
        expected = run.scaling
        actual = fresh.context.sources["scaling"].data
        for name in ("r", "delta_log_applied", "aod_target", "spatial_support"):
            np.testing.assert_array_equal(actual[name], expected[name])
    finally:
        _close_result(fresh)


@pytest.mark.integration
def test_fable_v2_null_policy_pipeline(compact_v2_null_run: _CompactV2Run) -> None:
    """Exercise unchanged null metrics after the complete compact v2 policy chain."""
    run = compact_v2_null_run
    scaling = run.scaling
    filtered = run.filtered
    policy = v2_calibration_policies()[0]
    base_policy = ScientificPolicy(
        policy_id=policy.policy_id,
        covariance_model="diagonal_plus_low_rank_common",
        band_days=run.spec.filter_band_days,
        min_segment_days=run.spec.filter_min_segment_days,
        keep_significant=False,
    )
    with xr.open_dataset(run.root / "oracle" / "truth.nc") as truth:
        metrics = evaluate_null_control(
            scaling,
            filtered,
            truth,
            base_policy,
            split="calibration",
        )
        persisted_metrics = evaluate_null_control(
            scaling,
            scaling,
            truth,
            base_policy,
            split="calibration",
        )
        assert persisted_metrics == metrics
        assert float(abs(truth["delta_filter_target_true"]).max()) == 0.0
    assert set(metrics) == {
        "null_retained_energy_fraction",
        "scored_day_count",
        "null_significant_fraction",
        "significance_candidate_count",
    }
    assert np.isfinite(metrics["null_retained_energy_fraction"])
    assert metrics["null_retained_energy_fraction"] >= 0.0
    assert 0.0 <= metrics["null_significant_fraction"] <= 1.0
    assert metrics["scored_day_count"] > 0.0
    assert metrics["significance_candidate_count"] > 0.0
