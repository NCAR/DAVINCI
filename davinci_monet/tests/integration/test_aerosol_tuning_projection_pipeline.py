"""Synthetic FABLE projection and null-control pipeline integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from davinci_monet.pipeline.runner import PipelineResult, PipelineRunner
from davinci_monet.tests.synthetic.aerosol_tuning import (
    SyntheticTuningBundle,
    SyntheticTuningSpec,
    generate_aerosol_tuning_bundle,
)


def _pipeline_inputs(root: Path, spec: SyntheticTuningSpec) -> dict[str, Any]:
    start = f"{spec.time_config.start} 00:00:00"
    end = f"{spec.time_config.end} 23:59:59"
    return {
        "analysis": {
            "start_time": start,
            "end_time": end,
            "output_dir": str(root / "output"),
        },
        "sources": {
            "model_hourly": {
                "type": "generic",
                "files": str(root / "inputs/model/MERRA2_SYNTH.tavg1_2d_aer_Nx.nc4"),
                "variables": {"TOTEXTTAU": {"units": "1"}},
                "time_padding": "1D",
            },
            "sensor_a_raw": {
                "type": "satellite_l3",
                "files": str(root / "inputs/obs/sensor_a.nc"),
                "qa_variable": "QA",
                "qa_values": [3],
                "variables": {
                    "aod_550nm": {"units": "1"},
                    "reported_sigma_log": {"units": "1"},
                    "QA": {"units": "1"},
                },
            },
            "sensor_b_raw": {
                "type": "satellite_l3",
                "files": str(root / "inputs/obs/sensor_b.nc"),
                "qa_variable": "QA",
                "qa_values": [3],
                "variables": {
                    "aod_550nm": {"units": "1"},
                    "reported_sigma_log": {"units": "1"},
                    "QA": {"units": "1"},
                },
            },
        },
        "analyses": {
            "model_daily": {
                "type": "aod_preprocess",
                "source": "model_hourly",
                "variable": "TOTEXTTAU",
                "sample_local_time": spec.local_overpass_hour,
                "day_anchor_hour": 12.0,
                "target_grid": 60.0,
                "log_epsilon": spec.log_epsilon,
                "required": True,
            },
            "sensor_a_daily": {
                "type": "aod_preprocess",
                "source": "sensor_a_raw",
                "variable": "aod_550nm",
                "uncertainty_variable": "reported_sigma_log",
                "day_anchor_hour": 12.0,
                "target_grid_from": "model_daily",
                "log_epsilon": spec.log_epsilon,
                "required": True,
            },
            "sensor_b_daily": {
                "type": "aod_preprocess",
                "source": "sensor_b_raw",
                "variable": "aod_550nm",
                "uncertainty_variable": "reported_sigma_log",
                "day_anchor_hour": 12.0,
                "target_grid_from": "model_daily",
                "log_epsilon": spec.log_epsilon,
                "required": True,
            },
        },
    }


def _projection_spec(observations: list[str], start: str, end: str) -> dict[str, Any]:
    return {
        "type": "eof_projection",
        "basis": "aod_basis",
        "model": "model_daily",
        "model_variable": "log_aod",
        "obs": [
            {
                "source": source,
                "variable": "log_aod",
                "error_variable": "obs_error_std",
            }
            for source in observations
        ],
        "ridge": 1.0,
        "bias_fit_window": {"start": start, "end": end},
        "clim_bias": False,
        "support_min_fraction": 0.0,
        "support_full_fraction": 0.01,
        "support_smoothing_passes": 0,
        "min_resolution": 0.0,
        "required": True,
    }


def _run(config: dict[str, Any]) -> PipelineResult:
    return PipelineRunner(
        show_progress=False,
        close_datasets_after_run=False,
    ).run_from_config(config)


def _analytic_projection(
    basis: xr.Dataset,
    model: xr.Dataset,
    observations: list[xr.Dataset],
    day: int,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    patterns = np.asarray(basis["eofs"].values, dtype=np.float64)
    model_log = np.asarray(model["log_aod"].isel(time=day).values, dtype=np.float64)
    latitude = np.broadcast_to(np.asarray(model["lat"].values)[:, None], model_log.shape)
    information = np.zeros((patterns.shape[0], patterns.shape[0]), dtype=np.float64)
    right_hand_side = np.zeros(patterns.shape[0], dtype=np.float64)
    for observation in observations:
        valid = np.asarray(observation["valid"].isel(time=day).values, dtype=bool)
        flat = np.flatnonzero(valid.ravel())
        design = patterns.reshape(patterns.shape[0], -1)[:, flat].T
        innovation = (
            np.asarray(observation["log_aod"].isel(time=day).values).ravel()[flat]
            - model_log.ravel()[flat]
        )
        sigma = np.asarray(observation["obs_error_std"].isel(time=day).values).ravel()[flat]
        precision = np.cos(np.deg2rad(latitude.ravel()[flat])) / np.square(sigma)
        information += design.T @ (precision[:, None] * design)
        right_hand_side += design.T @ (precision * innovation)
    normal = information + ridge * np.eye(patterns.shape[0])
    return np.linalg.solve(normal, right_hand_side), np.diag(np.linalg.inv(normal))


def _assert_qa_counts(context: Any, truth: xr.Dataset) -> None:
    valid = np.asarray(truth["valid_mask"].values, dtype=bool)
    for sensor_index, sensor in enumerate(("sensor_a", "sensor_b")):
        raw = context.sources[f"{sensor}_raw"].data
        daily = context.sources[f"{sensor}_daily"].data
        assert int((raw["QA"] == 3).sum()) == int(np.count_nonzero(valid[sensor_index]))
        np.testing.assert_array_equal(daily["valid"].values, valid[sensor_index])


@pytest.mark.integration
def test_aerosol_tuning_multi_sensor_projection(tmp_path: Path) -> None:
    """T2: QA, precision, absent-sensor, and posterior contracts use the real DAG."""
    spec = SyntheticTuningSpec.multi_sensor_ci()
    bundle = generate_aerosol_tuning_bundle(tmp_path, spec)
    config = _pipeline_inputs(tmp_path, spec)
    start = f"{spec.time_config.start} 00:00:00"
    end = f"{spec.time_config.end} 23:59:59"
    config["analyses"]["aod_basis"] = {
        "type": "eof",
        "source": "model_daily",
        "variable": "log_aod",
        "n_modes": spec.n_modes,
        "standardize": False,
        "rotation": "none",
        "solver": "full",
        "fit_window": {"start": start, "end": end},
        "required": True,
    }
    config["analyses"]["a_only"] = _projection_spec(["sensor_a_daily"], start, end)
    config["analyses"]["b_only"] = _projection_spec(["sensor_b_daily"], start, end)
    config["analyses"]["blended"] = _projection_spec(
        ["sensor_a_daily", "sensor_b_daily"], start, end
    )

    result = _run(config)

    assert result.success, result.stage_errors
    assert result.context is not None
    context = result.context
    assert not context.metadata.get("analysis_errors")
    _assert_qa_counts(context, bundle.truth)
    valid = np.asarray(bundle.truth["valid_mask"].values, dtype=bool)
    expected_counts = valid.sum(axis=(2, 3)).T
    np.testing.assert_array_equal(
        context.sources["blended"].data["n_obs"].values,
        expected_counts,
    )

    basis = context.sources["aod_basis"].data
    model = context.sources["model_daily"].data
    sensor_a = context.sources["sensor_a_daily"].data
    sensor_b = context.sources["sensor_b_daily"].data
    cases = {
        "a_only": [sensor_a],
        "b_only": [sensor_b],
        "blended": [sensor_a, sensor_b],
    }
    for name, observations in cases.items():
        expected_pc, expected_posterior = _analytic_projection(
            basis, model, observations, 0, ridge=1.0
        )
        actual = context.sources[name].data
        np.testing.assert_allclose(actual["pc"].isel(time=0), expected_pc, rtol=2e-8, atol=5e-9)
        np.testing.assert_allclose(
            actual["posterior_variance"].isel(time=0),
            expected_posterior,
            rtol=2e-8,
            atol=5e-9,
        )

    blended = context.sources["blended"].data
    innovation = np.asarray(blended["innovation_mean"].isel(time=0).values)
    model_log = np.asarray(model["log_aod"].isel(time=0).values)
    a_log = np.asarray(sensor_a["log_aod"].isel(time=0).values)
    b_log = np.asarray(sensor_b["log_aod"].isel(time=0).values)
    a_valid = valid[0, 0]
    b_valid = valid[1, 0]
    a_alone = a_valid & ~b_valid
    b_alone = b_valid & ~a_valid
    overlap = a_valid & b_valid
    np.testing.assert_allclose(innovation[a_alone], (a_log - model_log)[a_alone])
    np.testing.assert_allclose(innovation[b_alone], (b_log - model_log)[b_alone])
    sigma_a, sigma_b = spec.sensor_error_sigma
    expected_overlap = ((a_log - model_log) / sigma_a**2 + (b_log - model_log) / sigma_b**2) / (
        1.0 / sigma_a**2 + 1.0 / sigma_b**2
    )
    np.testing.assert_allclose(innovation[overlap], expected_overlap[overlap], rtol=1e-10)
    assert np.all(blended["innovation_count"].isel(time=0).values[a_alone | b_alone] == 1)

    blend_variance = np.asarray(blended["posterior_variance"].isel(time=0).values)
    for name in ("a_only", "b_only"):
        single = np.asarray(context.sources[name].data["posterior_variance"].isel(time=0).values)
        assert np.all(blend_variance <= single + 1.0e-12)
        assert np.any(blend_variance < single - 1.0e-10)


def _write_prescribed_basis(path: Path, bundle: SyntheticTuningBundle) -> None:
    patterns = bundle.truth["pattern_true"].rename(
        {"truth_mode": "mode", "mode_lat": "lat", "mode_lon": "lon"}
    )
    patterns = patterns.assign_coords(mode=np.arange(1, patterns.sizes["mode"] + 1))
    patterns.attrs.update(units="1", kind="eofs")
    basis = xr.Dataset({"eofs": patterns})
    basis.attrs.update(
        eof_rotation="none",
        eof_standardize="false",
        eof_input_log_epsilon=bundle.spec.log_epsilon,
        spec_hash=bundle.truth.attrs["spec_hash"],
        synthetic="true",
    )
    basis.to_netcdf(path)


def _false_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    missing = ~np.asarray(valid, dtype=bool)
    changes = np.diff(np.pad(missing.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


@pytest.mark.integration
def test_aerosol_tuning_null_and_gap_behavior(tmp_path: Path) -> None:
    """T4: all-missing, bounded-gap, identity, and frozen null-energy gates."""
    spec = SyntheticTuningSpec.null_ci()
    bundle = generate_aerosol_tuning_bundle(tmp_path, spec)
    basis_path = tmp_path / "inputs/prescribed_basis.nc"
    _write_prescribed_basis(basis_path, bundle)
    config = _pipeline_inputs(tmp_path, spec)
    config["sources"]["aod_basis"] = {
        "type": "generic",
        "files": str(basis_path),
        "variables": {"eofs": {"units": "1"}},
    }
    start = f"{spec.time_config.start} 00:00:00"
    end = f"{spec.time_config.end} 23:59:59"
    projection_spec = _projection_spec(["sensor_a_daily", "sensor_b_daily"], start, end)
    projection_spec.update(
        support_min_fraction=spec.support_min_fraction,
        support_full_fraction=spec.support_full_fraction,
        support_smoothing_passes=2,
        min_resolution=0.3,
    )
    config["analyses"]["obs_pcs"] = projection_spec
    config["analyses"]["filtered_pcs"] = {
        "type": "wavelet_filter",
        "source": "obs_pcs",
        "variable": "pc",
        "resolution_variable": "resolution",
        "min_resolution": 0.3,
        "keep_significant": True,
        "significance_level": 0.95,
        "band": {"min": 4.0, "max": 16.0, "units": "days"},
        "max_bridge_days": 7.0,
        "min_segment_days": 32.0,
        "omega0": 6.0,
        "required": True,
    }
    config["analyses"]["scaling"] = {
        "type": "aod_scaling",
        "basis": "aod_basis",
        "projection": "obs_pcs",
        "coefficients": "filtered_pcs",
        "model": "model_daily",
        "r_bounds": list(spec.r_bounds),
        "aod_floor": spec.aod_floor,
        "required": True,
    }

    result = _run(config)

    assert result.success, result.stage_errors
    assert result.context is not None
    context = result.context
    projection = context.sources["obs_pcs"].data
    filtered = context.sources["filtered_pcs"].data
    scaling = context.sources["scaling"].data
    truth = bundle.truth

    valid = np.asarray(truth["valid_mask"].values, dtype=bool)
    all_missing = ~valid.any(axis=(0, 2, 3))
    assert np.any(all_missing)
    assert np.all(projection["n_obs"].values[all_missing] == 0)
    assert np.all(projection["pc"].values[all_missing] == 0.0)
    assert np.all(projection["resolution"].values[all_missing] == 0.0)
    assert np.all(projection["pc"].isel(mode=-1).values == 0.0)
    assert np.all(projection["resolution"].isel(mode=-1).values == 0.0)

    observable = np.asarray(truth["mode_observable_true"].values, dtype=bool)
    short_gap = np.asarray(truth["short_gap_day"].values, dtype=bool)
    long_gap = np.asarray(truth["long_gap_day"].values, dtype=bool)
    bridged = np.asarray(filtered["bridged"].values, dtype=bool)
    assert np.all(bridged[short_gap][:, observable])
    assert not np.any(bridged[long_gap])
    source_valid = np.asarray(projection["resolution"].values >= 0.3, dtype=bool)
    for mode in range(source_valid.shape[1]):
        for gap_start, gap_end in _false_runs(source_valid[:, mode]):
            bounded = gap_start > 0 and gap_end < source_valid.shape[0]
            expected = bounded and gap_end - gap_start <= 7
            assert np.all(bridged[gap_start:gap_end, mode] == expected)

    ratio = np.asarray(scaling["r"].values)
    applied = np.asarray(scaling["delta_log_applied"].values)
    np.testing.assert_allclose(ratio[long_gap], 1.0, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(applied[long_gap], 0.0, rtol=0.0, atol=2.0e-15)
    unsupported = np.asarray(truth["unobservable_region_true"].values, dtype=bool)
    np.testing.assert_allclose(ratio[:, unsupported], 1.0, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(applied[:, unsupported], 0.0, rtol=0.0, atol=2.0e-15)
    zero_support = np.asarray(scaling["spatial_support"].values) <= 0.0
    np.testing.assert_allclose(ratio[zero_support], 1.0, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(applied[zero_support], 0.0, rtol=0.0, atol=2.0e-15)

    split = np.asarray(truth["split"].values).astype(str) == "development_test"
    valid_segment = np.asarray(filtered["valid_segment"].values, dtype=bool)
    coi = np.asarray(filtered["coi"].values)
    periods = np.asarray(filtered["period"].values)
    in_band = (periods >= 4.0) & (periods <= 16.0)
    maximum_period = float(periods[in_band].max())
    score_days = split & np.all(
        valid_segment[:, observable] & (coi[:, observable] >= maximum_period), axis=1
    )
    assert np.any(score_days), "null scenario has no non-COI development-test days"
    noise = np.asarray(truth["innovation_noise_true"].values)
    scoring = score_days[:, None, None] & ~zero_support & np.isfinite(noise)
    weights = np.cos(np.deg2rad(np.asarray(scaling["lat"].values)))[:, None]
    numerator = float(np.sum(np.where(scoring, weights[None] * applied**2, 0.0)))
    denominator = float(np.sum(np.where(scoring, weights[None] * noise**2, 0.0)))
    assert denominator > 0.0
    false_positive_energy = numerator / denominator
    assert false_positive_energy <= 0.10, false_positive_energy

    significance = np.asarray(filtered["power_significance"].values)
    candidates = (
        split[:, None, None]
        & observable[None, :, None]
        & valid_segment[:, :, None]
        & in_band[None, None, :]
        & (periods[None, None, :] <= coi[:, :, None])
        & np.isfinite(significance)
    )
    assert np.any(candidates)
    significant_fraction = float(np.count_nonzero(significance[candidates] >= 1.0)) / float(
        np.count_nonzero(candidates)
    )
    assert significant_fraction <= 0.10, significant_fraction
