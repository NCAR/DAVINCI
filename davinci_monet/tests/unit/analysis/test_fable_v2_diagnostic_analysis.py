"""Tests for the registered FABLE v2 diagnostic adapter."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.analysis.fable_v2_diagnostic_analysis import (
    evaluate_fable_v2_diagnostics,
)
from davinci_monet.analysis.fable_v2_diagnostics import V2_STAGE_NAMES
from davinci_monet.analysis.fable_v2_projection_diagnostics import (
    masked_projection_coefficients,
)
from davinci_monet.analysis.projection_batches import solve_projection_batched
from davinci_monet.analysis.projection_core import MonthlyBiasFit
from davinci_monet.analysis.projection_inputs import ProjectionObservation


def _inputs() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    time = np.arange("2006-01-01", "2006-01-05", dtype="datetime64[D]")
    lat = [-30.0, 30.0]
    lon = [0.0, 180.0]
    mode = [1]
    basis = xr.DataArray(
        [[[1.0, -1.0], [1.0, -1.0]]],
        dims=("mode", "lat", "lon"),
        coords={"mode": mode, "lat": lat, "lon": lon},
    )
    pc = xr.DataArray(
        [[0.1], [-0.2], [0.3], [-0.1]],
        dims=("time", "mode"),
        coords={"time": time, "mode": mode},
    )
    anomaly = xr.dot(pc, basis, dim="mode").transpose("time", "lat", "lon")
    bias = xr.full_like(anomaly, 0.025)
    target = bias + anomaly
    support = xr.ones_like(target)
    estimate = xr.Dataset(
        {
            "eofs": basis,
            "pc": pc,
            "delta_log_requested": target,
            "delta_log_applied": target,
            "clim_bias_applied": bias,
            "spatial_support": support,
            "coefficient_available": ("time", np.ones(time.size, dtype=bool)),
            "valid_segment": (
                ("time", "mode"),
                np.ones((time.size, len(mode)), dtype=bool),
            ),
            "coi": (
                ("time", "mode"),
                np.asarray([[10.0], [2.0], [10.0], [2.0]]),
            ),
        },
        attrs={"band_max": 8.0},
    )
    projection = xr.Dataset(
        {
            "pc": pc,
            "n_obs": (
                ("time", "sensor"),
                np.full((time.size, 2), 4, dtype=np.int64),
            ),
            "sensor_offset": ("sensor", [0.01, -0.01]),
            "sensor_offset_standard_error": ("sensor", [0.001, 0.001]),
            "sensor_overlap_count": (
                ("sensor", "sensor_pair"),
                [[0, 12], [12, 0]],
            ),
        },
        coords={
            "sensor": ["sensor_a_daily", "sensor_b_daily"],
            "sensor_pair": ["sensor_a_daily", "sensor_b_daily"],
        },
        attrs={"projection_sensor_offset_method": "overlap_zero_sum"},
    )
    monthly_bias = np.full((12, len(lat), len(lon)), 0.025)
    monthly_support = np.ones_like(monthly_bias)
    sensor_shape = (2, time.size, len(lat), len(lon))
    truth = xr.Dataset(
        {
            "delta_filter_target_true": target,
            "delta_best_representable_true": 2.0 * target,
            "clim_bias_raw_true": (("month", "lat", "lon"), monthly_bias),
            "spatial_support_true": (("month", "lat", "lon"), monthly_support),
            "valid_mask": (
                ("sensor", "time", "lat", "lon"),
                np.ones(sensor_shape, dtype=np.uint8),
            ),
            "reported_sigma_log": (
                ("sensor", "time", "lat", "lon"),
                np.full(sensor_shape, 0.02),
            ),
            "obs_error_log": (
                ("sensor", "time", "lat", "lon"),
                np.zeros(sensor_shape),
            ),
            "pattern_true": (
                ("truth_mode", "lat", "lon"),
                basis.values,
            ),
            "mode_observable_true": ("truth_mode", [1]),
            "correction_pc_true": (("time", "truth_mode"), pc.values),
            "correction_pc_filter_target_true": (
                ("time", "truth_mode"),
                pc.values,
            ),
            "sensor_bias_log_true": ("sensor", [0.01, -0.01]),
            "split": ("time", ["development_test"] * time.size),
        },
        coords={
            "sensor": ["sensor_a", "sensor_b"],
            "time": time,
            "truth_mode": mode,
            "month": np.arange(1, 13),
            "lat": lat,
            "lon": lon,
        },
    )
    return estimate, projection, truth


def test_adapter_builds_all_seven_stages_with_nonranking_provenance() -> None:
    estimate, projection, truth = _inputs()

    report = evaluate_fable_v2_diagnostics(
        estimate,
        projection,
        truth,
        {
            "evaluation_splits": ["development_test"],
            "projection_to_truth_sensor": {
                "sensor_a_daily": "sensor_a",
                "sensor_b_daily": "sensor_b",
            },
            "reported_common_factor_amplitude": 0.04,
        },
    )

    np.testing.assert_array_equal(report["stage"], V2_STAGE_NAMES)
    assert report["learned_basis_oracle_nrmse"].item() < 1.0e-12
    assert report["stage_field_nrmse"].sel(stage="final_policy").item() < 1.0e-12
    assert report["estimate_vs_unfiltered_in_span_nrmse"].item() == 0.5
    assert report.attrs["diagnostic_only"] == "true"
    assert report.attrs["eligible_for_calibration"] == "false"
    assert report.attrs["projection_noise_covariance"] == (
        "reported_diagonal_plus_shared_constant_low_rank_factor"
    )
    assert report.attrs["reported_common_factor_amplitude"] == 0.04
    assert report.attrs["supplemental_evidence_use"] == "diagnostic_only_nonranking"
    assert report.attrs["absolute_sensor_offset_identifiable"] == "false"
    assert report.attrs["sensor_coordinate_mapping"] == (
        "sensor_a_daily->sensor_a,sensor_b_daily->sensor_b"
    )
    np.testing.assert_array_equal(report["sensor"], ["sensor_a", "sensor_b"])
    assert report["common_bias_nrmse"].item() == 0.0
    assert report["relative_sensor_offset_nrmse"].item() == 0.0
    assert report["coefficient_nrmse"].sel(coefficient_stage="unfiltered_projection").item() == 0.0
    assert report["coefficient_nrmse"].sel(coefficient_stage="post_wavelet").item() == 0.0
    assert set(report["diagnostic_stratum"].values.astype(str)) == {
        "primary",
        "full_domain",
        "support_zero",
        "support_partial",
        "support_full",
        "coi_interior",
        "coi_edge",
        "segment_unavailable",
    }
    assert (
        report["stratum_stage_valid_count"]
        .sel(
            diagnostic_stratum="coi_interior",
            stage="final_policy",
        )
        .item()
        == 8
    )
    assert (
        report["stratum_stage_valid_count"]
        .sel(
            diagnostic_stratum="coi_edge",
            stage="final_policy",
        )
        .item()
        == 8
    )


def test_adapter_materializes_chunked_inputs_once_before_metric_reductions() -> None:
    eager_inputs = _inputs()
    chunked_inputs = tuple(dataset.chunk({"time": 2}) for dataset in _inputs())
    spec = {
        "evaluation_splits": ["development_test"],
        "projection_to_truth_sensor": {
            "sensor_a_daily": "sensor_a",
            "sensor_b_daily": "sensor_b",
        },
        "reported_common_factor_amplitude": 0.04,
    }

    eager = evaluate_fable_v2_diagnostics(eager_inputs[0], eager_inputs[1], eager_inputs[2], spec)
    chunked = evaluate_fable_v2_diagnostics(
        chunked_inputs[0], chunked_inputs[1], chunked_inputs[2], spec
    )

    xr.testing.assert_allclose(chunked, eager)
    assert all(value.chunks is None for value in chunked.data_vars.values())


def test_adapter_rejects_noncovering_sensor_mapping() -> None:
    estimate, projection, truth = _inputs()

    with np.testing.assert_raises_regex(
        ValueError,
        "keys must exactly cover projection sensor coordinates",
    ):
        evaluate_fable_v2_diagnostics(
            estimate,
            projection,
            truth,
            {
                "evaluation_splits": ["development_test"],
                "projection_to_truth_sensor": {"sensor_a_daily": "sensor_a"},
                "reported_common_factor_amplitude": 0.04,
            },
        )


def test_adapter_maps_control_sensor_coordinates_without_an_offset_fit() -> None:
    estimate, projection, truth = _inputs()
    projection = projection.drop_vars(
        ["sensor_offset", "sensor_offset_standard_error", "sensor_overlap_count"]
    )
    projection.attrs["projection_sensor_offset_method"] = "none"

    report = evaluate_fable_v2_diagnostics(
        estimate,
        projection,
        truth,
        {
            "evaluation_splits": ["development_test"],
            "projection_to_truth_sensor": {
                "sensor_a_daily": "sensor_a",
                "sensor_b_daily": "sensor_b",
            },
            "reported_common_factor_amplitude": 0.04,
        },
    )

    np.testing.assert_array_equal(report["sensor"], ["sensor_a", "sensor_b"])
    np.testing.assert_array_equal(report["fitted_relative_sensor_offset"], [0.0, 0.0])
    assert report.attrs["sensor_offset_method"] == "none"


def test_masked_projection_uses_common_factor_and_fitted_offsets() -> None:
    time = [np.datetime64("2006-01-01")]
    lat = [0.0]
    lon = [0.0, 180.0]
    sensor = ["a", "b"]
    target = xr.DataArray(
        [[[1.0, 0.0]]],
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
    )
    basis = xr.DataArray(
        [[[1.0, 2.0]]],
        dims=("mode", "lat", "lon"),
        coords={"mode": [1], "lat": lat, "lon": lon},
    )
    shape = (len(sensor), len(time), len(lat), len(lon))
    sensor_error = np.asarray([0.2, -0.2])[:, None, None, None] * np.ones(shape)
    truth = xr.Dataset(
        {
            "valid_mask": (("sensor", "time", "lat", "lon"), np.ones(shape, dtype=bool)),
            "reported_sigma_log": (
                ("sensor", "time", "lat", "lon"),
                np.full(shape, 0.1),
            ),
            "obs_error_log": (("sensor", "time", "lat", "lon"), sensor_error),
        },
        coords={"sensor": sensor, "time": time, "lat": lat, "lon": lon},
    )
    zero = xr.zeros_like(target)
    support = xr.ones_like(target)
    partial_support = xr.DataArray(
        [[[0.25, 0.75]]],
        dims=target.dims,
        coords=target.coords,
    )
    offsets = xr.DataArray([0.2, -0.2], dims="sensor", coords={"sensor": sensor})

    diagonal = masked_projection_coefficients(
        target,
        basis,
        zero,
        support,
        truth,
        noisy=False,
        common_factor_amplitude=0.0,
    )
    structured = masked_projection_coefficients(
        target,
        basis,
        zero,
        support,
        truth,
        noisy=False,
        common_factor_amplitude=0.5,
    )
    offset_corrected = masked_projection_coefficients(
        target,
        basis,
        zero,
        support,
        truth,
        noisy=True,
        common_factor_amplitude=0.0,
        sensor_offsets=offsets,
    )
    partial = masked_projection_coefficients(
        partial_support * target,
        basis,
        zero,
        partial_support,
        truth,
        noisy=False,
        common_factor_amplitude=0.0,
    )

    factor = xr.DataArray(
        np.zeros((1, 1, 1, 2)),
        dims=("time", "common_mode", "lat", "lon"),
        coords={
            "time": time,
            "common_mode": ["shared_sensor_error"],
            "lat": lat,
            "lon": lon,
        },
    )
    observations = tuple(
        ProjectionObservation(
            name=name,
            values=target,
            errors=xr.full_like(target, 0.1),
            valid=xr.ones_like(target, dtype=bool),
            factors=factor,
            factor_names=("shared_sensor_error",),
        )
        for name in sensor
    )
    monthly = np.zeros((12, 1, 2), dtype=np.float64)
    monthly_support = monthly.copy()
    monthly_support[0] = partial_support.values[0]
    fit = MonthlyBiasFit(
        raw_mean=monthly.copy(),
        bias=monthly.copy(),
        bias_applied=monthly.copy(),
        support=monthly_support,
        support_fraction=monthly_support.copy(),
        support_count=np.zeros_like(monthly, dtype=np.int64),
        support_day_total=np.zeros(12, dtype=np.int64),
        sensor_count=np.zeros((12, 2, 1, 2), dtype=np.int64),
        standard_error=monthly.copy(),
    )
    production = solve_projection_batched(
        zero,
        observations,
        np.asarray(basis.values),
        fit,
        np.asarray([1]),
        apply_bias=True,
        ridge=1.0,
        time_chunk_size=1,
    )
    production_pc = xr.DataArray(
        production.coefficients,
        dims=("time", "mode"),
        coords={"time": time, "mode": [1]},
    )
    diagnostic_final = partial_support * xr.dot(partial, basis, dim="mode")
    production_final = partial_support * xr.dot(production_pc, basis, dim="mode")

    assert structured.item() != diagonal.item()
    xr.testing.assert_allclose(offset_corrected, diagonal)
    xr.testing.assert_allclose(partial, diagonal)
    xr.testing.assert_allclose(diagnostic_final, production_final)
