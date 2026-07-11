"""Tests for versioned, evaluation-only FABLE v2 diagnostics."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import xarray as xr

from davinci_monet.analysis.fable_v2_diagnostic_evidence import (
    COEFFICIENT_STAGE_NAMES,
    coefficient_stage_evidence,
    common_bias_evidence,
    relative_sensor_offset_evidence,
    score_stagewise_strata,
)
from davinci_monet.analysis.fable_v2_diagnostics import (
    V2_STAGE_NAMES,
    build_fable_v2_diagnostic_report,
    learned_basis_filtered_target_oracle,
    reconstruct_stage_field,
    score_stagewise_fields,
)


def _basis() -> xr.DataArray:
    return xr.DataArray(
        [
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        ],
        dims=("mode", "lat", "lon"),
        coords={"mode": [1, 2], "lat": [-60.0, 0.0, 60.0], "lon": [0.0, 180.0]},
    )


def _coefficients() -> xr.DataArray:
    return xr.DataArray(
        [[0.2, -0.1], [-0.3, 0.4]],
        dims=("time", "mode"),
        coords={
            "time": np.arange("2006-01-01", "2006-01-03", dtype="datetime64[D]"),
            "mode": [1, 2],
        },
    )


def test_learned_basis_oracle_exactly_recovers_filtered_target() -> None:
    basis = _basis()
    coefficients = _coefficients()
    template = xr.dot(coefficients, basis, dim="mode").transpose("time", "lat", "lon")
    support = xr.ones_like(template)
    support.loc[{"lat": -60.0}] = 0.5
    bias = xr.full_like(template, 0.025) * support
    target = bias + support * template

    result = learned_basis_filtered_target_oracle(
        target,
        basis,
        oracle_bias=bias,
        spatial_multiplier=support,
    )

    xr.testing.assert_allclose(result["learned_basis_oracle_pc"], coefficients)
    xr.testing.assert_allclose(result["learned_basis_oracle_delta"], target)
    np.testing.assert_array_equal(result["learned_basis_oracle_fit_rank"], [2, 2])
    assert result["learned_basis_oracle_nrmse"].item() == pytest.approx(0.0, abs=1.0e-15)
    assert result.attrs["fit_method"] == ("daily_full_grid_cosine_latitude_weighted_least_squares")
    assert result.attrs["diagnostic_only"] == "true"
    assert result.attrs["eligible_for_calibration"] == "false"
    assert result.attrs["oracle_bias_supplied"] == "true"
    assert result.attrs["spatial_multiplier_supplied"] == "true"


def test_learned_basis_oracle_uses_cosine_latitude_weights() -> None:
    time = [np.datetime64("2006-01-01")]
    basis = xr.DataArray(
        np.ones((1, 2, 1)),
        dims=("mode", "lat", "lon"),
        coords={"mode": [1], "lat": [0.0, 60.0], "lon": [0.0]},
    )
    target = xr.DataArray(
        [[[0.0], [2.0]]],
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": basis["lat"], "lon": basis["lon"]},
    )

    result = learned_basis_filtered_target_oracle(target, basis)

    assert result["learned_basis_oracle_pc"].item() == pytest.approx(2.0 / 3.0)


def test_reconstruct_stage_field_requires_exact_mode_coordinates() -> None:
    coefficients = _coefficients().assign_coords(mode=[3, 4])

    with pytest.raises(ValueError, match="coefficient modes must match"):
        reconstruct_stage_field(_basis(), coefficients)


def test_stagewise_report_has_corrected_names_common_mask_and_provenance() -> None:
    basis = _basis()
    coefficients = _coefficients()
    target = xr.dot(coefficients, basis, dim="mode").transpose("time", "lat", "lon") + 1.0
    scales = (1.0, 0.8, 1.1, 0.7, 0.9, 1.05, 1.0)
    stages = OrderedDict(
        (name, scale * target.copy()) for name, scale in zip(V2_STAGE_NAMES, scales, strict=True)
    )
    stages["oracle_bias_noisy_projection"].values[0, 0, 0] = np.nan
    mask = xr.ones_like(target, dtype=bool)
    mask.values[0, 0, 1] = False

    report = build_fable_v2_diagnostic_report(
        stages,
        target,
        unfiltered_in_span=2.0 * target,
        mask=mask,
    )

    np.testing.assert_array_equal(report["stage"], V2_STAGE_NAMES)
    np.testing.assert_array_equal(report["stage_valid_count"], [10] * len(V2_STAGE_NAMES))
    assert report["learned_basis_oracle_nrmse"].item() == pytest.approx(0.0)
    assert report["estimate_vs_unfiltered_in_span_nrmse"].item() == pytest.approx(0.5)
    assert "best_representable_nrmse" not in report
    assert np.isnan(report["stage_nrmse_change_from_previous"].isel(stage=0).item())
    assert report["stage_nrmse_change_from_previous"].sel(
        stage="oracle_bias_noiseless_projection"
    ).item() == pytest.approx(0.2)
    assert report["stage_transition_nrmse"].sel(
        stage="oracle_bias_noiseless_projection"
    ).item() == pytest.approx(0.2)
    assert report["stage_transition_nrmse"].attrs["additivity"] == "non_additive"
    assert report.attrs["diagnostic_only"] == "true"
    assert report.attrs["eligible_for_calibration"] == "false"
    assert report.attrs["calibration_use"] == "prohibited"


def test_stagewise_report_requires_exact_preregistered_stage_order() -> None:
    target = xr.dot(_coefficients(), _basis(), dim="mode").transpose("time", "lat", "lon")
    stages = OrderedDict((name, target) for name in reversed(V2_STAGE_NAMES))

    with pytest.raises(ValueError, match="preregistered order exactly"):
        build_fable_v2_diagnostic_report(
            stages,
            target,
            unfiltered_in_span=target,
        )


def test_stagewise_fields_reject_changed_coordinates() -> None:
    target = xr.dot(_coefficients(), _basis(), dim="mode").transpose("time", "lat", "lon")
    changed = target.assign_coords(lon=[0.0, 179.0])

    with pytest.raises(ValueError, match="coordinates must match exactly"):
        score_stagewise_fields({"stage": changed}, target)

    with pytest.raises(ValueError, match="selects no common finite values"):
        score_stagewise_fields({"stage": target}, target, xr.zeros_like(target, dtype=bool))


def test_common_bias_evidence_reports_fields_metrics_and_gauge_caveat() -> None:
    target = xr.dot(_coefficients(), _basis(), dim="mode").transpose("time", "lat", "lon")
    truth = 0.25 * target
    fitted = truth.copy()
    mask = xr.ones_like(truth, dtype=bool)
    mask.loc[{"lat": -60.0}] = False

    result = common_bias_evidence(fitted, truth, mask)

    xr.testing.assert_allclose(result["fitted_common_bias"], fitted)
    xr.testing.assert_allclose(result["true_common_bias"], truth)
    assert result["common_bias_nrmse"].item() == 0.0
    assert result["common_bias_valid_count"].item() == 8
    assert "unidentifiable mean absolute sensor offset" in result.attrs["common_bias_gauge_caveat"]
    assert result.attrs["diagnostic_only"] == "true"


def test_relative_sensor_offsets_remove_only_the_unidentifiable_mean() -> None:
    sensors = ["a", "b", "c"]
    truth = xr.DataArray([0.13, 0.10, 0.07], dims="sensor", coords={"sensor": sensors})
    fitted = xr.DataArray([0.03, 0.0, -0.03], dims="sensor", coords={"sensor": sensors})
    standard_error = xr.DataArray([0.01, 0.02, 0.01], dims="sensor", coords={"sensor": sensors})
    overlap = xr.DataArray(
        np.arange(9).reshape(3, 3),
        dims=("sensor", "sensor_pair"),
        coords={"sensor": sensors, "sensor_pair": sensors},
    )

    result = relative_sensor_offset_evidence(
        fitted,
        truth,
        standard_error=standard_error,
        overlap_count=overlap,
    )

    np.testing.assert_allclose(
        result["true_relative_sensor_offset"], [0.03, 0.0, -0.03], atol=1.0e-15
    )
    np.testing.assert_allclose(
        result["fitted_relative_sensor_offset"], [0.03, 0.0, -0.03], atol=1.0e-15
    )
    assert result["true_absolute_sensor_offset_mean"].item() == pytest.approx(0.10)
    assert result["relative_sensor_offset_nrmse"].item() == pytest.approx(0.0)
    xr.testing.assert_allclose(result["relative_sensor_offset_standard_error"], standard_error)
    xr.testing.assert_allclose(result["sensor_offset_overlap_count"], overlap)
    assert result.attrs["absolute_sensor_offset_fit_status"] == (
        "not_scored_scientifically_unidentifiable"
    )


def test_coefficient_evidence_matches_modes_and_scores_pre_post_filter() -> None:
    learned = _basis()
    truth_basis = xr.concat(
        (-2.0 * learned.isel(mode=1), 0.5 * learned.isel(mode=0)),
        dim=xr.IndexVariable("mode", [10, 20]),
    )
    estimate = _coefficients()
    truth_pc = xr.DataArray(
        np.column_stack((-0.5 * estimate[:, 1], 2.0 * estimate[:, 0])),
        dims=("time", "mode"),
        coords={"time": estimate["time"], "mode": [10, 20]},
    )
    mask = xr.DataArray(
        [True, True],
        dims=("time",),
        coords={"time": estimate["time"]},
    )

    result = coefficient_stage_evidence(
        learned,
        truth_basis,
        {
            COEFFICIENT_STAGE_NAMES[0]: 0.5 * estimate,
            COEFFICIENT_STAGE_NAMES[1]: estimate,
        },
        {
            COEFFICIENT_STAGE_NAMES[0]: truth_pc,
            COEFFICIENT_STAGE_NAMES[1]: truth_pc,
        },
        mask,
        observable_modes=xr.DataArray([True, True], dims="mode", coords={"mode": [10, 20]}),
    )

    np.testing.assert_allclose(
        result["coefficient_origin_slope"].sel(coefficient_stage="unfiltered_projection"),
        [0.5, 0.5],
    )
    np.testing.assert_allclose(
        result["coefficient_nrmse"].sel(coefficient_stage="unfiltered_projection"),
        [0.5, 0.5],
    )
    np.testing.assert_allclose(
        result["coefficient_nrmse"].sel(coefficient_stage="post_wavelet"),
        [0.0, 0.0],
        atol=1.0e-15,
    )
    assert result.attrs["coefficient_matching"] == (
        "cosine_latitude_weighted_Hungarian_sign_and_scale"
    )
    assert result.attrs["eligible_for_calibration"] == "false"


def test_stagewise_strata_use_explicit_candidate_domains() -> None:
    target = xr.dot(_coefficients(), _basis(), dim="mode").transpose("time", "lat", "lon")
    support_mask = xr.zeros_like(target, dtype=bool)
    support_mask.loc[{"lat": 0.0}] = True
    empty = xr.zeros_like(target, dtype=bool)

    result = score_stagewise_strata(
        {"exact": target, "half": 0.5 * target},
        target,
        {
            "supported": ("support", support_mask),
            "coi_edge": ("coi", empty),
        },
    )

    assert (
        result["stratum_stage_candidate_count"]
        .sel(diagnostic_stratum="supported", stage="exact")
        .item()
        == 4
    )
    assert result["stratum_stage_field_nrmse"].sel(
        diagnostic_stratum="supported", stage="half"
    ).item() == pytest.approx(0.5)
    assert (
        result["stratum_stage_valid_count"].sel(diagnostic_stratum="coi_edge", stage="exact").item()
        == 0
    )
    assert result.attrs["calibration_use"] == "prohibited"
