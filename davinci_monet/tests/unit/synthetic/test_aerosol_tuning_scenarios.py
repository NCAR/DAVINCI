"""Locked scenario-ladder contracts for the FABLE synthetic generator."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.analysis.aod_preprocess import preprocess_aod
from davinci_monet.analysis.artifacts import ArtifactService
from davinci_monet.analysis.base import AnalysisRuntime
from davinci_monet.config import load_config
from davinci_monet.config.schema import AODPreprocessSpec, EOFProjectionSpec
from davinci_monet.pipeline.stages import LoadSourcesStage, PipelineContext, StageStatus
from davinci_monet.tests.synthetic._aerosol_inputs import monthly_support, split_labels
from davinci_monet.tests.synthetic.aerosol_tuning import (
    SyntheticTuningSpec,
    generate_aerosol_tuning_bundle,
)
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig


def test_reduced_osse_generator_broadcasts_spatial_model_terms() -> None:
    spec = replace(
        SyntheticTuningSpec.synthetic_osse(20260712),
        native_domain=Domain(-180.0, 180.0, -90.0, 90.0, 12, 6),
        mode_domain=Domain(-180.0, 180.0, -90.0, 90.0, 6, 3),
        time_config=TimeConfig("2001-01-01", "2001-01-12", "1h"),
        split_windows=(
            ("basis_train", "2001-01-01", "2001-01-03"),
            ("bias_fit", "2001-01-04", "2001-01-06"),
            ("calibration", "2001-01-07", "2001-01-09"),
            ("development_test", "2001-01-10", "2001-01-12"),
        ),
    )

    bundle = generate_aerosol_tuning_bundle(spec)

    assert bundle.model["TOTEXTTAU"].dims == ("time", "lat", "lon")
    assert bundle.model["TOTEXTTAU"].shape == (336, 6, 12)
    assert bundle.truth.sizes["time"] == 12
    assert set(bundle.observations) == {"sensor_a", "sensor_b"}


def test_reduced_full_size_null_preserves_stress_and_zeroes_physical_truth() -> None:
    base = SyntheticTuningSpec.synthetic_osse_null(20260811)
    spec = replace(
        base,
        native_domain=Domain(-180.0, 180.0, -90.0, 90.0, 12, 6),
        mode_domain=Domain(-180.0, 180.0, -90.0, 90.0, 6, 3),
        time_config=TimeConfig("2001-01-01", "2001-01-12", "1h"),
        split_windows=(
            ("basis_train", "2001-01-01", "2001-01-03"),
            ("bias_fit", "2001-01-04", "2001-01-06"),
            ("calibration", "2001-01-07", "2001-01-09"),
            ("development_test", "2001-01-10", "2001-01-12"),
        ),
    )

    bundle = generate_aerosol_tuning_bundle(spec)

    assert spec.sensor_bias_log == (0.015, -0.02)
    assert spec.error_temporal_correlation == 0.55
    assert spec.error_spatial_correlation == 0.6
    assert spec.mnar_cloud_strength == 0.8
    assert np.all(bundle.truth["clim_bias_raw_true"] == 0.0)
    assert np.all(bundle.truth["correction_pc_true"] == 0.0)
    assert np.all(bundle.truth["delta_requested_true"] == 0.0)
    assert np.all(bundle.truth["delta_filter_target_true"] == 0.0)
    assert np.any(bundle.truth["obs_error_log"] != 0.0)


def test_masked_chain_ci_has_locked_six_year_schedule_and_resource_shape() -> None:
    spec = SyntheticTuningSpec.masked_chain_ci(master_seed=8)

    assert (spec.native_domain.n_lat, spec.native_domain.n_lon) == (12, 24)
    assert (spec.mode_domain.n_lat, spec.mode_domain.n_lon) == (6, 12)
    assert spec.correction_periods_days == (20.0, 60.0, 120.0)
    assert all(
        spec.filter_band_days[0] < period < spec.filter_band_days[1]
        for period in spec.correction_periods_days
    )
    assert spec.split_windows == (
        ("basis_train", "2001-01-01", "2002-12-31"),
        ("bias_fit", "2003-01-01", "2004-12-31"),
        ("calibration", "2005-01-01", "2005-12-31"),
        ("development_test", "2006-01-01", "2006-12-31"),
    )
    bundle = generate_aerosol_tuning_bundle(spec)
    assert bundle.truth.sizes["time"] == 2191
    assert bundle.model["TOTEXTTAU"].dtype == np.float32
    assert bundle.model["TOTEXTTAU"].nbytes < 75 * 1024**2
    labels, counts = np.unique(bundle.truth["split"], return_counts=True)
    assert dict(zip(labels.tolist(), counts.tolist())) == {
        "basis_train": 730,
        "bias_fit": 731,
        "calibration": 365,
        "development_test": 365,
    }


def test_multi_sensor_ci_has_controlled_complementarity_and_precision() -> None:
    bundle = generate_aerosol_tuning_bundle(SyntheticTuningSpec.multi_sensor_ci())
    truth = bundle.truth
    valid = truth["valid_mask"].values.astype(bool)
    a_valid, b_valid = valid[:, 0]

    assert np.any(a_valid & ~b_valid)
    assert np.any(b_valid & ~a_valid)
    assert np.any(a_valid & b_valid)
    np.testing.assert_array_equal(
        bundle.observations["sensor_a"]["QA"], bundle.observations["sensor_a"]["qa_flag"]
    )
    assert np.count_nonzero(bundle.observations["sensor_a"]["QA"] == 3) == np.count_nonzero(
        valid[0]
    )
    sigma = truth["reported_sigma_log"].values[:, 0]
    overlap = a_valid & b_valid
    combined_sigma = 1.0 / np.sqrt(1.0 / sigma[0, overlap] ** 2 + 1.0 / sigma[1, overlap] ** 2)
    assert np.all(combined_sigma < sigma[0, overlap])
    assert np.all(combined_sigma < sigma[1, overlap])


def test_reported_sensor_covariance_separates_diagonal_and_shared_terms() -> None:
    spec = SyntheticTuningSpec(
        scenario="masked_chain_ci",
        sensor_error_sigma=(0.02, 0.04),
        common_error_sigma=0.015,
    )
    bundle = generate_aerosol_tuning_bundle(spec)
    truth = bundle.truth

    np.testing.assert_allclose(truth["reported_sigma_log"].isel(sensor=0), 0.02)
    np.testing.assert_allclose(truth["reported_sigma_log"].isel(sensor=1), 0.04)
    assert np.any(truth["common_error_log_true"].values != 0.0)
    for observation in bundle.observations.values():
        factor = observation["common_error_factor"]
        assert factor.dims == ("time", "common_mode", "lat", "lon")
        np.testing.assert_allclose(factor, 0.015)

    covariance = np.diag(np.square(spec.sensor_error_sigma))
    covariance += np.full((2, 2), spec.common_error_sigma**2)
    np.testing.assert_allclose(np.diag(covariance), [0.02**2 + 0.015**2, 0.04**2 + 0.015**2])
    assert covariance[0, 1] == spec.common_error_sigma**2

    common = np.asarray(truth["common_error_log_true"].values, dtype=np.float64)
    np.testing.assert_array_equal(
        common,
        np.broadcast_to(common[:, :1, :1], common.shape),
    )


def test_truth_support_uses_only_bias_fit_masks_and_declared_smoothing() -> None:
    spec = SyntheticTuningSpec(scenario="masked_chain_ci")
    days = pd.date_range(spec.time_config.start, spec.time_config.end, freq="1D")
    valid = np.zeros((2, len(days), 2, 4), dtype=bool)
    labels = split_labels(spec, days)
    bias_fit = labels == "bias_fit"
    valid[:, bias_fit, :, :] = True
    valid[:, bias_fit, 0, 1] = False
    valid[0, np.flatnonzero(bias_fit)[0], 0, 1] = True

    expected_support, expected_count = monthly_support(spec, days, valid)
    changed_future = valid.copy()
    changed_future[:, ~bias_fit] = True
    actual_support, actual_count = monthly_support(spec, days, changed_future)

    np.testing.assert_array_equal(actual_count, expected_count)
    np.testing.assert_array_equal(actual_support, expected_support)
    assert expected_count[0, 0, 0] == np.count_nonzero(bias_fit)
    np.testing.assert_allclose(
        expected_support[0],
        np.array(
            [
                [76.0 / 81.0, 49.0 / 54.0, 76.0 / 81.0, 76.0 / 81.0],
                [76.0 / 81.0, 49.0 / 54.0, 76.0 / 81.0, 76.0 / 81.0],
            ]
        ),
        rtol=0.0,
        atol=2.0e-16,
    )


def test_null_ci_has_zero_signal_bounded_gaps_and_unobservable_mode() -> None:
    bundle = generate_aerosol_tuning_bundle(SyntheticTuningSpec.null_ci())
    truth = bundle.truth

    assert np.all(truth["clim_bias_raw_true"].values == 0.0)
    assert np.all(truth["correction_pc_true"].values == 0.0)
    assert np.all(truth["delta_requested_true"].values == 0.0)
    assert int(truth["short_gap_day"].sum()) == bundle.spec.short_gap_days
    assert int(truth["long_gap_day"].sum()) == bundle.spec.long_gap_days
    gaps = (truth["short_gap_day"] | truth["long_gap_day"]).values.astype(bool)
    assert not truth["valid_mask"].values[:, gaps].any()

    assert truth["mode_observable_true"].values.tolist() == [1, 1, 0]
    valid = truth["valid_mask"].values.astype(bool)
    last_pattern = truth["pattern_true"].values[-1]
    assert np.all(np.where(valid, last_pattern[None, None, :, :], 0.0) == 0.0)
    assert np.any(last_pattern != 0.0)


def test_calibration_null_uses_six_year_schedule_and_frozen_policy() -> None:
    spec = SyntheticTuningSpec.calibration_null(master_seed=17)

    assert spec.scenario == "calibration_null"
    assert spec.time_config.start == "2001-01-01"
    assert spec.time_config.end == "2006-12-31"
    assert spec.filter_band_days == (4.0, 180.0)
    assert spec.filter_min_segment_days == 360

    bundle = generate_aerosol_tuning_bundle(spec)
    assert np.all(bundle.truth["correction_pc_true"] == 0.0)
    assert np.all(bundle.truth["delta_filter_target_true"] == 0.0)


def test_low_aod_ci_exercises_floor_clips_and_support_identity() -> None:
    truth = generate_aerosol_tuning_bundle(SyntheticTuningSpec.low_aod_ci()).truth
    model = truth["model_aod_overpass_true"].values
    clip = truth["clip_mask_true"].values
    ratio = truth["r_applied_true"].values

    assert np.any(model == 0.0)
    assert np.any((model > 0.0) & (model < 0.001))
    for bit in (1, 2, 4, 8):
        assert np.any((clip & bit) != 0)
    assert np.all(np.isfinite(ratio))
    assert float(np.min(ratio)) == 0.2
    assert float(np.max(ratio)) == 5.0
    assert np.all(ratio[(clip & (4 | 8)) != 0] == 1.0)


def test_fit_config_cannot_see_oracle_while_evaluation_config_can() -> None:
    root = Path(__file__).parents[4] / "analyses" / "aerosol-tuning" / "configs"
    fitting = (root / "fable-synthetic.example.yaml").read_text(encoding="utf-8")
    evaluation = (root / "fable-synthetic-eval.example.yaml").read_text(encoding="utf-8")

    assert "/oracle/" not in fitting
    assert "_true" not in fitting
    assert "${FABLE_SYNTH}/inputs/" in fitting
    assert "/oracle/truth.nc" in evaluation
    assert "delta_filter_target_true" in evaluation


def test_serialized_fit_config_loads_qa_masks_and_covariance(tmp_path: Path, monkeypatch) -> None:
    bundle = generate_aerosol_tuning_bundle(
        tmp_path, SyntheticTuningSpec(scenario="masked_chain_ci")
    )
    monkeypatch.setenv("FABLE_SYNTH", str(tmp_path))
    config_path = (
        Path(__file__).parents[4]
        / "analyses"
        / "aerosol-tuning"
        / "configs"
        / "fable-synthetic.example.yaml"
    )
    config = load_config(config_path)
    sensor_config = config.sources["sensor_a_raw"]
    assert sensor_config.type == "satellite_l3"
    assert getattr(sensor_config, "qa_variable") == "QA"
    assert getattr(sensor_config, "qa_values") == [3]
    projection_spec = config.analyses["obs_pcs"]
    assert isinstance(projection_spec, EOFProjectionSpec)
    assert [entry.source for entry in projection_spec.obs] == [
        "sensor_a_daily",
        "sensor_b_daily",
    ]

    context = PipelineContext(config=config)
    result = LoadSourcesStage().execute(context)
    assert result.status is StageStatus.COMPLETED
    loaded = context.sources["sensor_a_raw"].data
    invalid = ~bundle.truth["valid_mask"].sel(sensor="sensor_a").values.astype(bool)
    assert np.isnan(loaded["aod_550nm"].values[invalid]).all()
    assert np.all(loaded["QA"].values[invalid] == 0)

    target = xr.Dataset(coords={"lat": loaded["lat"], "lon": loaded["lon"]})
    preprocess_spec = config.analyses["sensor_a_daily"]
    assert isinstance(preprocess_spec, AODPreprocessSpec)
    processed = preprocess_aod(
        loaded,
        preprocess_spec,
        AnalysisRuntime(
            cast(datetime | None, config.analysis.start_time),
            cast(datetime | None, config.analysis.end_time),
            ArtifactService(tmp_path / "artifacts"),
        ),
        target_grid=target,
    )
    valid = processed["valid"].values
    np.testing.assert_allclose(processed["obs_error_std"].values[valid], 0.02)
    factors = processed["common_error_factor"].isel(common_mode=0).values
    np.testing.assert_allclose(factors[valid], 0.01)
    assert np.isnan(factors[~valid]).all()
