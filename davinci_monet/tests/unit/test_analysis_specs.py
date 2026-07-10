"""Analysis spec models parse and dispatch by type."""

from __future__ import annotations

import pytest

from davinci_monet.config.schema import (
    AODPreprocessSpec,
    AODScalingSpec,
    EOFProjectionSpec,
    EOFSpec,
    GriddedAnalysisSpec,
    KnownTruthSpec,
    MMRWriterSpec,
    PointReduce,
    WaveletFilterSpec,
    WaveletSpec,
    build_analysis_spec,
)


def test_build_eof_spec() -> None:
    spec = build_analysis_spec({"type": "eof", "source": "cam", "variable": "O3", "n_modes": 6})
    assert isinstance(spec, EOFSpec)
    assert spec.n_modes == 6
    assert spec.standardize is False
    assert spec.rotation == "none"
    assert spec.required is False
    assert spec.input_refs() == {"source": "cam"}


def test_build_wavelet_spec_with_point_reduce() -> None:
    spec = build_analysis_spec(
        {"type": "wavelet", "source": "cam", "variable": "O3", "reduce": {"point": [40.0, -105.0]}}
    )
    assert isinstance(spec, WaveletSpec)
    assert isinstance(spec.reduce, PointReduce)
    assert spec.reduce.point == (40.0, -105.0)


def test_wavelet_default_reduce_is_area_mean() -> None:
    spec = build_analysis_spec(
        {"type": "wavelet", "source": "cam", "variable": "O3", "required": True}
    )
    assert isinstance(spec, WaveletSpec)
    assert spec.reduce == "area_mean"
    assert spec.required is True
    assert spec.input_refs() == {"source": "cam"}


def test_gridded_spec_exposes_legacy_source_as_named_input() -> None:
    spec = build_analysis_spec(
        {
            "type": "gridded_analysis",
            "source": "cam",
            "roles": {"analysis": "O3"},
            "fields": {"mean": {"formula": 'mean(analysis, dim="time")'}},
        }
    )

    assert isinstance(spec, GriddedAnalysisSpec)
    assert spec.input_refs() == {"source": "cam"}


def test_aod_preprocess_spec_exposes_optional_target_dependency() -> None:
    spec = build_analysis_spec(
        {
            "type": "aod_preprocess",
            "source": "sensor_raw",
            "variable": "aod_550nm",
            "target_grid_from": "model_daily",
            "log_epsilon": 0.01,
            "uncertainty_variable": "sigma",
            "uncertainty_covariance": "independent",
            "common_factor_variables": ["sensor_common"],
            "required": True,
        }
    )

    assert isinstance(spec, AODPreprocessSpec)
    assert spec.input_refs() == {
        "source": "sensor_raw",
        "target_grid_from": "model_daily",
    }
    assert spec.required is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_local_time", 24.0, "hour values"),
        ("day_anchor_hour", -1.0, "hour values"),
        ("sample_tolerance", "-1h", "non-negative"),
        ("target_grid", 7.0, "divide 180"),
    ],
)
def test_aod_preprocess_rejects_invalid_numeric_contracts(
    field: str, value: object, message: str
) -> None:
    config = {
        "type": "aod_preprocess",
        "source": "model",
        "variable": "aod",
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        build_analysis_spec(config)


def test_aod_preprocess_rejects_two_target_grid_policies() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_analysis_spec(
            {
                "type": "aod_preprocess",
                "source": "model",
                "variable": "aod",
                "target_grid": 30.0,
                "target_grid_from": "other",
            }
        )


def test_eof_projection_spec_exposes_all_named_inputs() -> None:
    spec = build_analysis_spec(
        {
            "type": "eof_projection",
            "basis": "basis",
            "model": "model_daily",
            "model_variable": "log_aod",
            "obs": [
                {
                    "source": "sensor_a",
                    "variable": "log_aod",
                    "error_variable": "obs_error_std",
                },
                {
                    "source": "sensor_b",
                    "variable": "log_aod",
                    "error_variable": "obs_error_std",
                    "common_factor_variables": ["common_error_factor"],
                },
            ],
            "bias_fit_artifact": "saved_bias",
        }
    )

    assert isinstance(spec, EOFProjectionSpec)
    assert spec.input_refs() == {
        "basis": "basis",
        "model": "model_daily",
        "obs[0]": "sensor_a",
        "obs[1]": "sensor_b",
        "bias_fit_artifact": "saved_bias",
    }


def test_eof_projection_rejects_invalid_fit_and_support_contracts() -> None:
    base = {
        "type": "eof_projection",
        "basis": "basis",
        "model": "model",
        "model_variable": "log_aod",
        "obs": [{"source": "obs", "variable": "log_aod", "error_variable": "sigma"}],
    }
    with pytest.raises(ValueError, match="requires bias_fit_window"):
        build_analysis_spec(base)
    with pytest.raises(ValueError, match="support_min_fraction"):
        build_analysis_spec(
            {
                **base,
                "bias_fit_window": {"start": "2001-01-01", "end": "2001-12-31"},
                "support_min_fraction": 0.6,
                "support_full_fraction": 0.5,
            }
        )


def test_wavelet_filter_spec_enforces_segment_context() -> None:
    spec = build_analysis_spec(
        {
            "type": "wavelet_filter",
            "source": "projection",
            "band": {"min": 4.0, "max": 32.0, "units": "days"},
            "min_segment_days": 64.0,
        }
    )
    assert isinstance(spec, WaveletFilterSpec)
    assert spec.input_refs() == {"source": "projection"}

    with pytest.raises(ValueError, match="at least twice"):
        build_analysis_spec(
            {
                "type": "wavelet_filter",
                "source": "projection",
                "band": {"min": 4.0, "max": 32.0},
                "min_segment_days": 63.0,
            }
        )


def test_aod_scaling_spec_exposes_named_inputs_and_validates_bounds() -> None:
    spec = build_analysis_spec(
        {
            "type": "aod_scaling",
            "basis": "basis",
            "projection": "projection",
            "coefficients": "filtered",
            "model": "model_daily",
            "r_bounds": [0.25, 4.0],
        }
    )
    assert isinstance(spec, AODScalingSpec)
    assert spec.input_refs() == {
        "basis": "basis",
        "projection": "projection",
        "coefficients": "filtered",
        "model": "model_daily",
    }

    with pytest.raises(ValueError, match="r_bounds"):
        build_analysis_spec(
            {
                "type": "aod_scaling",
                "basis": "basis",
                "projection": "projection",
                "coefficients": "filtered",
                "model": "model_daily",
                "r_bounds": [1.0, 1.0],
            }
        )


def test_mmr_writer_is_always_required_and_validates_species() -> None:
    spec = build_analysis_spec(
        {
            "type": "mmr_writer",
            "scaling": "scaling",
            "files": "/synthetic/mmr/*.nc4",
            "output_dir": "/synthetic/corrected",
        }
    )
    assert isinstance(spec, MMRWriterSpec)
    assert spec.required is True
    assert spec.input_refs() == {"scaling": "scaling"}

    with pytest.raises(ValueError, match="species names must be unique"):
        build_analysis_spec(
            {
                "type": "mmr_writer",
                "scaling": "scaling",
                "files": "/synthetic/mmr/*.nc4",
                "output_dir": "/synthetic/corrected",
                "species": ["SO4", "SO4"],
            }
        )


def test_known_truth_spec_is_read_only_named_evaluation() -> None:
    spec = build_analysis_spec(
        {
            "type": "known_truth",
            "estimate": "scaling",
            "truth": "oracle",
            "evaluation_splits": ["development_test", "acceptance"],
        }
    )
    assert isinstance(spec, KnownTruthSpec)
    assert spec.required is True
    assert spec.input_refs() == {"estimate": "scaling", "truth": "oracle"}

    with pytest.raises(ValueError, match="evaluation_splits must be unique"):
        build_analysis_spec(
            {
                "type": "known_truth",
                "estimate": "scaling",
                "truth": "oracle",
                "evaluation_splits": ["test", "test"],
            }
        )
    with pytest.raises(ValueError, match="split_variable is required"):
        build_analysis_spec(
            {
                "type": "known_truth",
                "estimate": "scaling",
                "truth": "oracle",
                "split_variable": None,
            }
        )
    with pytest.raises(ValueError, match="Input should be True"):
        build_analysis_spec(
            {
                "type": "mmr_writer",
                "scaling": "scaling",
                "files": "/synthetic/mmr/*.nc4",
                "output_dir": "/synthetic/corrected",
                "required": False,
            }
        )


def test_unknown_analysis_type_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown analysis type"):
        build_analysis_spec({"type": "bogus", "source": "cam", "variable": "O3"})
