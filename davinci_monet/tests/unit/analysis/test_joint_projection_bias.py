"""Focused production-contract tests for the joint projection-bias fit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.analysis.artifacts import ArtifactService
from davinci_monet.analysis.base import ArtifactDeclaration
from davinci_monet.analysis.projection import project_eof
from davinci_monet.analysis.projection_batches import fit_monthly_bias_batched
from davinci_monet.analysis.projection_inputs import prepare_projection_inputs
from davinci_monet.analysis.projection_joint import _seasonal_gauge
from davinci_monet.analysis.projection_joint_core import seasonal_design
from davinci_monet.config.schema import EOFProjectionSpec


@dataclass(frozen=True)
class JointCase:
    basis: xr.Dataset
    model: xr.Dataset
    sensors: dict[str, xr.Dataset]
    seasonal: np.ndarray
    anomaly: np.ndarray
    temporal_design: np.ndarray


def _case(*, disconnected: bool = False) -> JointCase:
    time = pd.date_range("2001-01-01T12:00:00", periods=365, freq="1D")
    lat = np.array([0.0])
    lon = np.array([-90.0, 90.0])
    pattern = np.array([[[1.0, -1.0]]])
    basis = xr.Dataset(
        {"eofs": (("mode", "lat", "lon"), pattern)},
        coords={"mode": [1], "lat": lat, "lon": lon},
        attrs={
            "eof_rotation": "none",
            "eof_standardize": "false",
            "eof_input_log_epsilon": 0.01,
        },
    )
    model = xr.Dataset(
        {"log_aod": (("time", "lat", "lon"), np.zeros((time.size, 1, 2)))},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"log_epsilon": 0.01},
    )
    month = time.month.to_numpy()
    angle = 2.0 * np.pi * (month - 0.5) / 12.0
    temporal_design = np.column_stack((np.ones(time.size), np.sin(angle), np.cos(angle)))
    seasonal = 0.08 + 0.03 * temporal_design[:, 1] - 0.02 * temporal_design[:, 2]
    anomaly = 0.04 * np.sin(2.0 * np.pi * np.arange(time.size) / 20.0)
    anomaly -= temporal_design @ np.linalg.lstsq(temporal_design, anomaly, rcond=None)[0]
    common = np.array([0.2, 0.2])[None, None, :]
    signal = common + (seasonal + anomaly)[:, None, None] * pattern

    sensors: dict[str, xr.Dataset] = {}
    for index, (name, offset) in enumerate((("sensor_a", -0.03), ("sensor_b", 0.03))):
        valid = np.ones(signal.shape, dtype=bool)
        if disconnected:
            valid[..., 1 - index] = False
        values = np.where(valid, signal + offset, np.nan)
        error = xr.DataArray(
            np.full(signal.shape, 0.001),
            dims=("time", "lat", "lon"),
            attrs={"units": "1", "space": "shifted_log"},
        )
        sensors[name] = xr.Dataset(
            {
                "log_aod": (("time", "lat", "lon"), values),
                "obs_error_std": error,
                "valid": (("time", "lat", "lon"), valid),
            },
            coords={"time": time, "lat": lat, "lon": lon},
            attrs={"log_epsilon": 0.01},
        )
    return JointCase(basis, model, sensors, seasonal, anomaly, temporal_design)


def _spec(sensor_names: tuple[str, ...], *, artifact: bool = False) -> EOFProjectionSpec:
    fit_source: dict[str, object]
    if artifact:
        fit_source = {"bias_fit_artifact": "saved_projection"}
    else:
        fit_source = {
            "bias_fit_window": {
                "start": "2001-01-01",
                "end": "2001-12-31T23:59:59",
            }
        }
    return EOFProjectionSpec.model_validate(
        {
            "type": "eof_projection",
            "basis": "basis",
            "model": "model",
            "model_variable": "log_aod",
            "obs": [
                {
                    "source": name,
                    "variable": "log_aod",
                    "error_variable": "obs_error_std",
                }
                for name in sensor_names
            ],
            "ridge": 1.0,
            "bias_fit_method": "joint_seasonal",
            "sensor_offset_method": "overlap_zero_sum",
            "support_min_fraction": 0.2,
            "support_full_fraction": 0.5,
            "support_smoothing_passes": 0,
            "delta_bounds": (-2.0, 2.0),
            "joint_bias_laplacian_strength": 1.0,
            "joint_bias_tolerance": 1.0e-6,
            "joint_bias_max_iterations": 20,
            "time_chunk_size": 31,
            **fit_source,
        }
    )


def _run(case: JointCase, names: tuple[str, ...]) -> xr.Dataset:
    spec = _spec(names)
    return project_eof(
        case.basis,
        case.model,
        [(entry, case.sensors[entry.source]) for entry in spec.obs],
        spec,
    )


@pytest.fixture(scope="module")
def fitted_case() -> tuple[JointCase, xr.Dataset]:
    case = _case()
    return case, _run(case, ("sensor_a", "sensor_b"))


def test_joint_policy_schema_is_explicit_and_rejects_sequential_offsets() -> None:
    spec = _spec(("sensor_a", "sensor_b"))

    assert spec.bias_fit_method == "joint_seasonal"
    assert spec.sensor_offset_method == "overlap_zero_sum"
    assert spec.joint_bias_laplacian_strength == 1.0
    assert spec.joint_bias_tolerance == 1.0e-6
    assert spec.joint_bias_max_iterations == 20

    values = spec.model_dump(mode="python")
    values["bias_fit_method"] = "monthly_mean"
    with pytest.raises(ValueError, match="requires bias_fit_method='joint_seasonal'"):
        EOFProjectionSpec.model_validate(values)


def test_temporal_gauge_uses_ridge_precision_and_ignores_unobserved_days() -> None:
    design = np.repeat(seasonal_design(), 2, axis=0)
    observed = np.ones(design.shape[0], dtype=bool)
    observed[[1, 8, 17]] = False
    coefficients = np.column_stack(
        (np.linspace(-0.2, 0.3, design.shape[0]), np.cos(np.arange(design.shape[0])))
    )
    coefficients[~observed] = 1.0e6

    theta = _seasonal_gauge(design, coefficients, observed, ridge=1.0)
    anomaly = coefficients - design @ theta
    weighted_inner = design.T @ (anomaly * observed[:, None])

    np.testing.assert_allclose(weighted_inner, 0.0, atol=2.0e-15)
    changed = coefficients.copy()
    changed[~observed] = -1.0e12
    np.testing.assert_allclose(
        _seasonal_gauge(design, changed, observed, ridge=1.0), theta, atol=0.0
    )


def test_joint_support_is_independent_of_innovation_values() -> None:
    case = _case()
    spec = _spec(("sensor_a", "sensor_b"))

    def support_fit(sensors: dict[str, xr.Dataset]):
        prepared = prepare_projection_inputs(
            case.basis,
            case.model,
            [(entry, sensors[entry.source]) for entry in spec.obs],
            spec.model_variable,
        )
        return fit_monthly_bias_batched(
            prepared.model,
            prepared.observations,
            prepared.time.month.to_numpy(dtype=np.int64),
            np.ones(prepared.time.size, dtype=bool),
            support_min_fraction=spec.support_min_fraction,
            support_full_fraction=spec.support_full_fraction,
            smoothing_passes=spec.support_smoothing_passes,
            delta_bounds=spec.delta_bounds,
            time_chunk_size=spec.time_chunk_size,
        )

    original = support_fit(case.sensors)
    changed = {name: dataset.copy(deep=True) for name, dataset in case.sensors.items()}
    for index, dataset in enumerate(changed.values(), start=1):
        dataset["log_aod"] = xr.where(
            dataset["valid"], dataset["log_aod"] * (10.0 * index) - index, np.nan
        )
    mutated = support_fit(changed)

    for field in ("support", "support_fraction", "support_count", "support_day_total"):
        np.testing.assert_array_equal(getattr(mutated, field), getattr(original, field))


def test_joint_fit_recovers_identifiable_decomposition_and_gauges(
    fitted_case: tuple[JointCase, xr.Dataset],
) -> None:
    case, output = fitted_case

    np.testing.assert_allclose(output["clim_bias_perpendicular"], 0.2, atol=2.0e-8)
    np.testing.assert_allclose(
        output["clim_bias_mode_coefficient"].values[:, 0],
        case.seasonal[[14 + 30 * month for month in range(12)]],
        atol=2.0e-8,
    )
    np.testing.assert_allclose(output["sensor_offset"], [-0.03, 0.03], atol=2.0e-8)
    np.testing.assert_allclose(output["pc"].values[:, 0], case.anomaly, atol=3.0e-8)

    pattern = case.basis["eofs"].values[0]
    perpendicular_inner = np.sum(
        output["clim_bias_perpendicular"].values * pattern[None, :, :], axis=(1, 2)
    )
    np.testing.assert_allclose(perpendicular_inner, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(output["sensor_offset"].sum(), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(case.temporal_design.T @ output["pc"].values[:, 0], 0.0, atol=2.0e-7)
    assert np.all(np.diff(output["joint_objective"].values) <= 1.0e-8)
    assert output.attrs["projection_joint_bias_converged"] == "true"
    np.testing.assert_array_equal(output["pooled_observable_rank"], np.ones(12))
    np.testing.assert_array_equal(output["sensor_overlap_count"], np.full((2, 2), 365))


def test_joint_fit_is_sensor_order_invariant(
    fitted_case: tuple[JointCase, xr.Dataset],
) -> None:
    case, expected = fitted_case
    reordered = _run(case, ("sensor_b", "sensor_a"))

    np.testing.assert_allclose(reordered["clim_bias"], expected["clim_bias"], atol=2.0e-10)
    np.testing.assert_allclose(reordered["pc"], expected["pc"], atol=2.0e-10)
    np.testing.assert_allclose(
        reordered["sensor_offset"].sel(sensor=["sensor_a", "sensor_b"]),
        expected["sensor_offset"],
        atol=2.0e-10,
    )


def test_joint_offset_fit_rejects_disconnected_sensor_overlap() -> None:
    case = _case(disconnected=True)

    with pytest.raises(ValueError, match="connected pairwise-overlap graph"):
        _run(case, ("sensor_a", "sensor_b"))


def test_joint_offset_fit_uses_zero_for_one_sensor() -> None:
    case = _case()

    output = _run(case, ("sensor_a",))

    np.testing.assert_array_equal(output["sensor_offset"], [0.0])
    assert output.attrs["projection_absolute_sensor_offset_identifiable"] == "false"


def test_joint_projection_fit_round_trips_through_saved_artifact(
    fitted_case: tuple[JointCase, xr.Dataset], tmp_path: Path
) -> None:
    case, fitted = fitted_case
    persisted = (
        ArtifactService(tmp_path)
        .materialize(
            "saved_projection",
            fitted,
            (ArtifactDeclaration(kind="product", reload=True),),
        )
        .dataset
    )
    spec = _spec(("sensor_a", "sensor_b"), artifact=True)

    reused = project_eof(
        case.basis,
        case.model,
        [(entry, case.sensors[entry.source]) for entry in spec.obs],
        spec,
        bias_fit_artifact=persisted,
    )

    for variable in (
        "clim_bias",
        "clim_bias_perpendicular",
        "clim_bias_mode_coefficient",
        "sensor_offset",
        "pc",
    ):
        np.testing.assert_allclose(reused[variable], fitted[variable], atol=1.0e-7)
    assert reused.attrs["projection_bias_fit_selection"] == "artifact"
    assert (
        reused.attrs["projection_bias_fit_policy_signature"]
        == fitted.attrs["projection_bias_fit_policy_signature"]
    )

    ambiguous_spec = spec.model_dump()
    ambiguous_spec["bias_fit_window"] = {
        "start": "2001-01-01",
        "end": "2001-12-31T23:59:59",
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        project_eof(
            case.basis,
            case.model,
            [(entry, case.sensors[entry.source]) for entry in spec.obs],
            ambiguous_spec,
            bias_fit_artifact=persisted,
        )

    changed_basis = case.basis.copy(deep=True)
    changed_basis["eofs"] = changed_basis["eofs"] * 1.01
    with pytest.raises(ValueError, match="basis signature"):
        project_eof(
            changed_basis,
            case.model,
            [(entry, case.sensors[entry.source]) for entry in spec.obs],
            spec,
            bias_fit_artifact=persisted,
        )

    mismatch_attrs = (
        ("projection_grid_signature", "0" * 64, "grid signature"),
        ("projection_bias_fit_start", "2001-02-01", "fit-window signature"),
        ("projection_bias_fit_policy_signature", "0" * 64, "joint policy signature"),
    )
    for attribute, value, message in mismatch_attrs:
        mismatched = persisted.copy(deep=True)
        mismatched.attrs[attribute] = value
        with pytest.raises(ValueError, match=message):
            project_eof(
                case.basis,
                case.model,
                [(entry, case.sensors[entry.source]) for entry in spec.obs],
                spec,
                bias_fit_artifact=mismatched,
            )
