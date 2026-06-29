from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.analysis.gridded import GriddedAnalysis
from davinci_monet.config.schema import GriddedAnalysisSpec


def _cam_ds() -> xr.Dataset:
    time = np.array(
        ["2008-07-01T00:00", "2008-07-01T03:00", "2008-07-02T00:00"],
        dtype="datetime64[ns]",
    )
    lat = np.array([-1.0, 1.0])
    lon = np.array([0.0, 90.0])
    obs = np.array(
        [
            [[0.2, -999.0], [0.3, -999.0]],
            [[0.4, 0.5], [0.6, -999.0]],
            [[0.8, 0.9], [1.0, 1.1]],
        ]
    )
    pre = np.array(
        [
            [[0.1, 0.1], [0.2, 0.2]],
            [[0.2, 0.2], [0.3, 0.3]],
            [[0.4, 0.4], [0.5, 0.5]],
        ]
    )
    post = np.array(
        [
            [[0.18, 0.1], [0.28, 0.2]],
            [[0.36, 0.46], [0.58, 0.3]],
            [[0.7, 0.8], [0.9, 1.0]],
        ]
    )
    mask = np.array([[[1, 0], [1, 0]], [[1, 1], [1, 0]], [[0, 1], [1, 1]]], dtype=float)
    return xr.Dataset(
        {
            "AODNDG_OBS": (("time", "lat", "lon"), obs),
            "AODNDG_MODEL_PRE": (("time", "lat", "lon"), pre),
            "AODNDG_MODEL_POST": (("time", "lat", "lon"), post),
            "AODNDG_MASK": (("time", "lat", "lon"), mask),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"geometry": "grid", "source_label": "cam"},
    )


def _spec() -> GriddedAnalysisSpec:
    return GriddedAnalysisSpec(
        type="gridded_analysis",
        source="cam",
        groupby="day",
        roles={
            "observation": "AODNDG_OBS",
            "first_guess": "AODNDG_MODEL_PRE",
            "analysis": "AODNDG_MODEL_POST",
            "mask": "AODNDG_MASK",
        },
        fields={
            "analyzed_aod": {"formula": 'mean(analysis, dim="time")'},
            "analysis_increment_aod": {
                "formula": 'active_mean(analysis - first_guess, mask > 0.5, dim="time")'
            },
            "increment_x2": {"formula": "analysis_increment_aod * 2"},
            "nudge_fraction": {"formula": 'mean(mask, dim="time")'},
        },
    )


def test_gridded_analysis_builds_grouped_fields() -> None:
    out = GriddedAnalysis().analyze(_cam_ds(), _spec())
    assert out.attrs["geometry"] == "grid"
    assert out.attrs["analysis_type"] == "gridded_analysis"
    assert out.attrs["groupby"] == "day"
    assert "group" in out.dims
    assert "analyzed_aod" in out
    assert "analysis_increment_aod" in out
    assert "increment_x2" in out
    assert "nudge_fraction" in out
    assert out["group"].values.tolist() == ["2008-07-01", "2008-07-02"]
    assert out["analyzed_aod"].dims == ("group", "lat", "lon")
    np.testing.assert_allclose(out["nudge_fraction"].values[0], [[1.0, 0.5], [1.0, 0.0]])
    np.testing.assert_allclose(out["increment_x2"], out["analysis_increment_aod"] * 2)
    assert "source_label" not in out.attrs


def test_gridded_analysis_rejects_coordinate_roles() -> None:
    spec = _spec()
    spec.roles["analysis"] = "time"
    try:
        GriddedAnalysis().analyze(_cam_ds(), spec)
    except ValueError as exc:
        assert "time" in str(exc)
    else:
        raise AssertionError("coordinate role should fail")


def test_gridded_analysis_uses_output_group_as_source_label() -> None:
    spec = _spec()
    spec.output_group = "daily_cam_aod"
    out = GriddedAnalysis().analyze(_cam_ds(), spec)
    assert out.attrs["source_label"] == "daily_cam_aod"


def test_gridded_analysis_names_formula_failures() -> None:
    spec = _spec()
    spec.fields["bad"] = type(spec.fields["analyzed_aod"])(formula="unknown + 1")
    try:
        GriddedAnalysis().analyze(_cam_ds(), spec)
    except ValueError as exc:
        assert "bad" in str(exc)
        assert "unknown" in str(exc)
    else:
        raise AssertionError("bad formula should fail")
