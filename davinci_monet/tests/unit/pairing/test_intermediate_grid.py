"""IntermediateGridStrategy symmetric binning (method: grid, 2-D)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy


def _point_ds(lats, lons, vals, var, t="2024-02-01"):
    n = len(lats)
    time = pd.to_datetime([t] * n)
    return xr.Dataset(
        {var: (["site"], np.asarray(vals, float), {"units": "1"})},
        coords={
            "site": np.arange(n),
            "time": ("site", time),
            "latitude": ("site", np.asarray(lats, float)),
            "longitude": ("site", np.asarray(lons, float)),
        },
    )


def test_symmetric_bins_both_point_sources_cell_means():
    # Two points fall in the SAME 1-degree cell -> their mean; one alone elsewhere.
    x = _point_ds([10.2, 10.7, 40.5], [20.2, 20.6, 50.5], [1.0, 3.0, 9.0], "aod")
    y = _point_ds([10.4, 40.4], [20.4, 50.4], [2.0, 8.0], "AOD")
    paired = IntermediateGridStrategy().pair_sources(
        x_data=x,
        y_data=y,
        x_var="aod",
        y_var="AOD",
        x_source="obs",
        y_source="mod",
        horizontal_res=1.0,
        time_resolution="1D",
        min_sample_count=1,
    )
    assert list(paired["x_aod"].dims) == ["time", "lon", "lat"]
    assert "y_AOD" in paired and "x_sample_count" in paired and "y_sample_count" in paired
    xa = paired["x_aod"].squeeze("time", drop=True)
    ya = paired["y_AOD"].squeeze("time", drop=True)
    # the cell containing (~10.x, ~20.x): x mean = (1+3)/2 = 2.0, y = 2.0, counts 2 and 1
    cell_x = xa.sel(lat=10.5, lon=20.5, method="nearest").item()
    cell_y = ya.sel(lat=10.5, lon=20.5, method="nearest").item()
    assert cell_x == pytest.approx(2.0)
    assert cell_y == pytest.approx(2.0)
    assert paired["x_sample_count"].sel(lat=10.5, lon=20.5, method="nearest").max().item() == 2
    # tagged for downstream stats/plots
    assert paired["x_aod"].attrs["axis"] == "x" and paired["x_aod"].attrs["source_label"] == "obs"
    assert paired["y_AOD"].attrs["axis"] == "y" and paired["y_AOD"].attrs["source_label"] == "mod"


def test_match_dataset_binning_handles_descending_y_latitude():
    x = _point_ds([40.0], [-100.0], [5.0], "aod")
    y = xr.Dataset(
        {"AOD": (["time", "lat", "lon"], np.ones((1, 3, 1), dtype=float))},
        coords={
            "time": pd.to_datetime(["2024-02-01"]),
            "lat": [50.0, 40.0, 30.0],
            "lon": [-100.0],
        },
    )

    paired = IntermediateGridStrategy().pair_sources(
        x_data=x,
        y_data=y,
        x_var="aod",
        y_var="AOD",
        grid_mode="match_dataset",
        time_resolution="1D",
        min_sample_count=1,
    )

    assert int(paired["sample_count"].sum().item()) == 1
    assert np.isfinite(paired["x_aod"]).any()
    assert np.isfinite(paired["y_AOD"]).any()
    np.testing.assert_allclose(paired["lat"].values, [30.0, 40.0, 50.0])


def test_symmetric_auto_extent_normalizes_mixed_longitude_conventions():
    x = _point_ds([35.0], [-100.0], [1.0], "aod")
    y = _point_ds([35.0], [260.0], [2.0], "AOD")

    paired = IntermediateGridStrategy().pair_sources(
        x_data=x,
        y_data=y,
        x_var="aod",
        y_var="AOD",
        x_source="obs",
        y_source="mod",
        horizontal_res=1.0,
        time_resolution="1D",
        min_sample_count=1,
    )

    assert paired.sizes["lon"] == 1
    assert int(paired["x_sample_count"].sum().item()) == 1
    assert int(paired["y_sample_count"].sum().item()) == 1


def test_small_span_auto_extent_keeps_all_points():
    # Regression: data span (~0.9 deg) smaller than horizontal_res (2.0) with the
    # default (auto) extent must NOT silently drop edge points — the grid must
    # still span the full data extent.
    x = _point_ds(
        [30.0, 30.3, 30.6, 30.9], [-110.0, -110.3, -110.6, -110.9], [1.0, 1.0, 1.0, 1.0], "aod"
    )
    y = _point_ds([30.4, 30.5], [-110.4, -110.5], [2.0, 2.0], "AOD")
    paired = IntermediateGridStrategy().pair_sources(
        x_data=x,
        y_data=y,
        x_var="aod",
        y_var="AOD",
        x_source="obs",
        y_source="mod",
        horizontal_res=2.0,
        time_resolution="1D",
        min_sample_count=1,
    )
    assert int(paired["x_sample_count"].sum().item()) == 4  # all 4 x points retained
    assert int(paired["y_sample_count"].sum().item()) == 2  # all 2 y points retained


def test_min_sample_count_masks_sparse_cells():
    x = _point_ds([10.2, 10.7], [20.2, 20.6], [1.0, 3.0], "aod")  # 2 in one cell
    y = _point_ds([10.4], [20.4], [2.0], "AOD")  # 1 in that cell
    paired = IntermediateGridStrategy().pair_sources(
        x_data=x,
        y_data=y,
        x_var="aod",
        y_var="AOD",
        x_source="obs",
        y_source="mod",
        horizontal_res=1.0,
        time_resolution="1D",
        min_sample_count=2,
    )
    # y has only 1 sample in the cell -> masked to NaN under min_sample_count=2
    ya = paired["y_AOD"].squeeze("time", drop=True)
    assert np.isnan(ya.sel(lat=10.5, lon=20.5, method="nearest").item())


def test_engine_routes_method_grid_to_symmetric():
    from davinci_monet.pairing.engine import PairingEngine

    x = _point_ds([10.2, 10.7], [20.2, 20.6], [1.0, 3.0], "aod")
    y = _point_ds([10.4], [20.4], [2.0], "AOD")
    paired = PairingEngine().pair_sources(
        x_data=x,
        y_data=y,
        x_vars=["aod"],
        y_vars=["AOD"],
        x_source="obs",
        y_source="mod",
        method="grid",
        horizontal_res=1.0,
        time_resolution="1D",
        min_sample_count=1,
    )
    data = getattr(paired, "data", paired)
    assert isinstance(data, xr.Dataset)
    # The engine relabels grid output to public <source_label>_<var> names.
    assert "obs_aod" in data and "mod_AOD" in data
    assert list(data["obs_aod"].dims) == ["time", "lon", "lat"]


@pytest.mark.integration
def test_method_grid_runs_through_pipeline(tmp_path):
    from davinci_monet.pipeline.runner import PipelineRunner

    x = _point_ds([10.2, 10.7, 40.5], [20.2, 20.6, 50.5], [1.0, 3.0, 9.0], "aod")
    y = _point_ds([10.4, 40.4], [20.4, 50.4], [2.0, 8.0], "AOD")
    xp, yp = tmp_path / "x.nc", tmp_path / "y.nc"
    x.to_netcdf(xp)
    y.to_netcdf(yp)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "obs": {"type": "generic", "files": str(xp), "variables": {"aod": {"units": "1"}}},
            "mod": {"type": "generic", "files": str(yp), "variables": {"AOD": {"units": "1"}}},
        },
        "pairs": {
            "obs_vs_mod": {
                "x": {"source": "obs", "variable": "aod"},
                "y": {"source": "mod", "variable": "AOD"},
                "method": "grid",
                "grid": {"horizontal_res": 1.0, "time_resolution": "1D", "min_sample_count": 1},
            }
        },
        "plots": {"sc": {"type": "scatter", "pairs": ["obs_vs_mod"]}},
    }
    result = PipelineRunner(show_progress=False).run_from_config(config)
    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None
    assert "obs_vs_mod" in ctx.paired


def test_pressure_to_altitude_standard_atmosphere():
    import numpy as np

    from davinci_monet.pairing.strategies.track import pressure_to_altitude

    p = np.array([1013.25, 500.0, 700.0])
    z = pressure_to_altitude(p)
    assert z[0] == pytest.approx(0.0, abs=1.0)  # sea level
    assert z[1] == pytest.approx(5572.0, abs=50.0)  # ~500 hPa
    assert z[2] == pytest.approx(3012.0, abs=50.0)  # ~700 hPa
    # round-trips with the existing forward conversion
    from davinci_monet.pairing.strategies.track import altitude_to_pressure

    assert altitude_to_pressure(z)[1] == pytest.approx(500.0, rel=1e-3)


def _track_alt_ds(alts_m, var="O3"):
    import numpy as np
    import pandas as pd
    import xarray as xr

    n = len(alts_m)
    return xr.Dataset(
        {var: (["time"], np.arange(n, dtype=float))},
        coords={
            "time": pd.to_datetime(["2012-05-29"] * n),
            "latitude": ("time", np.full(n, 35.0)),
            "longitude": ("time", np.full(n, -97.0)),
            "altitude": ("time", np.asarray(alts_m, float), {"units": "m"}),
        },
    )


def test_source_altitude_native_meters():
    from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy

    ds = _track_alt_ds([500.0, 8000.0])
    alt = IntermediateGridStrategy()._source_altitude(ds, "O3", "m")
    assert list(alt.dims) == ["time"]
    assert float(alt.values[0]) == pytest.approx(500.0)
    assert float(alt.values[1]) == pytest.approx(8000.0)


def test_source_altitude_native_km_units_conversion():
    from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy

    ds = _track_alt_ds([500.0, 8000.0])  # source in metres
    alt = IntermediateGridStrategy()._source_altitude(ds, "O3", "km")  # request km
    assert float(alt.values[1]) == pytest.approx(8.0)


def test_source_altitude_pressure_fallback():
    import numpy as np
    import pandas as pd
    import xarray as xr

    from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy

    ds = xr.Dataset(
        {"O3": (["time", "lev"], np.zeros((1, 2)))},
        coords={
            "time": pd.to_datetime(["2012-05-29"]),
            "lev": ("lev", np.array([1013.25, 500.0]), {"units": "hPa"}),
            "latitude": ("time", [35.0]),
            "longitude": ("time", [-97.0]),
        },
    )
    alt = IntermediateGridStrategy()._source_altitude(ds, "O3", "m")
    # broadcast to (time, lev): lev 1013->~0 m, 500->~5572 m
    vals = alt.transpose("time", "lev").values[0]
    assert vals[0] == pytest.approx(0.0, abs=1.0)
    assert vals[1] == pytest.approx(5572.0, abs=50.0)


def test_source_altitude_errors_without_vertical():
    import numpy as np
    import pandas as pd
    import xarray as xr

    from davinci_monet.core.exceptions import PairingError
    from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy

    # hybrid 'z' level, no length units, no geopotential, no pressure
    ds = xr.Dataset(
        {"O3": (["time", "z"], np.zeros((1, 2)))},
        coords={
            "time": pd.to_datetime(["2012-05-29"]),
            "z": ("z", np.array([0.5, 0.9])),  # hybrid, unitless
            "latitude": ("time", [35.0]),
            "longitude": ("time", [-97.0]),
        },
    )
    with pytest.raises(PairingError, match="vertical"):
        IntermediateGridStrategy()._source_altitude(ds, "O3", "m")


def test_symmetric_3d_bins_by_altitude():
    import numpy as np
    import pandas as pd
    import xarray as xr

    from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy

    # x: a track with native altitude (m); two points at ~500 m, one at ~6000 m
    x = xr.Dataset(
        {"O3": (["time"], np.array([10.0, 30.0, 99.0]))},
        coords={
            "time": pd.to_datetime(["2012-05-29"] * 3),
            "latitude": ("time", [35.1, 35.2, 35.3]),
            "longitude": ("time", [-97.1, -97.2, -97.3]),
            "altitude": ("time", np.array([400.0, 700.0, 6000.0]), {"units": "m"}),
        },
    )
    # y: a 3-D grid with geopotential height Z3 (m)
    lev = np.array([1, 2])
    lat = np.array([35.0])
    lon = np.array([-97.0])
    y = xr.Dataset(
        {
            "O3": (["time", "lev", "lat", "lon"], np.full((1, 2, 1, 1), 50.0)),
            "Z3": (
                ["time", "lev", "lat", "lon"],
                np.array([[[[500.0]], [[6000.0]]]]),
                {"units": "m"},
            ),
        },
        coords={
            "time": pd.to_datetime(["2012-05-29"]),
            "lev": lev,
            "lat": lat,
            "lon": lon,
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    )
    paired = IntermediateGridStrategy().pair_sources(
        x_data=x,
        y_data=y,
        x_var="O3",
        y_var="O3",
        x_source="dc8",
        y_source="cam",
        horizontal_res=1.0,
        time_resolution="1D",
        min_sample_count=1,
        vertical={"res": 1000.0, "units": "m", "extent": [0.0, 7000.0]},
    )
    assert list(paired["x_O3"].dims) == ["time", "lon", "lat", "alt"]
    assert "alt" in paired.coords and "x_sample_count" in paired and "y_sample_count" in paired
    # the two ~500 m x points share the 0-1000 m alt bin -> mean(10,30)=20
    low = paired["x_O3"].sel(alt=500.0, method="nearest").max().item()
    assert low == pytest.approx(20.0)
    # the 6000 m x point sits in a higher alt bin, away from the 500 m bin
    assert int(paired["x_sample_count"].sel(alt=500.0, method="nearest").max().item()) == 2


@pytest.mark.integration
def test_method_grid_3d_runs_through_pipeline(tmp_path):
    import numpy as np
    import pandas as pd
    import xarray as xr

    from davinci_monet.pipeline.runner import PipelineRunner

    # two file-backed sources, each with a native altitude coordinate (m)
    def alt_ds(seed):
        rng = np.random.default_rng(seed)
        n = 20
        return xr.Dataset(
            {"O3": (["time"], rng.uniform(20, 80, n))},
            coords={
                "time": pd.to_datetime(["2012-05-29"] * n),
                "latitude": ("time", rng.uniform(34, 36, n)),
                "longitude": ("time", rng.uniform(-98, -96, n)),
                "altitude": ("time", rng.uniform(0, 10000, n), {"units": "m"}),
            },
        )

    xp, yp = tmp_path / "x.nc", tmp_path / "y.nc"
    alt_ds(1).to_netcdf(xp)
    alt_ds(2).to_netcdf(yp)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "obs": {"type": "generic", "files": str(xp), "variables": {"O3": {"units": "ppb"}}},
            "mod": {"type": "generic", "files": str(yp), "variables": {"O3": {"units": "ppb"}}},
        },
        "pairs": {
            "obs_vs_mod": {
                "x": {"source": "obs", "variable": "O3"},
                "y": {"source": "mod", "variable": "O3"},
                "method": "grid",
                "grid": {
                    "horizontal_res": 1.0,
                    "time_resolution": "1D",
                    "vertical": {"res": 1000.0, "units": "m"},
                },
            }
        },
    }
    result = PipelineRunner(show_progress=False).run_from_config(config)
    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None and "obs_vs_mod" in ctx.paired
    paired = ctx.paired["obs_vs_mod"]
    data = paired.data if hasattr(paired, "data") else paired
    assert "alt" in data.coords and list(data["obs_O3"].dims) == ["time", "lon", "lat", "alt"]
