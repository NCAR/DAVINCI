"""AnomalyAnalysis removes a climatology measured over a fixed baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.analysis.anomaly import AnomalyAnalysis
from davinci_monet.config.schema import AnomalySpec
from davinci_monet.core.protocols import DataGeometry

BASELINE = {"baseline_start": "1991-01-01", "baseline_end": "2020-12-31"}


def _spec(**over: object) -> AnomalySpec:
    kwargs: dict[str, object] = {
        "type": "anomaly",
        "source": "power",
        "variable": "T2M",
        **BASELINE,
    }
    kwargs.update(over)
    return AnomalySpec(**kwargs)  # type: ignore[arg-type]


def _points(
    *,
    start: str = "1981-01-01",
    periods: int = 528,
    amplitude: float = 20.0,
    offsets: tuple[float, ...] = (0.0, 50.0),
    trend_per_step: float = 0.0,
    geometry: str = "point",
) -> xr.Dataset:
    """(time, site) monthly series: a seasonal cycle plus a per-site offset."""
    times = pd.date_range(start, periods=periods, freq="MS")
    season = amplitude * np.sin(2 * np.pi * (times.month.to_numpy() - 1) / 12.0)
    ramp = trend_per_step * np.arange(periods, dtype=float)
    values = (season + ramp)[:, None] + np.asarray(offsets)[None, :]
    return xr.Dataset(
        {"T2M": (("time", "site"), values, {"units": "K", "display_name": "2 m Temperature"})},
        coords={"time": times, "site": ("site", [f"s{i}" for i in range(len(offsets))])},
        attrs={"geometry": geometry},
    )


class TestSeasonalRemoval:
    def test_monthly_climatology_removes_the_seasonal_cycle(self) -> None:
        """A pure seasonal cycle plus a per-site offset must anomalise to zero."""
        out = AnomalyAnalysis().analyze(_points(), _spec())

        anomaly = out["T2M"].sel(time=slice("1991-01-01", "2020-12-31"))
        assert np.allclose(anomaly.values, 0.0, atol=1e-9)

    def test_climatology_none_leaves_the_seasonal_cycle_standing(self) -> None:
        """Subtracting a scalar baseline mean removes the offset, not the cycle."""
        out = AnomalyAnalysis().analyze(_points(), _spec(climatology="none"))

        anomaly = out["T2M"].sel(time=slice("1991-01-01", "2020-12-31"))
        assert anomaly.std().item() == pytest.approx(20.0 / np.sqrt(2.0), rel=0.05)

    def test_per_site_offsets_are_removed_independently(self) -> None:
        out = AnomalyAnalysis().analyze(_points(offsets=(0.0, 50.0, -273.0)), _spec())

        anomaly = out["T2M"].sel(time=slice("1991-01-01", "2020-12-31"))
        assert np.allclose(anomaly.values, 0.0, atol=1e-9)


class TestBaselineWindow:
    def test_baseline_does_not_drift_with_the_trend_it_measures(self) -> None:
        """The reference is the window, not the record.

        A step that begins after the baseline must appear at full size in the
        anomaly. Were the climatology taken over the whole record, the step
        would pull its own reference upward and be reported smaller than it is.
        """
        data = _points()
        step_from = data["time"] >= np.datetime64("2021-01-01")
        data["T2M"] = data["T2M"] + xr.where(step_from, 5.0, 0.0)

        out = AnomalyAnalysis().analyze(data, _spec())

        before = out["T2M"].sel(time=slice("1991-01-01", "2020-12-31"))
        after = out["T2M"].sel(time=slice("2021-01-01", None))
        assert np.allclose(before.values, 0.0, atol=1e-9)
        assert np.allclose(after.values, 5.0, atol=1e-9)

    def test_full_record_baseline_is_used_when_no_window_is_given(self) -> None:
        out = AnomalyAnalysis().analyze(_points(), _spec(baseline_start=None, baseline_end=None))

        assert np.allclose(out["T2M"].mean("time").values, 0.0, atol=1e-9)

    def test_a_baseline_selecting_no_times_raises_naming_the_record(self) -> None:
        """An empty baseline yields an all-NaN plot, so it must fail loudly."""
        spec = _spec(baseline_start="1950-01-01", baseline_end="1955-12-31")

        with pytest.raises(ValueError, match="selects no times") as excinfo:
            AnomalyAnalysis().analyze(_points(), spec)

        assert "1981-01-01" in str(excinfo.value)


class TestGeometryIsPreserved:
    def test_point_input_yields_point_output(self) -> None:
        analysis = AnomalyAnalysis()
        analysis.analyze(_points(geometry="point"), _spec())

        assert analysis.output_geometry is DataGeometry.POINT

    def test_grid_input_yields_grid_output(self) -> None:
        times = pd.date_range("1991-01-01", periods=48, freq="MS")
        field = np.random.default_rng(0).normal(size=(48, 3, 4))
        data = xr.Dataset(
            {"T2M": (("time", "lat", "lon"), field, {"units": "K"})},
            coords={"time": times, "lat": np.arange(3.0), "lon": np.arange(4.0)},
            attrs={"geometry": "grid"},
        )
        analysis = AnomalyAnalysis()
        out = analysis.analyze(data, _spec(baseline_start=None, baseline_end=None))

        assert analysis.output_geometry is DataGeometry.GRID
        assert out["T2M"].dims == ("time", "lat", "lon")

    def test_missing_geometry_attr_raises_rather_than_guessing(self) -> None:
        data = _points()
        del data.attrs["geometry"]

        with pytest.raises(ValueError, match="no attrs\\['geometry'\\]"):
            AnomalyAnalysis().analyze(data, _spec())

    def test_unknown_geometry_attr_raises(self) -> None:
        data = _points(geometry="hyperbolic")

        with pytest.raises(ValueError, match="unknown geometry"):
            AnomalyAnalysis().analyze(data, _spec())


class TestLabelling:
    def test_units_survive_and_the_quantity_is_marked_as_an_anomaly(self) -> None:
        out = AnomalyAnalysis().analyze(_points(), _spec())

        attrs = out["T2M"].attrs
        assert attrs["units"] == "K"  # subtraction does not change units
        assert attrs["display_name"] == "2 m Temperature Anomaly"
        assert attrs["long_name"] == "2 m Temperature Anomaly"
        assert "1991-01-01" in attrs["baseline"]

    def test_the_climatology_is_emitted_alongside_the_anomaly(self) -> None:
        out = AnomalyAnalysis().analyze(_points(), _spec())

        assert "T2M_climatology" in out.data_vars
        assert out["T2M_climatology"].sizes["month"] == 12

    def test_a_missing_variable_names_what_the_source_does_have(self) -> None:
        with pytest.raises(ValueError, match="ALLSKY_SFC_SW_DWN"):
            AnomalyAnalysis().analyze(_points(), _spec(variable="ALLSKY_SFC_SW_DWN"))


class TestSmoothing:
    def test_smoothing_suppresses_month_to_month_noise(self) -> None:
        rng = np.random.default_rng(1)
        data = _points()
        data["T2M"] = data["T2M"] + xr.DataArray(
            rng.normal(scale=3.0, size=data["T2M"].shape), dims=data["T2M"].dims
        )

        rough = AnomalyAnalysis().analyze(data, _spec())["T2M"]
        smooth = AnomalyAnalysis().analyze(data, _spec(smooth=12))["T2M"]

        assert float(smooth.std()) < 0.5 * float(rough.std())

    def test_smoothing_is_centred_and_leaves_the_edges_nan(self) -> None:
        """A trailing mean would shift features in time; a centred one does not."""
        out = AnomalyAnalysis().analyze(_points(trend_per_step=0.01), _spec(smooth=12))

        series = out["T2M"].isel(site=0)
        assert bool(np.isnan(series.isel(time=0)))
        assert bool(np.isnan(series.isel(time=-1)))
        assert not bool(np.isnan(series.isel(time=len(series) // 2)))

    def test_smoothing_records_itself_in_the_attrs(self) -> None:
        out = AnomalyAnalysis().analyze(_points(), _spec(smooth=12))

        assert "12-step" in out["T2M"].attrs["smoothing"]

    def test_a_window_longer_than_the_record_raises(self) -> None:
        data = _points(periods=24, start="1991-01-01")

        with pytest.raises(ValueError, match="exceeds the 24-step time axis"):
            AnomalyAnalysis().analyze(data, _spec(smooth=36))

    def test_no_smoothing_by_default(self) -> None:
        out = AnomalyAnalysis().analyze(_points(), _spec())

        assert "smoothing" not in out["T2M"].attrs


class TestSpecValidation:
    def test_a_reversed_baseline_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="is after"):
            _spec(baseline_start="2020-01-01", baseline_end="1991-01-01")

    def test_a_degenerate_smoothing_window_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            _spec(smooth=1)
