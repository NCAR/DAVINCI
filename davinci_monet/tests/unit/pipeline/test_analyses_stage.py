"""AnalysesStage runs analyses in dependency order and registers pseudo-sources."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.analysis import DerivedAnalysis
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry
from davinci_monet.pipeline.stages.analyses import AnalysesStage
from davinci_monet.pipeline.stages.base import PipelineContext, SourceData, StageStatus


@pytest.fixture
def _fake_eof_registered():
    _prev = {name: analysis_registry.get_or_none(name) for name in ("eof", "wavelet")}

    class _FakeEOF(DerivedAnalysis):
        name = "eof"
        output_geometry = DataGeometry.GRID

        def analyze(self, data, spec):
            return xr.Dataset({"pc": ("time", np.arange(3.0))}, coords={"time": np.arange(3)})

    class _FakeWavelet(DerivedAnalysis):
        name = "wavelet"
        output_geometry = DataGeometry.SPECTRUM

        def analyze(self, data, spec):
            assert "pc" in data.data_vars  # depends on the EOF output
            return xr.Dataset({"power": (("time", "period"), np.ones((3, 2)))})

    analysis_registry.register("eof", _FakeEOF, replace=True)
    analysis_registry.register("wavelet", _FakeWavelet, replace=True)

    yield

    for name, prev in _prev.items():
        if prev is not None:
            analysis_registry.register(name, prev, replace=True)
        else:
            analysis_registry.unregister(name)


def _ctx() -> PipelineContext:
    cam = SourceData(
        data=xr.Dataset({"O3": ("time", np.arange(3.0))}, coords={"time": np.arange(3)}),
        label="cam",
        source_type="generic",
        geometry=DataGeometry.GRID,
    )
    return PipelineContext(
        config={
            "sources": {
                "cam": {"type": "generic", "files": "x.nc", "variables": {"O3": {"units": "ppb"}}}
            },
            "analyses": {
                "pc1_wav": {"type": "wavelet", "source": "cam_O3_eof", "variable": "pc", "mode": 1},
                "cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3"},
            },
        },
        sources={"cam": cam},
    )


def test_stage_registers_derived_sources_in_order(_fake_eof_registered) -> None:
    ctx = _ctx()
    stage = AnalysesStage()
    assert stage.validate(ctx) is True
    result = stage.execute(ctx)

    assert result.status is StageStatus.COMPLETED
    assert "cam_O3_eof" in ctx.sources
    assert "pc1_wav" in ctx.sources
    eof_src = ctx.sources["cam_O3_eof"]
    assert isinstance(eof_src, SourceData)
    assert eof_src.source_type == "eof"
    assert eof_src.geometry is DataGeometry.GRID
    assert eof_src.data.attrs["derived"] is True
    assert eof_src.data.attrs["geometry"] == "grid"
    assert ctx.sources["pc1_wav"].geometry is DataGeometry.SPECTRUM


def test_stage_defaults_gridded_analysis_source_label_to_analysis_key(tmp_path) -> None:
    time = np.array(["2008-07-01T00:00", "2008-07-01T03:00"], dtype="datetime64[ns]")
    cam = SourceData(
        data=xr.Dataset(
            {
                "AOD": (("time", "lat", "lon"), np.ones((2, 1, 2))),
                "MASK": (("time", "lat", "lon"), np.ones((2, 1, 2))),
            },
            coords={"time": time, "lat": [0.0], "lon": [0.0, 90.0]},
        ),
        label="cam",
        source_type="generic",
        geometry=DataGeometry.GRID,
    )
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "sources": {
                "cam": {"type": "generic", "files": "cam.nc", "variables": {"AOD": {}, "MASK": {}}}
            },
            "analyses": {
                "daily_aod": {
                    "type": "gridded_analysis",
                    "source": "cam",
                    "groupby": "day",
                    "roles": {"analysis": "AOD", "mask": "MASK"},
                    "fields": {"analyzed_aod": {"formula": 'mean(analysis, dim="time")'}},
                }
            },
        },
        sources={"cam": cam},
    )

    result = AnalysesStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    assert ctx.sources["daily_aod"].data.attrs["source_label"] == "daily_aod"


def test_stage_validate_false_when_no_analyses() -> None:
    ctx = PipelineContext(config={"sources": {}})
    assert AnalysesStage().validate(ctx) is False
