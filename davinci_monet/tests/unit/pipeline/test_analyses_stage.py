"""AnalysesStage runs analyses in dependency order and registers pseudo-sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from davinci_monet.analysis import AnalysisResult, ArtifactDeclaration, DerivedAnalysis
from davinci_monet.analysis.base import AnalysisExecutionError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry
from davinci_monet.pipeline.stages.analyses import AnalysesStage
from davinci_monet.pipeline.stages.base import PipelineContext, SourceData, StageStatus


@dataclass(frozen=True)
class _NamedSpec:
    type: str
    refs: dict[str, str]
    required: bool = False

    def input_refs(self) -> dict[str, str]:
        return dict(self.refs)

    def model_dump(self) -> dict[str, object]:
        return {"type": self.type, "refs": dict(self.refs), "required": self.required}


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


def test_stage_analysis_item_error_completes_with_warning(caplog) -> None:
    _prev = analysis_registry.get_or_none("eof")

    class _SometimesFailingEOF(DerivedAnalysis):
        name = "eof"
        output_geometry = DataGeometry.GRID

        def analyze(self, data, spec):
            if spec.variable == "BAD":
                raise RuntimeError("forced analysis failure")
            return xr.Dataset({"pc": ("time", np.arange(3.0))}, coords={"time": np.arange(3)})

    analysis_registry.register("eof", _SometimesFailingEOF, replace=True)
    try:
        ctx = _ctx()
        assert isinstance(ctx.config, dict)
        ctx.config["analyses"] = {
            "bad_eof": {"type": "eof", "source": "cam", "variable": "BAD"},
            "good_eof": {"type": "eof", "source": "cam", "variable": "O3"},
        }

        with caplog.at_level(logging.ERROR, logger="davinci_monet.pipeline.stages.analyses"):
            result = AnalysesStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        assert result.error is None
        assert "good_eof" in ctx.sources
        assert "bad_eof" not in ctx.sources
        assert ctx.metadata["analysis_errors"] == ["bad_eof: forced analysis failure"]
        records = [
            record
            for record in caplog.records
            if record.getMessage() == "Analysis 'bad_eof' failed"
        ]
        assert len(records) == 1
        assert records[0].exc_info is not None
    finally:
        if _prev is not None:
            analysis_registry.register("eof", _prev, replace=True)
        else:
            analysis_registry.unregister("eof")


def test_stage_resolves_named_inputs_runtime_and_declared_artifacts(tmp_path, monkeypatch) -> None:
    previous = analysis_registry.get_or_none("named_merge")
    seen: list[float] = []

    class _NamedMerge(DerivedAnalysis):
        name = "named_merge"
        output_geometry = DataGeometry.ARTIFACT

        def analyze_inputs(self, inputs, spec, runtime):  # noqa: ANN001
            assert runtime.start_time is not None
            assert runtime.end_time is not None
            merged = inputs["left"]["x"] + inputs["right"]["x"]
            seen.append(float(merged.values[0]))
            return AnalysisResult(
                xr.Dataset({"x": merged}),
                artifacts=(ArtifactDeclaration("product", role="merged", reload=True),),
                manifest_entries=({"role": "diagnostic", "value": seen[-1]},),
            )

    analysis_registry.register("named_merge", _NamedMerge, replace=True)
    try:
        left = SourceData(
            xr.Dataset({"x": ("time", [1.0])}),
            "left",
            "generic",
            DataGeometry.GRID,
        )
        right = SourceData(
            xr.Dataset({"x": ("time", [10.0])}),
            "right",
            "generic",
            DataGeometry.GRID,
        )
        ctx = PipelineContext(
            config={
                "analysis": {
                    "start_time": "2024-01-01",
                    "end_time": "2024-01-02",
                    "output_dir": str(tmp_path),
                }
            },
            sources={"left": left, "right": right},
        )
        specs = {
            "downstream": _NamedSpec("named_merge", {"left": "combined", "right": "right"}),
            "combined": _NamedSpec("named_merge", {"left": "left", "right": "right"}),
        }
        monkeypatch.setattr(ctx, "analyses_config", lambda: specs)

        result = AnalysesStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        assert seen == [11.0, 21.0]
        assert ctx.sources["combined"].geometry is DataGeometry.ARTIFACT
        assert ctx.sources["downstream"].data["x"].item() == 21.0
        assert (tmp_path / "products" / "combined" / "analysis.nc").exists()
        assert ctx.metadata["product_artifacts"]["combined"]["artifact_path"].endswith(
            "analysis.nc"
        )
        assert {entry["analysis"] for entry in ctx.metadata["analysis_artifacts"]} == {
            "combined",
            "downstream",
        }
        persisted = [
            entry for entry in ctx.metadata["analysis_artifacts"] if entry.get("kind") == "product"
        ]
        assert len(persisted) == 2
        for entry in persisted:
            identity = entry["identity"]
            for bucket in ("source_hashes", "config_hashes", "code_hashes"):
                assert identity[bucket]
                assert all(len(value) == 64 for value in identity[bucket].values())
            assert identity["declared"]["config_normalized"]["type"] == "named_merge"
    finally:
        if previous is not None:
            analysis_registry.register("named_merge", previous, replace=True)
        else:
            analysis_registry.unregister("named_merge")


def test_required_failure_is_fatal_and_descendant_is_dependency_blocked() -> None:
    previous = {name: analysis_registry.get_or_none(name) for name in ("eof", "wavelet")}
    wavelet_calls = 0

    class _FailingEOF(DerivedAnalysis):
        name = "eof"
        output_geometry = DataGeometry.GRID

        def analyze(self, data, spec):  # noqa: ANN001
            if spec.variable == "BAD":
                raise RuntimeError("forced required failure")
            return xr.Dataset({"pc": ("time", np.arange(3.0))})

    class _CountingWavelet(DerivedAnalysis):
        name = "wavelet"
        output_geometry = DataGeometry.SPECTRUM

        def analyze(self, data, spec):  # noqa: ANN001
            nonlocal wavelet_calls
            wavelet_calls += 1
            return xr.Dataset({"power": ("time", np.ones(3))})

    analysis_registry.register("eof", _FailingEOF, replace=True)
    analysis_registry.register("wavelet", _CountingWavelet, replace=True)
    try:
        ctx = _ctx()
        assert isinstance(ctx.config, dict)
        ctx.config["analyses"] = {
            "blocked_child": {
                "type": "wavelet",
                "source": "bad_parent",
                "variable": "pc",
                "required": True,
            },
            "bad_parent": {
                "type": "eof",
                "source": "cam",
                "variable": "BAD",
                "required": True,
            },
            "independent": {"type": "eof", "source": "cam", "variable": "O3"},
        }

        result = AnalysesStage().execute(ctx)

        assert result.status is StageStatus.FAILED
        assert result.error == "bad_parent: forced required failure"
        assert wavelet_calls == 0
        assert "independent" in ctx.sources
        assert ctx.metadata["analysis_status"] == {
            "bad_parent": "failed",
            "blocked_child": "dependency_blocked",
            "independent": "completed",
        }
        assert ctx.metadata["analysis_dependency_blocked"] == [
            {
                "analysis": "blocked_child",
                "dependencies": ["bad_parent"],
                "required": True,
            }
        ]
    finally:
        for name, old in previous.items():
            if old is not None:
                analysis_registry.register(name, old, replace=True)
            else:
                analysis_registry.unregister(name)


def test_declared_artifact_failure_is_fatal_for_optional_analysis(
    tmp_path, monkeypatch, caplog
) -> None:
    previous = analysis_registry.get_or_none("bad_artifact")

    class _BadArtifact(DerivedAnalysis):
        name = "bad_artifact"
        output_geometry = DataGeometry.GRID

        def analyze_inputs(self, inputs, spec, runtime):  # noqa: ANN001
            return AnalysisResult(
                dataset=inputs["source"],
                artifacts=(ArtifactDeclaration("unsupported"),),
            )

    analysis_registry.register("bad_artifact", _BadArtifact, replace=True)
    try:
        source = SourceData(
            xr.Dataset({"x": ("time", [1.0])}),
            "source",
            "generic",
            DataGeometry.GRID,
        )
        ctx = PipelineContext(
            config={"analysis": {"output_dir": str(tmp_path)}},
            sources={"source": source},
        )
        specs = {"bad": _NamedSpec("bad_artifact", {"source": "source"})}
        monkeypatch.setattr(ctx, "analyses_config", lambda: specs)

        with caplog.at_level(logging.ERROR, logger="davinci_monet.pipeline.stages.analyses"):
            result = AnalysesStage().execute(ctx)

        assert result.status is StageStatus.FAILED
        assert result.error is not None
        assert "artifact write failed" in result.error
        assert "bad" not in ctx.sources
        assert ctx.metadata["analysis_status"] == {"bad": "failed"}
        records = [
            record
            for record in caplog.records
            if record.getMessage() == "Artifact write failed for analysis 'bad'"
        ]
        assert len(records) == 1
        assert records[0].exc_info is not None
    finally:
        if previous is not None:
            analysis_registry.register("bad_artifact", previous, replace=True)
        else:
            analysis_registry.unregister("bad_artifact")


def test_analysis_failure_preserves_finalized_manifest_receipts(tmp_path, monkeypatch) -> None:
    previous = analysis_registry.get_or_none("partial_writer")

    class _PartialWriter(DerivedAnalysis):
        name = "partial_writer"
        output_geometry = DataGeometry.ARTIFACT

        def analyze_inputs(self, inputs, spec, runtime):  # noqa: ANN001
            raise AnalysisExecutionError(
                "second output failed",
                manifest_entries=(
                    {
                        "role": "corrected_mmr",
                        "kind": "mmr_file",
                        "status": "written",
                        "path": str(tmp_path / "first.nc4"),
                    },
                ),
            )

    analysis_registry.register("partial_writer", _PartialWriter, replace=True)
    try:
        source = SourceData(
            xr.Dataset({"x": ("time", [1.0])}),
            "source",
            "generic",
            DataGeometry.GRID,
        )
        ctx = PipelineContext(
            config={"analysis": {"output_dir": str(tmp_path)}},
            sources={"source": source},
        )
        specs = {"writer": _NamedSpec("partial_writer", {"source": "source"}, required=True)}
        monkeypatch.setattr(ctx, "analyses_config", lambda: specs)

        result = AnalysesStage().execute(ctx)

        assert result.status is StageStatus.FAILED
        assert ctx.metadata["analysis_artifacts"] == [
            {
                "role": "corrected_mmr",
                "kind": "mmr_file",
                "status": "written",
                "path": str(tmp_path / "first.nc4"),
                "analysis": "writer",
            }
        ]
        assert ctx.metadata["analysis_partial_failure"] == [
            {"analysis": "writer", "finalized_artifacts": 1}
        ]
    finally:
        if previous is not None:
            analysis_registry.register("partial_writer", previous, replace=True)
        else:
            analysis_registry.unregister("partial_writer")


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
