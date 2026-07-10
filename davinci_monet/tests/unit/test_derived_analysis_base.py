"""Derived-analysis compatibility and named-input contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr

from davinci_monet.analysis import AnalysisResult, AnalysisRuntime, DerivedAnalysis
from davinci_monet.analysis.artifacts import ArtifactService
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry


def test_concrete_analysis_runs_and_registers() -> None:
    @analysis_registry.register("identity_t3")
    class Identity(DerivedAnalysis):
        name = "identity_t3"
        long_name = "Identity"
        output_geometry = DataGeometry.GRID

        def analyze(self, data: xr.Dataset, spec: object) -> xr.Dataset:
            return data

    ds = xr.Dataset({"x": ("t", np.arange(3.0))})
    out = analysis_registry.get("identity_t3")().analyze(ds, None)
    assert out is ds
    analysis_registry.unregister("identity_t3")


def test_named_entry_point_adapts_legacy_analyze(tmp_path) -> None:
    class Identity(DerivedAnalysis):
        output_geometry = DataGeometry.GRID

        def analyze(self, data: xr.Dataset, spec: object) -> xr.Dataset:
            return data

    ds = xr.Dataset({"x": ("t", np.arange(3.0))})
    runtime = AnalysisRuntime(None, None, ArtifactService(tmp_path))

    out = Identity().analyze_inputs({"source": ds}, None, runtime)

    assert out is ds
    assert AnalysisResult.adapt(out).dataset is ds


def test_named_only_analysis_is_concrete_and_runtime_is_frozen(tmp_path) -> None:
    class NamedOnly(DerivedAnalysis):
        output_geometry = DataGeometry.ARTIFACT

        def analyze_inputs(self, inputs, spec, runtime):  # noqa: ANN001
            return AnalysisResult(inputs["left"] + inputs["right"])

    runtime = AnalysisRuntime(None, None, ArtifactService(tmp_path))
    left = xr.Dataset({"x": ("t", [1.0])})
    right = xr.Dataset({"x": ("t", [2.0])})

    out = NamedOnly().analyze_inputs({"left": left, "right": right}, None, runtime)

    assert isinstance(out, AnalysisResult)
    np.testing.assert_allclose(out.dataset["x"], [3.0])
    with pytest.raises(FrozenInstanceError):
        runtime.start_time = None  # type: ignore[misc]


def test_base_analyze_reports_missing_implementation() -> None:
    with pytest.raises(NotImplementedError, match=r"analyze\(\) or analyze_inputs\(\)"):
        DerivedAnalysis().analyze(xr.Dataset(), None)
