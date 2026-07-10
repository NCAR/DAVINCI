"""Contracts shared by derived analyses and their pipeline adapter."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr

    from davinci_monet.core.protocols import DataGeometry

    from .artifacts import ArtifactService


@dataclass(frozen=True)
class ArtifactDeclaration:
    """Request that the pipeline persist an analysis result using ``kind`` policy."""

    kind: str
    role: str = "analysis"
    reload: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisRuntime:
    """Immutable services and requested window exposed during one analysis run."""

    start_time: datetime | None
    end_time: datetime | None
    artifact_service: "ArtifactService"


@dataclass(frozen=True)
class AnalysisResult:
    """Dataset plus optional persistence and manifest declarations."""

    dataset: "xr.Dataset"
    artifacts: tuple[ArtifactDeclaration, ...] = ()
    manifest_entries: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def adapt(cls, value: "AnalysisResult | xr.Dataset") -> "AnalysisResult":
        """Adapt the legacy plain-dataset return contract."""
        import xarray as xr

        if isinstance(value, cls):
            return value
        if isinstance(value, xr.Dataset):
            return cls(dataset=value)
        raise TypeError(
            "derived analysis must return an xarray.Dataset or AnalysisResult, "
            f"got {type(value).__name__}"
        )


class AnalysisExecutionError(RuntimeError):
    """Analysis failure carrying receipts for outputs finalized before the error."""

    def __init__(
        self,
        message: str,
        *,
        manifest_entries: tuple[Mapping[str, Any], ...] = (),
    ) -> None:
        super().__init__(message)
        self.manifest_entries = manifest_entries


class DerivedAnalysis(ABC):
    """An analysis that consumes named datasets and emits a derived dataset.

    Concrete analyses register via ``@analysis_registry.register("<type>")`` and
    set ``output_geometry`` to the geometry of their principal output field.
    Existing single-source implementations only need to implement :meth:`analyze`;
    the default :meth:`analyze_inputs` adapter preserves that contract.
    """

    name: str = "base"
    long_name: str = "Base Derived Analysis"
    output_geometry: "DataGeometry"

    def analyze(self, data: "xr.Dataset", spec: Any) -> "xr.Dataset":
        """Legacy single-source compatibility hook.

        ``data`` is the fully-built input dataset (a raw source or an
        already-built derived source). ``spec`` is the validated Pydantic
        params for this analysis entry.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement analyze() or analyze_inputs()"
        )

    def analyze_inputs(
        self,
        inputs: Mapping[str, "xr.Dataset"],
        spec: Any,
        runtime: AnalysisRuntime,
    ) -> "AnalysisResult | xr.Dataset":
        """Run with named inputs, adapting legacy single-source analyses."""
        del runtime
        try:
            source = inputs["source"]
        except KeyError as exc:
            raise ValueError("legacy analysis requires a named 'source' input") from exc
        return self.analyze(source, spec)
