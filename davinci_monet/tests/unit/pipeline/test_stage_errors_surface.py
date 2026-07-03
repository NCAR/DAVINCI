"""Tests for per-stage-error surfacing in PipelineResult (WS5-A item 2).

Verifies that PipelineResult.stage_errors collects:
- Per-item error lists stashed by stages in context.metadata
  (pairing_errors, stats_errors, plot_errors)
- Analysis item error lists stashed by stages in context.metadata
  (analysis_errors)
- Stage-level failures from StageResult.error
"""

from __future__ import annotations

from davinci_monet.pipeline.runner import PipelineResult
from davinci_monet.pipeline.stages import (
    PipelineContext,
    StageResult,
    StageStatus,
)


def _make_result(
    *,
    metadata: dict | None = None,
    stage_results: list[StageResult] | None = None,
) -> PipelineResult:
    ctx = PipelineContext(config={})
    if metadata:
        ctx.metadata.update(metadata)
    return PipelineResult(
        success=True,
        stage_results=stage_results or [],
        context=ctx,
    )


class TestStageErrorsProperty:
    """PipelineResult.stage_errors aggregates all per-item and stage failures."""

    def test_empty_when_no_errors(self) -> None:
        """Returns an empty dict when no errors occurred."""
        result = _make_result()
        assert result.stage_errors == {}

    def test_collects_pairing_errors(self) -> None:
        """pairing_errors from context.metadata appear under 'pairing_errors'."""
        result = _make_result(metadata={"pairing_errors": ["pair_a failed: timeout"]})
        errors = result.stage_errors
        assert "pairing_errors" in errors
        assert errors["pairing_errors"] == ["pair_a failed: timeout"]

    def test_collects_stats_errors(self) -> None:
        """stats_errors from context.metadata appear under 'stats_errors'."""
        result = _make_result(metadata={"stats_errors": ["pair_b: ZeroDivisionError"]})
        errors = result.stage_errors
        assert "stats_errors" in errors
        assert errors["stats_errors"] == ["pair_b: ZeroDivisionError"]

    def test_collects_plot_errors(self) -> None:
        """plot_errors from context.metadata appear under 'plot_errors'."""
        result = _make_result(metadata={"plot_errors": ["scatter_pm25: FileNotFoundError"]})
        errors = result.stage_errors
        assert "plot_errors" in errors
        assert errors["plot_errors"] == ["scatter_pm25: FileNotFoundError"]

    def test_collects_analysis_errors(self) -> None:
        """analysis_errors from context.metadata appear under 'analysis_errors'."""
        result = _make_result(metadata={"analysis_errors": ["pc1_wavelet: TypeError"]})
        errors = result.stage_errors
        assert "analysis_errors" in errors
        assert errors["analysis_errors"] == ["pc1_wavelet: TypeError"]

    def test_collects_stage_level_failure(self) -> None:
        """A FAILED StageResult surfaces under 'stage:<name>'."""
        failed_sr = StageResult(
            stage_name="pairing",
            status=StageStatus.FAILED,
            error="KeyError: 'y_o3'",
        )
        result = _make_result(stage_results=[failed_sr])
        errors = result.stage_errors
        assert "stage:pairing" in errors
        assert errors["stage:pairing"] == ["KeyError: 'y_o3'"]

    def test_skips_empty_metadata_lists(self) -> None:
        """Empty per-item error lists are not included."""
        result = _make_result(metadata={"pairing_errors": [], "plot_errors": []})
        errors = result.stage_errors
        assert "pairing_errors" not in errors
        assert "plot_errors" not in errors

    def test_multiple_error_sources_combined(self) -> None:
        """Multiple error sources are all present in one call."""
        failed_sr = StageResult(
            stage_name="statistics",
            status=StageStatus.FAILED,
            error="RuntimeError: division by zero",
        )
        result = _make_result(
            metadata={
                "pairing_errors": ["pair_c: missing data"],
                "plot_errors": ["scatter_o3: render failed"],
            },
            stage_results=[failed_sr],
        )
        errors = result.stage_errors
        assert "pairing_errors" in errors
        assert "plot_errors" in errors
        assert "stage:statistics" in errors

    def test_no_errors_from_completed_stages(self) -> None:
        """COMPLETED StageResults with no error do not appear."""
        ok_sr = StageResult(
            stage_name="load_sources",
            status=StageStatus.COMPLETED,
        )
        result = _make_result(stage_results=[ok_sr])
        errors = result.stage_errors
        assert "stage:load_sources" not in errors

    def test_no_context_does_not_raise(self) -> None:
        """stage_errors with context=None only surfaces stage-level failures."""
        failed_sr = StageResult(
            stage_name="plotting",
            status=StageStatus.FAILED,
            error="IOError: disk full",
        )
        result = PipelineResult(
            success=False,
            stage_results=[failed_sr],
            context=None,
        )
        errors = result.stage_errors
        assert "stage:plotting" in errors
