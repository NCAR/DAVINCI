"""Non-fatal per-item errors must be surfaced, not silently collected.

Pairing/statistics/plotting stages continue past an individual item failure and
stash the error in ``context.metadata`` (``pairing_errors``/``stats_errors``/
``plot_errors``/``analysis_errors``) instead of failing the run. Those errors
used to be collected but never shown, so a pipeline could report success while
dropping items. These tests pin both surfaces: the terminal formatter and the
Markdown report.
"""

from __future__ import annotations

from davinci_monet.pipeline.reporting import LogCollector
from davinci_monet.pipeline.runner import ProgressFormatter
from davinci_monet.pipeline.stages import PipelineContext


def test_formatter_logs_item_errors() -> None:
    fmt = ProgressFormatter(show_output=False)
    fmt.print_item_errors(
        {
            "pairing_errors": ["cam_vs_airnow_o3: no temporal overlap"],
            "analysis_errors": ["pc1_wavelet: irregular data"],
            "plot_errors": ["o3_scatter: empty paired data"],
        },
        pipeline_success=True,
    )
    log_text = "\n".join(fmt._lines)
    assert "3 non-fatal error(s)" in log_text
    assert "pipeline still succeeded" in log_text
    assert "cam_vs_airnow_o3: no temporal overlap" in log_text
    assert "pc1_wavelet: irregular data" in log_text
    assert "o3_scatter: empty paired data" in log_text


def test_formatter_failed_pipeline_item_errors_do_not_claim_success() -> None:
    fmt = ProgressFormatter(show_output=False)
    fmt.print_item_errors(
        {"plot_errors": ["o3_scatter: empty paired data"]},
        pipeline_success=False,
    )
    log_text = "\n".join(fmt._lines)
    assert "1 non-fatal error(s)" in log_text
    assert "pipeline still succeeded" not in log_text


def test_formatter_no_output_when_no_item_errors() -> None:
    fmt = ProgressFormatter(show_output=False)
    fmt.print_item_errors({}, pipeline_success=True)
    assert fmt._lines == []


def test_report_captures_item_errors() -> None:
    ctx = PipelineContext(config={})
    ctx.metadata["pairing_errors"] = ["cam_vs_airnow_o3: no temporal overlap"]
    ctx.metadata["stats_errors"] = ["cam_vs_airnow_o3: all-NaN slice"]
    ctx.metadata["analysis_errors"] = ["pc1_wavelet: irregular data"]

    collector = LogCollector()
    collector.start_pipeline(config_path=None)
    collector.end_pipeline(success=True)
    collector.extract_context_data(ctx)

    markdown = collector.to_markdown()
    assert "no temporal overlap" in markdown
    assert "all-NaN slice" in markdown
    assert "irregular data" in markdown


def test_report_distinguishes_restored_stage_from_computation() -> None:
    collector = LogCollector()
    collector.start_pipeline()
    collector.start_stage("load_sources")
    collector.end_stage(
        "load_sources",
        "completed",
        0.0,
        disposition="restored",
    )
    collector.end_pipeline(success=True)

    markdown = collector.to_markdown()

    assert "| load_sources | ✓ Completed | restored | 0.0s |" in markdown
