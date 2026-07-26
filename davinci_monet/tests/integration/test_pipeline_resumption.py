"""Compact pipeline-entry proofs for item checkpoint continuation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.analysis.gridded import GriddedAnalysis
from davinci_monet.config.schema import MonetConfig
from davinci_monet.pipeline.checkpoints.datasets import scientific_dataset_sha256
from davinci_monet.pipeline.checkpoints.manager import CheckpointManager
from davinci_monet.pipeline.runner import PipelineRunner
from davinci_monet.pipeline.stages import (
    AnalysesStage,
    BaseStage,
    InspectionStage,
    LoadSourcesStage,
    PairingStage,
    PlottingStage,
    SaveResultsStage,
    StageStatus,
    StatisticsStage,
    SummaryStage,
)


def _source(path: Path, offset: float) -> xr.Dataset:
    dataset = xr.Dataset(
        {
            "aod": (
                ("time", "lat", "lon"),
                np.arange(24, dtype=np.float64).reshape(2, 3, 4) + offset,
            )
        },
        coords={
            "time": np.array(["2008-01-01", "2008-01-02"], dtype="datetime64[ns]"),
            "lat": [-30.0, 0.0, 30.0],
            "lon": [0.0, 90.0, 180.0, 270.0],
        },
    )
    dataset.to_netcdf(path)
    return dataset


def _config(root: Path, first: Path, second: Path) -> MonetConfig:
    return MonetConfig.model_validate(
        {
            "run": {"id": "source-resume-smoke", "kind": "smoke"},
            "execution": {
                "attempt_root": root,
                "checkpoints": {
                    "mode": "required",
                    "granularity": "item",
                    "loaded_sources": True,
                    "retain": "all",
                },
            },
            "analysis": {
                "output_dir": root / "output",
                "log_dir": root / "logs",
            },
            "sources": {
                "first": {"type": "generic", "files": first},
                "second": {"type": "generic", "files": second},
            },
        }
    )


def test_loaded_sources_resume_after_first_item_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = tmp_path / "first.nc"
    second_path = tmp_path / "second.nc"
    _source(first_path, 0.0)
    _source(second_path, 100.0)
    reference = PipelineRunner(
        stages=[LoadSourcesStage()],
        show_progress=False,
        close_datasets_after_run=False,
    ).run_from_config(_config(tmp_path / "a000", first_path, second_path))
    assert reference.context is not None
    expected = {
        label: scientific_dataset_sha256(source.data)
        for label, source in reference.context.sources.items()
    }
    root = tmp_path / "a001"
    config = _config(root, first_path, second_path)
    loads: dict[str, int] = {}
    original_load = LoadSourcesStage._load_unified_source
    original_capture = CheckpointManager.capture_source
    interrupted = False

    def counted_load(self, label, raw_config, context):  # noqa: ANN001
        loads[label] = loads.get(label, 0) + 1
        return original_load(self, label, raw_config, context)

    def interrupt_after_first(self, request, source, **kwargs):  # noqa: ANN001
        nonlocal interrupted
        receipt = original_capture(self, request, source, **kwargs)
        if request.item == "first" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return receipt

    monkeypatch.setattr(LoadSourcesStage, "_load_unified_source", counted_load)
    monkeypatch.setattr(CheckpointManager, "capture_source", interrupt_after_first)

    with pytest.raises(KeyboardInterrupt):
        PipelineRunner(stages=[LoadSourcesStage()], show_progress=False).run_from_config(config)

    result = PipelineRunner(
        stages=[LoadSourcesStage()],
        show_progress=False,
        close_datasets_after_run=False,
    ).run_from_config(config, resume=True)

    assert result.success
    assert loads == {"first": 1, "second": 1}
    assert result.context is not None
    actual = {
        label: scientific_dataset_sha256(source.data)
        for label, source in result.context.sources.items()
    }
    assert actual == expected


def test_corrupt_source_item_recomputes_only_that_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = tmp_path / "corrupt_first.nc"
    second_path = tmp_path / "corrupt_second.nc"
    _source(first_path, 0.0)
    _source(second_path, 100.0)
    root = tmp_path / "a003"
    config = _config(root, first_path, second_path)
    loads: dict[str, int] = {}
    original_load = LoadSourcesStage._load_unified_source

    def counted_load(self, label, raw_config, context):  # noqa: ANN001
        loads[label] = loads.get(label, 0) + 1
        return original_load(self, label, raw_config, context)

    class InterruptOnce(BaseStage):
        calls = 0

        def __init__(self) -> None:
            super().__init__("interrupt_once")

        def execute(self, context):  # noqa: ANN001
            del context
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return self._create_result(StageStatus.COMPLETED, data={"continued": True})

    monkeypatch.setattr(LoadSourcesStage, "_load_unified_source", counted_load)
    stages = [LoadSourcesStage(), InterruptOnce()]
    with pytest.raises(KeyboardInterrupt):
        PipelineRunner(stages=stages, show_progress=False).run_from_config(config)

    manager = CheckpointManager.create(
        config,
        resume=True,
        read_only=True,
    )
    assert manager is not None
    store = manager.store
    receipt = store.read_receipt("load_sources", "first")
    assert receipt is not None
    Path(receipt.objects[0].paths[0]).write_bytes(b"corrupt")

    resumed = PipelineRunner(stages=stages, show_progress=False).run_from_config(
        config,
        resume=True,
    )

    assert resumed.success, resumed.failed_stages[0].error
    assert loads == {"first": 2, "second": 1}
    assert resumed.context is not None
    manager = resumed.context.checkpoint_manager
    assert manager is not None
    first = manager.store.read_receipt("load_sources", "first")
    second = manager.store.read_receipt("load_sources", "second")
    assert first is not None and first.generation == 2
    assert second is not None and second.generation == 1
    assert any(
        event.get("event") == "checkpoint_finalized"
        and event.get("execution_id") == "e002"
        and event.get("stage") == "load_sources"
        and event.get("item") == "first"
        and event.get("disposition") == "recomputed"
        and event.get("reason") == "object_invalid"
        for event in manager.store.read_events()
    )


def _analysis_source(path: Path) -> None:
    time = np.arange(4)
    lat = np.linspace(-30.0, 30.0, 3)
    lon = np.linspace(0.0, 270.0, 4)
    field = np.arange(48, dtype=np.float64).reshape(4, 3, 4)
    xr.Dataset(
        {
            "aod": (("time", "lat", "lon"), field),
            "mask": (("time", "lat", "lon"), np.ones_like(field)),
        },
        coords={
            "time": np.datetime64("2008-01-01") + time * np.timedelta64(1, "D"),
            "lat": lat,
            "lon": lon,
        },
    ).to_netcdf(path)


def _analysis_config(root: Path, source: Path) -> MonetConfig:
    return MonetConfig.model_validate(
        {
            "run": {"id": "analysis-resume-smoke", "kind": "smoke"},
            "execution": {
                "attempt_root": root,
                "checkpoints": {
                    "mode": "required",
                    "granularity": "item",
                    "loaded_sources": True,
                    "retain": "all",
                },
            },
            "analysis": {
                "output_dir": root / "output",
                "log_dir": root / "logs",
            },
            "sources": {"model": {"type": "generic", "files": source}},
            "analyses": {
                "daily_aod": {
                    "type": "gridded_analysis",
                    "source": "model",
                    "groupby": "day",
                    "roles": {"analysis": "aod", "mask": "mask"},
                    "fields": {"analyzed_aod": {"formula": 'mean(analysis, dim="time")'}},
                },
            },
        }
    )


def test_gridded_analysis_resumes_from_finalized_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "analysis.nc"
    _analysis_source(source)
    stages = [LoadSourcesStage(), AnalysesStage()]
    reference = PipelineRunner(
        stages=stages,
        show_progress=False,
        close_datasets_after_run=False,
    ).run_from_config(_analysis_config(tmp_path / "a010", source))
    assert reference.context is not None
    expected = scientific_dataset_sha256(reference.context.sources["daily_aod"].data)

    original_analyze = GriddedAnalysis.analyze
    original_capture = CheckpointManager.capture_source
    calls = 0
    interrupted = False

    def counted_analyze(self, data, spec):  # noqa: ANN001
        nonlocal calls
        calls += 1
        return original_analyze(self, data, spec)

    def interrupt_after_analysis(self, request, value, **kwargs):  # noqa: ANN001
        nonlocal interrupted
        receipt = original_capture(self, request, value, **kwargs)
        if request.stage == "analyses" and request.item == "daily_aod" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return receipt

    monkeypatch.setattr(GriddedAnalysis, "analyze", counted_analyze)
    monkeypatch.setattr(
        CheckpointManager,
        "capture_source",
        interrupt_after_analysis,
    )
    config = _analysis_config(tmp_path / "a011", source)
    with pytest.raises(KeyboardInterrupt):
        PipelineRunner(stages=stages, show_progress=False).run_from_config(config)

    resumed = PipelineRunner(
        stages=stages,
        show_progress=False,
        close_datasets_after_run=False,
    ).run_from_config(config, resume=True)

    assert resumed.success
    assert calls == 1
    assert resumed.context is not None
    assert list(resumed.context.sources) == ["model", "daily_aod"]
    actual = scientific_dataset_sha256(resumed.context.sources["daily_aod"].data)
    assert actual == expected
    manager = resumed.context.checkpoint_manager
    assert manager is not None
    receipt = manager.store.read_receipt("analyses", "daily_aod")
    assert receipt is not None
    assert receipt.generation == 1


def _point_source(path: Path, variable: str, offset: float) -> None:
    count = 8
    xr.Dataset(
        {variable: ("time", np.arange(count, dtype=np.float64) + offset)},
        coords={
            "time": np.datetime64("2008-01-01") + np.arange(count) * np.timedelta64(1, "h"),
            "latitude": ("time", np.linspace(10.1, 11.8, count)),
            "longitude": ("time", np.linspace(20.1, 21.8, count)),
        },
    ).to_netcdf(path)


def _pair_config(root: Path, obs: Path, model: Path) -> MonetConfig:
    pair = {
        "x": {"source": "obs", "variable": "aod"},
        "y": {"source": "model", "variable": "AOD"},
        "method": "grid",
        "grid": {
            "horizontal_res": 1.0,
            "time_resolution": "1D",
            "min_sample_count": 1,
        },
    }
    return MonetConfig.model_validate(
        {
            "run": {"id": "pair-resume-smoke", "kind": "smoke"},
            "execution": {
                "attempt_root": root,
                "checkpoints": {
                    "mode": "required",
                    "granularity": "item",
                    "loaded_sources": True,
                    "retain": "all",
                },
            },
            "analysis": {
                "output_dir": root / "output",
                "log_dir": root / "logs",
            },
            "sources": {
                "obs": {"type": "generic", "files": obs},
                "model": {"type": "generic", "files": model},
            },
            "pairs": {"first_pair": pair, "second_pair": pair},
        }
    )


def test_intermediate_grid_pairs_resume_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = tmp_path / "obs.nc"
    model = tmp_path / "model.nc"
    _point_source(obs, "aod", 0.0)
    _point_source(model, "AOD", 0.5)
    stages = [LoadSourcesStage(), PairingStage()]
    reference = PipelineRunner(
        stages=stages,
        show_progress=False,
        close_datasets_after_run=False,
    ).run_from_config(_pair_config(tmp_path / "a020", obs, model))
    assert reference.context is not None
    expected = {
        name: scientific_dataset_sha256(value.data)
        for name, value in reference.context.paired.items()
    }

    original_capture = CheckpointManager.capture_paired
    interrupted = False

    def interrupt_after_first(self, request, value):  # noqa: ANN001
        nonlocal interrupted
        receipt = original_capture(self, request, value)
        if request.item == "first_pair" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return receipt

    monkeypatch.setattr(CheckpointManager, "capture_paired", interrupt_after_first)
    config = _pair_config(tmp_path / "a021", obs, model)
    with pytest.raises(KeyboardInterrupt):
        PipelineRunner(stages=stages, show_progress=False).run_from_config(config)

    resumed = PipelineRunner(
        stages=stages,
        show_progress=False,
        close_datasets_after_run=False,
    ).run_from_config(config, resume=True)

    assert resumed.success
    assert resumed.context is not None
    actual = {
        name: scientific_dataset_sha256(value.data)
        for name, value in resumed.context.paired.items()
    }
    xr.testing.assert_identical(
        resumed.context.paired["first_pair"].data,
        resumed.context.paired["second_pair"].data,
    )
    actual = {
        name: scientific_dataset_sha256(value.data)
        for name, value in resumed.context.paired.items()
    }
    assert actual == expected
    manager = resumed.context.checkpoint_manager
    assert manager is not None
    first_receipt = manager.store.read_receipt("pairing", "first_pair")
    second_receipt = manager.store.read_receipt("pairing", "second_pair")
    assert first_receipt is not None and first_receipt.generation == 1
    assert second_receipt is not None and second_receipt.generation == 1


def test_statistics_resume_after_first_item_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = tmp_path / "stats_obs.nc"
    model = tmp_path / "stats_model.nc"
    _point_source(obs, "aod", 0.0)
    _point_source(model, "AOD", 0.5)
    stages = [LoadSourcesStage(), PairingStage(), StatisticsStage()]
    config = _pair_config(tmp_path / "a022", obs, model)
    original_calculate = StatisticsStage._calculate_stats
    original_capture = CheckpointManager.capture_json
    calculations = 0
    interrupted = False

    def counted_calculate(self, paired_data, stats_cfg):  # noqa: ANN001
        nonlocal calculations
        calculations += 1
        return original_calculate(self, paired_data, stats_cfg)

    def interrupt_after_first(self, request, value, **kwargs):  # noqa: ANN001
        nonlocal interrupted
        receipt = original_capture(self, request, value, **kwargs)
        if request.stage == "statistics" and request.item == "first_pair" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return receipt

    monkeypatch.setattr(StatisticsStage, "_calculate_stats", counted_calculate)
    monkeypatch.setattr(CheckpointManager, "capture_json", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        PipelineRunner(stages=stages, show_progress=False).run_from_config(config)

    resumed = PipelineRunner(stages=stages, show_progress=False).run_from_config(
        config,
        resume=True,
    )

    assert resumed.success
    assert calculations == 2
    assert resumed.context is not None
    assert set(resumed.context.results["statistics"].data) == {
        "first_pair",
        "second_pair",
    }
    manager = resumed.context.checkpoint_manager
    assert manager is not None
    for item in ("first_pair", "second_pair"):
        receipt = manager.store.read_receipt("statistics", item)
        assert receipt is not None and receipt.generation == 1


def _product_config(
    root: Path,
    obs: Path,
    model: Path,
    *,
    summary: bool = False,
    inspection: bool = False,
) -> MonetConfig:
    config = _pair_config(root, obs, model).model_dump(mode="python")
    config["pairs"] = {"comparison": config["pairs"]["first_pair"]}
    config["stats"] = {"metrics": ["MB", "RMSE"], "min_samples": 1}
    config["plots"] = {
        "scatter_one": {"type": "scatter", "pairs": ["comparison"]},
        "scatter_two": {"type": "scatter", "pairs": ["comparison"]},
    }
    config["summary"] = {"enabled": summary}
    config["inspection"] = {
        "enabled": inspection,
        "required": inspection,
        "presets": ["gridded_aod_diagnostics"],
        "preview_format": "png",
    }
    return MonetConfig.model_validate(config)


def test_statistics_saved_files_and_plots_resume_by_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = tmp_path / "product_obs.nc"
    model = tmp_path / "product_model.nc"
    _point_source(obs, "aod", 0.0)
    _point_source(model, "AOD", 0.5)
    stages = [
        LoadSourcesStage(),
        PairingStage(),
        StatisticsStage(),
        SaveResultsStage(),
        PlottingStage(),
    ]
    config = _product_config(tmp_path / "a030", obs, model)
    original_capture = CheckpointManager.capture_files
    interrupted = False

    def interrupt_after_first_plot(self, request, paths, **kwargs):  # noqa: ANN001
        nonlocal interrupted
        receipt = original_capture(self, request, paths, **kwargs)
        if request.stage == "plotting" and request.item == "scatter_one" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return receipt

    monkeypatch.setattr(CheckpointManager, "capture_files", interrupt_after_first_plot)
    with pytest.raises(KeyboardInterrupt):
        PipelineRunner(stages=stages, show_progress=False).run_from_config(config)

    resumed = PipelineRunner(stages=stages, show_progress=False).run_from_config(
        config,
        resume=True,
    )

    assert resumed.success
    assert resumed.context is not None
    products = resumed.context.results["plotting"].data["plot_products"]
    assert set(products) == {"scatter_one", "scatter_two"}
    assert all(Path(path).is_file() for paths in products.values() for path in paths)
    saved = resumed.context.results["save_results"].data["saved_products"]
    assert Path(saved["statistics_summary"]).is_file()
    manager = resumed.context.checkpoint_manager
    assert manager is not None
    for stage, item in (
        ("statistics", "comparison"),
        ("save_results", "statistics_summary"),
        ("plotting", "scatter_one"),
        ("plotting", "scatter_two"),
    ):
        receipt = manager.store.read_receipt(stage, item)
        assert receipt is not None and receipt.generation == 1


def _fake_summary_client(calls: list[str]):
    def build(cfg):  # noqa: ANN001
        calls.append(str(cfg.model))

        class Messages:
            def create(self, **kwargs):  # noqa: ANN003
                class Block:
                    text = "## Headline metrics\n- resumed\n## Caveats\n- compact fixture\n"

                class Usage:
                    input_tokens = 1
                    output_tokens = 2

                class Response:
                    content = [Block()]
                    usage = Usage()
                    model = cfg.model

                return Response()

        class Client:
            messages = Messages()

        return Client()

    return build


def _fake_pdftoppm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    executable = binary_dir / "pdftoppm"
    executable.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[-1] + '.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary_dir}:/usr/bin")


def test_summary_provider_and_inspection_resume_without_repeating_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import davinci_monet.ai.summarizer as summarizer_module

    obs = tmp_path / "summary_obs.nc"
    model = tmp_path / "summary_model.nc"
    _point_source(obs, "aod", 0.0)
    _point_source(model, "AOD", 0.5)
    _fake_pdftoppm(tmp_path, monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(summarizer_module, "_build_client", _fake_summary_client(calls))
    stages = [
        LoadSourcesStage(),
        PairingStage(),
        StatisticsStage(),
        SaveResultsStage(),
        PlottingStage(),
        SummaryStage(),
        InspectionStage(),
    ]
    config = _product_config(
        tmp_path / "a031",
        obs,
        model,
        summary=True,
        inspection=True,
    )
    original_capture = CheckpointManager.capture_files
    interrupted = False

    def interrupt_after_summary(self, request, paths, **kwargs):  # noqa: ANN001
        nonlocal interrupted
        receipt = original_capture(self, request, paths, **kwargs)
        if request.stage == "summary" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return receipt

    monkeypatch.setattr(CheckpointManager, "capture_files", interrupt_after_summary)
    with pytest.raises(KeyboardInterrupt):
        PipelineRunner(stages=stages, show_progress=False).run_from_config(config)

    resumed = PipelineRunner(stages=stages, show_progress=False).run_from_config(
        config,
        resume=True,
    )

    assert resumed.success
    assert calls and len(calls) == 1
    assert resumed.context is not None
    assert resumed.context.results["summary"].data["bullets"] == [
        "resumed",
        "compact fixture",
    ]
    assert resumed.context.results["inspection"].data["passed"] is True
