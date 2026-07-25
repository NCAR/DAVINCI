"""PlottingStage dispatches pairwise and single-source plots in the same run."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.stages import PipelineContext, SourceData
from davinci_monet.pipeline.stages.plot import PlottingStage


def _source(label: str, values: np.ndarray) -> SourceData:
    ds = xr.Dataset(
        {"O3": (("time", "site"), values)},
        coords={
            "time": np.array(
                ["2024-01-01T00:00", "2024-01-01T01:00"],
                dtype="datetime64[ns]",
            ),
            "site": [0, 1],
            "latitude": ("site", [40.0, 41.0]),
            "longitude": ("site", [-105.0, -104.0]),
        },
        attrs={"geometry": "point", "source_label": label},
    )
    return SourceData(data=ds, label=label, source_type="generic", geometry=DataGeometry.POINT)


def _paired() -> xr.Dataset:
    ds = xr.Dataset(
        {
            "obs_O3": (("time", "site"), [[30.0, 32.0], [31.0, 33.0]]),
            "model_O3": (("time", "site"), [[31.0, 34.0], [32.0, 35.0]]),
        },
        coords={
            "time": np.array(
                ["2024-01-01T00:00", "2024-01-01T01:00"],
                dtype="datetime64[ns]",
            ),
            "site": [0, 1],
            "latitude": ("site", [40.0, 41.0]),
            "longitude": ("site", [-105.0, -104.0]),
        },
    )
    ds["obs_O3"].attrs.update(axis="x", source_label="obs", dataset_variable="O3")
    ds["model_O3"].attrs.update(axis="y", source_label="model", dataset_variable="O3")
    return ds


def _context(tmp_path, plots: dict[str, dict[str, object]]) -> PipelineContext:
    return PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "sources": {
                "obs": {"type": "generic", "variables": {"O3": {}}},
                "model": {"type": "generic", "variables": {"O3": {}}},
            },
            "pairs": {
                "model_vs_obs": {
                    "x": {"source": "obs", "variable": "O3"},
                    "y": {"source": "model", "variable": "O3"},
                }
            },
            "plots": plots,
        },
        sources={
            "obs": _source("obs", np.array([[30.0, 32.0], [31.0, 33.0]])),
            "model": _source("model", np.array([[31.0, 34.0], [32.0, 35.0]])),
        },
        paired={"model_vs_obs": _paired()},
    )


def test_plotting_stage_renders_pairwise_and_single_source_specs(tmp_path) -> None:
    context = _context(
        tmp_path,
        {
            "scatter_o3": {"type": "scatter", "pairs": ["model_vs_obs"]},
            "obs_map": {"type": "spatial", "source": "obs", "variable": "O3"},
        },
    )

    result = PlottingStage().execute(context)

    assert result.status.name == "COMPLETED"
    generated = result.data["plots_generated"]
    assert any("scatter_o3" in path for path in generated)
    assert any("obs_map" in path for path in generated)
    products = result.data["plot_products"]
    assert set(products) == {"scatter_o3", "obs_map"}
    assert all("scatter_o3" in path for path in products["scatter_o3"])
    assert all("obs_map" in path for path in products["obs_map"])


def test_plotting_stage_honors_pairwise_plot_formats(tmp_path) -> None:
    context = _context(
        tmp_path,
        {"scatter.o3": {"type": "scatter", "pairs": ["model_vs_obs"], "formats": ["pdf"]}},
    )
    stale_png = tmp_path / "obs" / "00_scatter.o3.png"
    stale_png.parent.mkdir(parents=True, exist_ok=True)
    stale_png.write_text("stale")

    result = PlottingStage().execute(context)

    assert result.status.name == "COMPLETED"
    generated = result.data["plots_generated"]
    assert any(path.endswith("scatter.o3.pdf") for path in generated)
    assert not any(path.endswith("scatter.o3.png") for path in generated)
    assert result.data["plot_products"] == {
        "scatter.o3": [path for path in generated if path.endswith("scatter.o3.pdf")]
    }
    assert not stale_png.exists()
    assert not (tmp_path / "obs" / "00_scatter.pdf").exists()
    assert list(tmp_path.rglob("*.png")) == []
