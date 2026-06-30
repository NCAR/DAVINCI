from __future__ import annotations

from davinci_monet.config.schema import MonetConfig


def test_plot_suites_and_inspection_config_parse() -> None:
    cfg = MonetConfig(
        sources={"product": {"type": "generic", "files": "analysis.nc"}},
        plots={},
        plot_suites={
            "daily_aod": {
                "preset": "gridded_aod_diagnostics",
                "source": "daily_aod",
                "group": "daily",
                "output_subdir": "plots/daily",
            }
        },
        inspection={
            "enabled": True,
            "required": True,
            "presets": ["gridded_aod_diagnostics"],
            "preview_format": "png",
        },
    )
    assert cfg.plot_suites["daily_aod"].preset == "gridded_aod_diagnostics"
    assert cfg.inspection.enabled is True
    assert cfg.inspection.required is True
