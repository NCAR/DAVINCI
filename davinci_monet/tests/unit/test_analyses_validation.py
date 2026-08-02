"""Cross-reference and dependency rules for the analyses: block."""

from __future__ import annotations

import pytest

from davinci_monet.config.schema import MonetConfig

_SOURCES = {"cam": {"type": "generic", "files": "x.nc", "variables": {"O3": {"units": "ppb"}}}}


def _config(**values: object) -> MonetConfig:
    return MonetConfig.model_validate({"sources": _SOURCES, **values})


def test_analysis_unknown_source_rejected() -> None:
    with pytest.raises(ValueError, match="references unknown source"):
        _config(
            analyses={"a": {"type": "eof", "source": "nope", "variable": "O3"}},
        )


def test_analysis_unknown_secondary_input_rejected_with_role() -> None:
    with pytest.raises(ValueError, match="target_grid_from.*unknown source 'nope'"):
        _config(
            analyses={
                "a": {
                    "type": "aod_preprocess",
                    "source": "cam",
                    "variable": "O3",
                    "target_grid_from": "nope",
                }
            },
        )


def test_analysis_cycle_rejected() -> None:
    with pytest.raises(ValueError, match="cycle"):
        _config(
            analyses={
                "a": {"type": "wavelet", "source": "b", "variable": "pc"},
                "b": {"type": "wavelet", "source": "a", "variable": "pc"},
            },
        )


def test_analysis_cycle_through_secondary_input_rejected() -> None:
    with pytest.raises(ValueError, match="cycle"):
        _config(
            analyses={
                "a": {
                    "type": "aod_preprocess",
                    "source": "cam",
                    "variable": "O3",
                    "target_grid_from": "b",
                },
                "b": {
                    "type": "aod_preprocess",
                    "source": "cam",
                    "variable": "O3",
                    "target_grid_from": "a",
                },
            },
        )


def test_analysis_key_collides_with_source_rejected() -> None:
    with pytest.raises(ValueError, match="collides"):
        _config(
            analyses={"cam": {"type": "eof", "source": "cam", "variable": "O3"}},
        )


def test_pair_referencing_derived_source_rejected() -> None:
    with pytest.raises(ValueError, match="derived sources are not pairable") as excinfo:
        _config(
            analyses={"cam_eof": {"type": "eof", "source": "cam", "variable": "O3"}},
            pairs={
                "p": {
                    "x": {"source": "cam", "variable": "O3"},
                    "y": {"source": "cam_eof", "variable": "O3"},
                }
            },
        )
    # The misleading duplicate "references unknown source" message must NOT fire
    # for this pair — only the specific not-pairable message.
    assert "references unknown source" not in str(excinfo.value)


def test_pair_may_reference_physical_aod_scaling_output() -> None:
    cfg = _config(
        analyses={
            "scaling": {
                "type": "aod_scaling",
                "basis": "cam",
                "projection": "cam",
                "coefficients": "cam",
                "model": "cam",
            }
        },
        pairs={
            "p": {
                "x": {"source": "cam", "variable": "O3"},
                "y": {"source": "scaling", "variable": "aod_target"},
            }
        },
    )

    assert cfg.pairs["p"].y.source == "scaling"


def test_plot_may_reference_derived_source() -> None:
    cfg = _config(
        analyses={"cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3"}},
        plots={"m": {"type": "eof_pattern", "source": "cam_O3_eof", "variable": "mode"}},
    )
    assert "m" in cfg.plots
