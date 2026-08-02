"""Plot config shape is validated before pipeline execution."""

from __future__ import annotations

import pytest

from davinci_monet.config import validate_config
from davinci_monet.core.exceptions import ConfigurationError


def _base_config(plot_spec: dict) -> dict:
    return {
        "sources": {
            "obs": {"type": "generic", "files": "obs.nc", "variables": {"O3": {}}},
            "model": {"type": "generic", "files": "model.nc", "variables": {"O3": {}}},
        },
        "pairs": {
            "model_vs_obs": {
                "x": {"source": "obs", "variable": "O3"},
                "y": {"source": "model", "variable": "O3"},
            }
        },
        "plots": {"plot_a": plot_spec},
    }


@pytest.mark.parametrize("plot_type", ["scatter", "spatial_bias", "taylor"])
def test_pairwise_plot_requires_pairs(plot_type: str) -> None:
    with pytest.raises(ConfigurationError, match=r"plots\.plot_a\.pairs is required"):
        validate_config(_base_config({"type": plot_type}))


def test_pairwise_plot_rejects_single_source_fields() -> None:
    with pytest.raises(ConfigurationError, match=r"source is invalid.*variable is invalid"):
        validate_config(
            _base_config(
                {"type": "scatter", "pairs": ["model_vs_obs"], "source": "obs", "variable": "O3"}
            )
        )


@pytest.mark.parametrize(
    ("plot_spec", "message"),
    [
        ({"type": "scatter", "pairs": ["model_vs_obs"], "source": "obs"}, "source"),
        ({"type": "scatter", "pairs": ["model_vs_obs"], "variable": "O3"}, "variable"),
    ],
)
def test_pairwise_plot_rejects_partial_single_source_fields(plot_spec: dict, message: str) -> None:
    with pytest.raises(ConfigurationError, match=rf"{message} is invalid"):
        validate_config(_base_config(plot_spec))


@pytest.mark.parametrize("plot_type", ["spatial", "flight_track", "histogram"])
def test_single_source_plot_requires_source_and_variable(plot_type: str) -> None:
    with pytest.raises(ConfigurationError, match=r"source is required.*variable is required"):
        validate_config(_base_config({"type": plot_type}))


@pytest.mark.parametrize(
    ("plot_spec", "message"),
    [
        ({"type": "spatial", "source": "obs"}, "variable"),
        ({"type": "spatial", "variable": "O3"}, "source"),
    ],
)
def test_single_source_plot_rejects_partial_source_variable(plot_spec: dict, message: str) -> None:
    with pytest.raises(ConfigurationError, match=rf"{message} is required"):
        validate_config(_base_config(plot_spec))


def test_single_source_plot_rejects_pairs() -> None:
    with pytest.raises(ConfigurationError, match=r"pairs is invalid for single-source"):
        validate_config(
            _base_config(
                {"type": "spatial", "pairs": ["model_vs_obs"], "source": "obs", "variable": "O3"}
            )
        )


def test_timeseries_accepts_pairwise_shape() -> None:
    cfg = validate_config(_base_config({"type": "timeseries", "pairs": ["model_vs_obs"]}))
    assert cfg.plots["plot_a"].pairs == ["model_vs_obs"]


def test_timeseries_accepts_single_source_shape() -> None:
    cfg = validate_config(_base_config({"type": "timeseries", "source": "obs", "variable": "O3"}))
    assert cfg.plots["plot_a"].source == "obs"


def test_explicit_multi_source_plot_accepts_named_source_variables() -> None:
    cfg = validate_config(
        _base_config(
            {
                "type": "eof_wavelet_science",
                "sources": [
                    {"source": "obs", "variable": "O3"},
                    {"source": "model", "variable": "O3"},
                ],
            }
        )
    )
    assert [ref.source for ref in cfg.plots["plot_a"].sources] == ["obs", "model"]


def test_explicit_multi_source_plot_rejects_unknown_source() -> None:
    with pytest.raises(ConfigurationError, match="sources references unknown source 'missing'"):
        validate_config(
            _base_config(
                {
                    "type": "eof_wavelet_science",
                    "sources": [
                        {"source": "obs", "variable": "O3"},
                        {"source": "missing", "variable": "O3"},
                    ],
                }
            )
        )


def test_timeseries_rejects_mixed_shape() -> None:
    with pytest.raises(ConfigurationError, match=r"must use either pairs or source/variable"):
        validate_config(
            _base_config(
                {"type": "timeseries", "pairs": ["model_vs_obs"], "source": "obs", "variable": "O3"}
            )
        )
