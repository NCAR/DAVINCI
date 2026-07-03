"""Pair config uses nested x:/y: (clean break from sources:/geometry:/variables:)."""

import pytest
from pydantic import ValidationError

from davinci_monet.config.schema import SourcePairConfig


def test_nested_xy_pair_parses():
    # Nested dicts are coerced to AxisRef by the field validator.
    p = SourcePairConfig(
        x={"source": "airnow", "variable": "o3"},  # type: ignore[arg-type]
        y={"source": "cam", "variable": "O3"},  # type: ignore[arg-type]
    )
    assert p.x.source == "airnow" and p.x.variable == "o3"
    assert p.y.source == "cam" and p.y.variable == "O3"
    assert p.sources == ["airnow", "cam"]


def test_old_shape_is_rejected_with_hint():
    with pytest.raises(ValidationError, match="x:|migrate"):
        SourcePairConfig(
            sources=["airnow", "cam"],
            geometry="airnow",  # type: ignore[call-arg]
            variables={"airnow": "o3", "cam": "O3"},
        )


def test_missing_axis_is_rejected():
    with pytest.raises(ValidationError):
        SourcePairConfig(x={"source": "airnow", "variable": "o3"})  # type: ignore[arg-type,call-arg]


def test_pair_config_rejects_unknown_fields_by_default():
    with pytest.raises(ValidationError, match="methd"):
        SourcePairConfig(
            x={"source": "airnow", "variable": "o3"},  # type: ignore[arg-type]
            y={"source": "cam", "variable": "O3"},  # type: ignore[arg-type]
            methd="grid",  # type: ignore[call-arg]
        )
