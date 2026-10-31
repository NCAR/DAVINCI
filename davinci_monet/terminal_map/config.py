"""Strict optional YAML controls for the standalone viewer, separate from pipelines."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SourceConfig(_Model):
    filename: str | None = None
    variable: str | None = None
    indices: dict[str, StrictInt] = Field(default_factory=dict)


class GridConfig(_Model):
    lat: int = Field(default=180, ge=1, strict=True)
    lon: int = Field(default=360, ge=1, strict=True)


class DisplayConfig(_Model):
    grid: GridConfig = Field(default_factory=GridConfig)
    interpolation: Literal["linear", "nearest"] = "linear"
    fit: bool = False
    bins: int = Field(default=10, ge=1, le=10, strict=True)
    limits: tuple[float, float] | None = None
    mask_below: float | None = None

    @field_validator("limits")
    @classmethod
    def ordered_limits(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and value[0] >= value[1]:
            raise ValueError("Scale limits require MIN < MAX.")
        return value


class TerminalConfig(_Model):
    new_window: bool = False
    rows: int = Field(default=40, ge=12, strict=True)
    columns: int = Field(default=120, ge=45, strict=True)
    font_size: int = Field(default=12, ge=6, le=72, strict=True)


class ViewerConfig(_Model):
    source: SourceConfig = Field(default_factory=SourceConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    terminal: TerminalConfig = Field(default_factory=TerminalConfig)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _UniqueLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(filename: str | Path | None) -> ViewerConfig:
    """Read controls without opening data; reject unknown and duplicate keys."""
    if filename is None:
        return ViewerConfig()
    with Path(filename).expanduser().open() as stream:
        data = yaml.load(stream, Loader=_UniqueLoader)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("The YAML root must be a mapping.")
    return ViewerConfig.model_validate(data)


def source_path(filename: str, config_file: str | None = None) -> Path:
    """Expand ${VAR}, $VAR, and ~; anchor YAML-relative paths at the YAML file."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name not in os.environ:
            raise ValueError(f"Unresolved environment variable {name!r} in source.filename.")
        return os.environ[name]

    expanded = re.sub(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))", replace, filename)
    path = Path(expanded).expanduser()
    if not path.is_absolute() and config_file is not None:
        path = Path(config_file).expanduser().absolute().parent / path
    return path.absolute()
