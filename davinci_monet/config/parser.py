"""YAML configuration parser for the ``sources:``/``pairs:`` schema.

This module loads and parses DAVINCI YAML configuration files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TextIO

import yaml

from davinci_monet.config.schema import (
    DataProcConfig,
    MonetConfig,
)
from davinci_monet.core.exceptions import ConfigurationError
from davinci_monet.core.schema_utils import dump_schema, validate_schema

_YAML_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+\s*:")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_BRACED_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_BARE_ENV_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def _looks_like_yaml_content(value: str) -> bool:
    """Return whether a string should be treated as inline YAML content."""
    stripped = value.strip()
    if not stripped:
        return True
    if "\n" in value or "\r" in value:
        return True
    if stripped.startswith(("{", "[", "---", "...")):
        return True
    if _WINDOWS_DRIVE_RE.match(stripped):
        return False
    return _YAML_KEY_RE.match(stripped) is not None


def _read_yaml_file(path: Path) -> Any:
    """Read YAML data from an existing configuration file."""
    try:
        with path.open() as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigurationError(f"Configuration file not found: {path}") from e


def _expand_env_value(name: str, original: str) -> str:
    """Return an environment variable value or raise a config error."""
    if name not in os.environ:
        raise ConfigurationError(
            f"Unresolved environment variable {name!r} in config value: {original!r}"
        )
    return os.environ[name]


def _should_expand_bare_env(value: str, start: int, end: int) -> bool:
    """Return whether a bare $VAR match is likely an environment variable."""
    previous = value[start - 1 : start]
    next_char = value[end : end + 1]
    if previous == "$" or next_char == "$":
        return False
    if previous and previous not in "/\\:":
        return False
    if next_char and next_char not in "/\\.-":
        return False
    return True


def _expand_string(value: str) -> str:
    """Expand supported env vars and leading home directories in a string."""

    def replace_braced(match: re.Match[str]) -> str:
        return _expand_env_value(match.group(1), value)

    expanded = _BRACED_ENV_RE.sub(replace_braced, value)
    bare_source = expanded

    def replace_bare(match: re.Match[str]) -> str:
        if not _should_expand_bare_env(bare_source, match.start(), match.end()):
            return match.group(0)
        return _expand_env_value(match.group(1), bare_source)

    expanded = _BARE_ENV_RE.sub(replace_bare, bare_source)
    if expanded == "~" or expanded.startswith(("~/", "~\\")):
        expanded = os.path.expanduser(expanded)
    return expanded


def load_yaml(source: str | Path | TextIO) -> dict[str, Any]:
    """Load raw YAML from file or string.

    Parameters
    ----------
    source
        File path, file object, or YAML string.

    Returns
    -------
    dict[str, Any]
        Raw YAML data as dictionary.

    Raises
    ------
    ConfigurationError
        If YAML cannot be parsed or file not found.

    Examples
    --------
    >>> data = load_yaml("config.yaml")
    >>> data = load_yaml(Path("config.yaml"))
    >>> data = load_yaml("analysis:\\n  debug: true")
    """
    try:
        if isinstance(source, Path):
            data = _read_yaml_file(source.expanduser())
        elif isinstance(source, str):
            if _looks_like_yaml_content(source):
                data = yaml.safe_load(source)
            else:
                path = Path(source).expanduser()
                if not path.exists():
                    raise ConfigurationError(f"Configuration file not found: {source}")
                data = _read_yaml_file(path)
        else:
            # File-like object
            data = yaml.safe_load(source)

        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"YAML root must be a mapping, got {type(data)}")
        return data

    except ConfigurationError:
        raise
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML: {e}") from e
    except OSError as e:
        raise ConfigurationError(f"Error reading configuration file {source}: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {e}") from e


def expand_env_vars(data: dict[str, Any]) -> dict[str, Any]:
    """Expand environment variables in string values.

    Supports ${VAR} and $VAR syntax.

    Parameters
    ----------
    data
        Dictionary to process.

    Returns
    -------
    dict[str, Any]
        Dictionary with environment variables expanded.

    Examples
    --------
    >>> os.environ["MY_PATH"] = "/data"
    >>> expand_env_vars({"output_dir": "${MY_PATH}/output"})
    {'output_dir': '/data/output'}
    """

    def _expand(value: Any) -> Any:
        if isinstance(value, str):
            return _expand_string(value)
        elif isinstance(value, dict):
            return {k: _expand(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_expand(item) for item in value]
        return value

    result: dict[str, Any] = _expand(data)
    return result


def preprocess_config(data: dict[str, Any]) -> dict[str, Any]:
    """Preprocess raw YAML data.

    Parameters
    ----------
    data
        Raw YAML dictionary.

    Returns
    -------
    dict[str, Any]
        Preprocessed dictionary ready for validation.
    """
    # Expand environment variables
    data = expand_env_vars(data)

    # Normalize analysis time fields
    if "analysis" in data and data["analysis"]:
        analysis = data["analysis"]
        # Some configs have times as comments or nulled out
        for key in ["start_time", "end_time"]:
            if key in analysis and analysis[key] is None:
                del analysis[key]

    # Ensure all sections exist as dicts (not None)
    for section in ["sources", "pairs", "plots"]:
        if section not in data or data[section] is None:
            data[section] = {}

    return data


def _extra_keys(value: Any) -> list[str]:
    """Return Pydantic extra keys carried by a parsed schema object."""
    extra = getattr(value, "__pydantic_extra__", None) or {}
    return sorted(str(k) for k in extra)


def _raise_for_extra(value: Any, path: str) -> None:
    keys = _extra_keys(value)
    if keys:
        joined = ", ".join(f"{path}.{key}" for key in keys)
        raise ConfigurationError(f"Strict validation rejected extra field(s): {joined}")


def _validate_strict_core_fields(config: MonetConfig) -> None:
    """Reject extra fields in core schema sections while preserving extension points.

    Source reader options and renderer-specific plot kwargs remain extension points.
    This gives ``--strict`` real behavior without breaking source-reader configs that
    intentionally pass extra keys through to reader.open().
    """
    _raise_for_extra(config.analysis, "analysis")
    if config.pairing is not None:
        _raise_for_extra(config.pairing, "pairing")

    for pair_name, pair in config.pairs.items():
        _raise_for_extra(pair, f"pairs.{pair_name}")
        _raise_for_extra(pair.x, f"pairs.{pair_name}.x")
        _raise_for_extra(pair.y, f"pairs.{pair_name}.y")
        if pair.grid is not None:
            _raise_for_extra(pair.grid, f"pairs.{pair_name}.grid")
            if pair.grid.vertical is not None:
                _raise_for_extra(pair.grid.vertical, f"pairs.{pair_name}.grid.vertical")

    for plot_name, plot in config.plots.items():
        # Renderer-specific plot kwargs are allowed in plot specs. ``data_proc``
        # is structured and checked below when present.
        data_proc = plot.data_proc
        if isinstance(data_proc, DataProcConfig):
            _raise_for_extra(data_proc, f"plots.{plot_name}.data_proc")

    if config.stats is not None:
        _raise_for_extra(config.stats, "stats")
        if isinstance(config.stats.data_proc, DataProcConfig):
            _raise_for_extra(config.stats.data_proc, "stats.data_proc")

    if config.summary is not None:
        _raise_for_extra(config.summary, "summary")


def load_config(source: str | Path | TextIO, *, strict: bool = False) -> MonetConfig:
    """Load and validate a MELODIES-MONET configuration.

    This is the main entry point for loading configuration files.
    It handles YAML parsing, preprocessing, and Pydantic validation.

    Parameters
    ----------
    source
        File path, file object, or YAML string.

    Returns
    -------
    MonetConfig
        Validated configuration object.

    Raises
    ------
    ConfigurationError
        If configuration is invalid.

    Examples
    --------
    >>> config = load_config("control.yaml")
    >>> print(config.analysis.output_dir)
    /path/to/output

    >>> config = load_config(Path("control.yaml"))

    >>> yaml_str = '''
    ... analysis:
    ...   start_time: '2024-01-01'
    ...   end_time: '2024-01-02'
    ... '''
    >>> config = load_config(yaml_str)
    """
    # Load raw YAML
    data = load_yaml(source)

    # Preprocess before schema validation.
    data = preprocess_config(data)

    # Validate with Pydantic
    try:
        config = validate_schema(MonetConfig, data)
        if strict:
            _validate_strict_core_fields(config)
        return config
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}") from e


def validate_config(data: dict[str, Any], *, strict: bool = False) -> MonetConfig:
    """Validate a dictionary as a MELODIES-MONET configuration.

    Parameters
    ----------
    data
        Dictionary to validate.

    Returns
    -------
    MonetConfig
        Validated configuration object.

    Raises
    ------
    ConfigurationError
        If validation fails.
    """
    data = preprocess_config(data)
    try:
        config = validate_schema(MonetConfig, data)
        if strict:
            _validate_strict_core_fields(config)
        return config
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}") from e


def dump_config(config: MonetConfig, path: str | Path) -> None:
    """Write configuration to YAML file.

    Parameters
    ----------
    config
        Configuration to write.
    path
        Output file path.
    """
    data = dump_schema(config, exclude_none=True, exclude_unset=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def config_to_yaml(config: MonetConfig) -> str:
    """Convert configuration to YAML string.

    Parameters
    ----------
    config
        Configuration object.

    Returns
    -------
    str
        YAML string representation.
    """
    data = dump_schema(config, exclude_none=True, exclude_unset=True)
    result: str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
    return result


def merge_configs(*configs: MonetConfig | dict[str, Any]) -> MonetConfig:
    """Merge multiple configurations, later ones override earlier.

    Parameters
    ----------
    *configs
        Configuration objects or dictionaries to merge.

    Returns
    -------
    MonetConfig
        Merged configuration.

    Examples
    --------
    >>> base = load_config("base.yaml")
    >>> override = {"analysis": {"debug": True}}
    >>> merged = merge_configs(base, override)
    """
    result: dict[str, Any] = {}

    for config in configs:
        if isinstance(config, MonetConfig):
            data = dump_schema(config, exclude_none=True)
        else:
            data = config

        result = _deep_merge(result, data)

    return validate_config(result)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Parameters
    ----------
    base
        Base dictionary.
    override
        Dictionary to merge in (takes precedence).

    Returns
    -------
    dict[str, Any]
        Merged dictionary.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


class ConfigBuilder:
    """Builder pattern for constructing configurations programmatically.

    Examples
    --------
    >>> config = (ConfigBuilder()
    ...     .set_analysis(start_time="2024-01-01", end_time="2024-01-02")
    ...     .add_source("cam", type="cesm_fv", files="/data/*.nc")
    ...     .add_source("airnow", type="pt_sfc")
    ...     .build())
    """

    def __init__(self) -> None:
        """Initialize empty configuration."""
        self._data: dict[str, Any] = {
            "analysis": {},
            "sources": {},
            "pairs": {},
            "plots": {},
        }

    def set_analysis(self, **kwargs: Any) -> "ConfigBuilder":
        """Set analysis configuration.

        Parameters
        ----------
        **kwargs
            Analysis parameters (start_time, end_time, output_dir, debug).

        Returns
        -------
        ConfigBuilder
            Self for chaining.
        """
        self._data["analysis"].update(kwargs)
        return self

    def add_source(self, name: str, **kwargs: Any) -> "ConfigBuilder":
        """Add a unified source configuration."""
        self._data["sources"][name] = kwargs
        return self

    def add_pair(
        self,
        name: str,
        x: dict[str, str],
        y: dict[str, str],
        **kwargs: Any,
    ) -> "ConfigBuilder":
        """Add a binary pair as an ordered ``(x, y)``.

        Each axis is ``{"source": <label>, "variable": <var>}``. ``x`` is the
        horizontal/reference axis; ``y`` is vertical (diffs are ``y - x``).
        """
        pair: dict[str, Any] = {"x": dict(x), "y": dict(y)}
        pair.update(kwargs)
        self._data["pairs"][name] = pair
        return self

    def add_plot(self, name: str, plot_type: str, **kwargs: Any) -> "ConfigBuilder":
        """Add a plot configuration.

        Parameters
        ----------
        name
            Plot group name.
        plot_type
            Type of plot (timeseries, taylor, etc.).
        **kwargs
            Plot parameters.

        Returns
        -------
        ConfigBuilder
            Self for chaining.
        """
        self._data["plots"][name] = {"type": plot_type, **kwargs}
        return self

    def set_stats(self, **kwargs: Any) -> "ConfigBuilder":
        """Set statistics configuration.

        Parameters
        ----------
        **kwargs
            Statistics parameters.

        Returns
        -------
        ConfigBuilder
            Self for chaining.
        """
        self._data["stats"] = kwargs
        return self

    def build(self) -> MonetConfig:
        """Build and validate the configuration.

        Returns
        -------
        MonetConfig
            Validated configuration.
        """
        return validate_config(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Get raw dictionary representation.

        Returns
        -------
        dict[str, Any]
            Configuration as dictionary.
        """
        return self._data.copy()
