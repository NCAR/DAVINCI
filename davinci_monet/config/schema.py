"""Pydantic schemas for DAVINCI configuration validation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from davinci_monet.plots.contracts import validate_plot_shape

# =============================================================================
# Base Configuration
# =============================================================================


class StrictSchema(
    BaseModel,
    extra="forbid",
    validate_default=True,
    str_strip_whitespace=True,
):
    """Base dataset with strict validation settings."""


class FlexibleSchema(
    BaseModel,
    extra="allow",
    validate_default=True,
    str_strip_whitespace=True,
):
    """Base schema that allows reader-specific extra fields."""


# =============================================================================
# Plot Style Configuration
# =============================================================================


class PlotStyleConfig(StrictSchema):
    """Configuration for plot styling.

    Parameters
    ----------
    theme
        Plot theme to apply. Options:
        - "ncar": NSF NCAR brand colors and fonts (Poppins)
        - "default": matplotlib defaults
        - None: no theme applied (use current matplotlib state)
    context
        Font size context for the theme:
        - "default": Standard sizes suitable for most uses
        - "presentation": Larger sizes for slides
        - "publication": Smaller sizes for journal figures
    use_seaborn
        If True and seaborn is available, apply seaborn theme
        for cleaner grid styling.
    seaborn_style
        Seaborn style to apply if use_seaborn is True.
        Options: "whitegrid", "darkgrid", "white", "dark", "ticks"
    """

    theme: Literal["ncar", "default"] | None = None
    context: Literal["default", "presentation", "publication"] = "default"
    use_seaborn: bool = True
    seaborn_style: str = "whitegrid"


# =============================================================================
# Analysis Section
# =============================================================================


class AnalysisConfig(StrictSchema):
    """Configuration for the analysis section.

    Parameters
    ----------
    start_time
        Start time of analysis window (UTC).
    end_time
        End time of analysis window (UTC).
    output_dir
        Directory for output files.
    log_dir
        Directory for log files.
    debug
        Enable debug mode.
    style
        Plot styling configuration (NCAR branding, fonts, colors).
    city_labels
        Optional mapping of ``city name -> [lat, lon]`` annotated on spatial
        and 3-D track plots (forwarded to renderers via the plotting stage).
    domain
        Optional named map domain (e.g. ``asia_aq``) applied as the fixed extent
        of every spatial map, so sparse-data maps are not auto-clipped to their
        few sites. A per-plot ``domain_type`` overrides it.
    """

    start_time: datetime | str | None = None
    end_time: datetime | str | None = None
    output_dir: Path | str | None = None
    log_dir: Path | str | None = None
    debug: bool = False
    style: PlotStyleConfig | dict[str, Any] | None = None
    city_labels: dict[str, list[float]] | None = None
    domain: str | None = None

    @field_validator("style", mode="before")
    @classmethod
    def parse_style(cls, v: Any) -> PlotStyleConfig | None:
        """Parse style configuration."""
        if v is None:
            return None
        if isinstance(v, dict):
            return PlotStyleConfig(**v)
        return v

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def parse_datetime(cls, v: Any) -> datetime | None:
        """Parse datetime from various formats."""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Handle MELODIES-MONET format: '2019-08-02-12:00:00'
            for fmt in [
                "%Y-%m-%d-%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse datetime: {v}")
        # Return value unchanged for Pydantic to handle
        result: datetime | None = v
        return result

    @field_validator("output_dir", mode="before")
    @classmethod
    def parse_path(cls, v: Any) -> Path | None:
        """Convert string to Path."""
        if v is None:
            return None
        return Path(v)


# =============================================================================
# Variable Configuration
# =============================================================================


class VariableConfig(StrictSchema):
    """Configuration for a single variable.

    Parameters
    ----------
    unit_scale
        Scaling factor for unit conversion.
    unit_scale_method
        Method for scaling: '*', '+', '-', '/'.
    valid_min
        Minimum valid value; values below this threshold are set to NaN.
    valid_max
        Maximum valid value; values above this threshold are set to NaN.
    nan_value
        Value to treat as NaN.
    source_name
        Original variable name in the data file (e.g., 'NO_ESRL').
        If set, the source variable is renamed to the config key name
        before other transforms are applied.
    rename
        Rename variable to this name.
    units
        Unit string for variable (e.g., 'ppb', 'μg/m³').
    display_name
        Display name for plots (e.g., 'PM₂.₅', 'O₃'). Overrides automatic formatting.
    ylabel_plot
        Y-axis label for plots.
    ty_scale
        Scale for Taylor diagrams.
    vmin_plot
        Minimum value for plot axis.
    vmax_plot
        Maximum value for plot axis.
    vdiff_plot
        +/- range for bias plots.
    nlevels_plot
        Number of contour levels.
    LLOD_value
        Lower limit of detection value.
    LLOD_setvalue
        Value to replace LLOD with.
    need
        Whether this variable is needed.
    """

    source_name: str | None = None
    unit_scale: float = 1.0
    unit_scale_method: Literal["*", "+", "-", "/"] = "*"
    valid_min: float | None = None
    valid_max: float | None = None
    nan_value: float | None = None
    rename: str | None = None
    units: str | None = None
    display_name: str | None = None
    ylabel_plot: str | None = None
    ty_scale: float | None = None
    vmin_plot: float | None = None
    vmax_plot: float | None = None
    vdiff_plot: float | None = None
    nlevels_plot: int | None = None
    LLOD_value: float | None = None
    LLOD_setvalue: float | None = None
    need: bool | None = None


# =============================================================================
# Plot / Source Keyword Arguments
# =============================================================================


class PlotKwargs(StrictSchema):
    """Matplotlib plot keyword arguments."""

    color: str | None = None
    marker: str | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    markersize: float | None = None


class FilterConfig(StrictSchema):
    """Configuration for data filtering."""

    value: Any
    oper: str  # 'isin', '<', '>', '==', etc.


# =============================================================================
# Plot Configuration
# =============================================================================


def _registered_plot_types() -> list[str]:
    """Return currently registered plotter names, importing built-ins first."""
    import davinci_monet.plots  # noqa: F401  # registers built-in renderers
    from davinci_monet.plots import list_plotters

    return list_plotters()


class SourceConfig(FlexibleSchema):
    """Unified configuration for a single data source.

    A data source is data with a declared geometry. Pairing direction is chosen
    by each pair's ``geometry`` field and by geometry precedence when that field
    is omitted. Extra reader-specific keys are accepted and passed through.
    """

    type: str | None = None
    files: str | Path | list[str | Path] | None = None
    filename: str | Path | None = None
    variables: dict[str, VariableConfig] = Field(default_factory=dict)
    radius_of_influence: float = 12000.0
    display_name: str | None = None
    resample: str | None = None
    min_sample_count: int | None = None
    track_sample_count: bool = False

    @model_validator(mode="before")
    @classmethod
    def reject_source_mapping(cls, data: Any) -> Any:
        """Pair variables must live in ``pairs:``, not in source metadata."""
        if isinstance(data, dict) and "mapping" in data:
            raise ValueError("source-level mapping is not supported; use pairs variables")
        return data

    @field_validator("files", mode="before")
    @classmethod
    def _convert_files(cls, v: Any) -> str | list[str] | None:
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return [str(item) for item in v]
        return str(v)

    @field_validator("filename", mode="before")
    @classmethod
    def _convert_filename(cls, v: Any) -> str | None:
        return None if v is None else str(v)

    @field_validator("variables", mode="before")
    @classmethod
    def _parse_variables(cls, v: Any) -> dict[str, VariableConfig]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return {
                str(name): VariableConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
        return dict(v)


class AxisRef(StrictSchema):
    """One axis of a pair: a source label and the variable to read from it."""

    source: str
    variable: str


class VerticalGridConfig(StrictSchema):
    """Vertical (altitude) settings for a 3-D intermediate grid (Phase 2)."""

    res: float
    units: str = "m"
    extent: tuple[float, float] | None = None


class GridConfig(StrictSchema):
    """Intermediate-grid settings for a pair using ``method: grid`` (2-D, Phase 1)."""

    horizontal_res: float
    extent: tuple[float, float, float, float] | None = None
    time_resolution: str = "1D"
    min_sample_count: int = 1
    vertical: VerticalGridConfig | None = None

    @field_validator("vertical", mode="before")
    @classmethod
    def _parse_vertical(cls, v: Any) -> Any:
        return VerticalGridConfig(**v) if isinstance(v, dict) else v


class PipelinePairingConfig(StrictSchema):
    """Runtime options for the pipeline pairing stage."""

    time_tolerance: str = "1h"
    time_method: str = "nearest"
    max_pair_workers: int | None = None
    dask_pair_workers: int = 1


class SourcePairConfig(FlexibleSchema):
    """Binary pair definition as an ordered (x, y).

    ``x`` is the horizontal/reference axis; ``y`` is vertical. Diffs are ``y - x``.
    Pairing *direction* (which source is resampled onto which) is decided by shape
    precedence, not by x/y — x/y is plot-axis assignment only. On a same-shape tie,
    ``x`` is the reference (pairing) geometry.
    """

    x: AxisRef
    y: AxisRef
    method: Literal["auto", "grid"] = "auto"
    grid: GridConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_shape(cls, data: Any) -> Any:
        if isinstance(data, dict) and any(k in data for k in ("sources", "geometry", "variables")):
            raise ValueError(
                "legacy pair shape (sources:/geometry:/variables:) is no longer "
                "supported; migrate to nested x:/y:, e.g.\n"
                "  x: {source: airnow, variable: o3}\n"
                "  y: {source: cam, variable: O3}"
            )
        return data

    @field_validator("x", "y", mode="before")
    @classmethod
    def _parse_axis(cls, v: Any) -> Any:
        return AxisRef(**v) if isinstance(v, dict) else v

    @field_validator("grid", mode="before")
    @classmethod
    def _parse_grid(cls, v: Any) -> Any:
        return GridConfig(**v) if isinstance(v, dict) else v

    @model_validator(mode="after")
    def _validate_method_grid(self) -> "SourcePairConfig":
        if self.method == "grid" and self.grid is None:
            raise ValueError("method: grid requires a 'grid:' block with horizontal_res")
        if self.method == "auto" and self.grid is not None:
            raise ValueError("'grid:' is only valid with method: grid (got method: auto)")
        return self

    @property
    def sources(self) -> list[str]:
        """Compatibility accessor: the two source labels in (x, y) order."""
        return [self.x.source, self.y.source]


class DataProcConfig(StrictSchema):
    """Data processing configuration for plots.

    Parameters
    ----------
    filter_dict
        Dictionary-based filtering.
    filter_string
        Pandas query string for filtering.
    rem_by_nan_pct
        Remove datasets by NaN percentage.
    rem_nan
        Remove NaN datasets.
    ts_select_time
        Time selection for timeseries: 'time' (UTC) or 'time_local'.
    ts_avg_window
        Pandas resample rule for averaging.
    set_axis
        Use variable-specified axis limits.
    """

    filter_dict: dict[str, FilterConfig | dict[str, Any]] | None = None
    filter_string: str | None = None
    rem_by_nan_pct: dict[str, Any] | None = None
    rem_nan: bool = True
    ts_select_time: Literal["time", "time_local"] = "time"
    ts_avg_window: str | None = None
    set_axis: bool = False


class FigKwargs(StrictSchema):
    """Figure keyword arguments."""

    figsize: list[float] | tuple[float, float] | None = None
    states: bool | None = None


class TextKwargs(StrictSchema):
    """Text styling keyword arguments."""

    fontsize: float = 12.0


class PlotGroupConfig(FlexibleSchema):
    """Configuration for a plot group.

    Parameters
    ----------
    type
        Plot type.
    fig_kwargs
        Figure keyword arguments.
    default_plot_kwargs
        Default plot styling.
    text_kwargs
        Text styling.
    domain_type
        List of domain types: 'all' or specific domains.
    domain_name
        List of domain names.
    data
        List of pair identifiers.
    data_proc
        Data processing configuration.
    """

    type: str
    fig_kwargs: FigKwargs | dict[str, Any] = Field(default_factory=dict)
    default_plot_kwargs: PlotKwargs | dict[str, Any] = Field(default_factory=dict)
    text_kwargs: TextKwargs | dict[str, Any] = Field(default_factory=dict)
    domain_type: list[str] = Field(default_factory=lambda: ["all"])
    domain_name: list[str] = Field(default_factory=lambda: ["CONUS"])
    pairs: list[str] = Field(default_factory=list)
    source: str | None = None
    variable: str | None = None
    mode: int | None = None
    display_level: int | None = None
    data_proc: DataProcConfig | dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_data_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and "data" in data:
            raise ValueError("plots.*.data is no longer supported; use plots.*.pairs")
        return data

    @field_validator("type")
    @classmethod
    def validate_plot_type(cls, v: str) -> str:
        """Reject unknown plot types during config validation."""
        registered = _registered_plot_types()
        if v not in registered:
            available = ", ".join(registered)
            raise ValueError(f"Unknown plot type '{v}'. Available plot types: {available}")
        return v

    @field_validator("data_proc", mode="before")
    @classmethod
    def parse_data_proc(cls, v: Any) -> DataProcConfig | dict[str, Any]:
        """Parse data processing config."""
        if v is None:
            return DataProcConfig()
        if isinstance(v, dict):
            return DataProcConfig(**v)
        result: DataProcConfig | dict[str, Any] = v
        return result


# =============================================================================
# Statistics Configuration
# =============================================================================


StatMetric = Literal[
    "MB",
    "MdnB",
    "NMB",
    "NMdnB",
    "R2",
    "RMSE",
    "STDX",
    "STDY",
    "MX",
    "MY",
    "MdnX",
    "MdnY",
    "RM",
    "RMdn",
    "FB",
    "ME",
    "MdnE",
    "NME",
    "NMdnE",
    "FE",
    "d1",
    "E1",
    "IOA",
    "AC",
]


class OutputTableKwargs(StrictSchema):
    """Keyword arguments for statistics output table."""

    figsize: list[float] | tuple[float, float] | None = None
    fontsize: float = 12.0
    xscale: float = 1.0
    yscale: float = 1.0
    edges: str = "horizontal"


class StatsConfig(StrictSchema):
    """Configuration for statistics calculation.

    Parameters
    ----------
    stat_list
        List of statistics to calculate.
    round_output
        Decimal places for rounding.
    output_table
        Generate output table image.
    output_table_kwargs
        Table styling options.
    domain_type
        List of domain types.
    domain_name
        List of domain names.
    data
        List of pair identifiers.
    data_proc
        Data processing configuration.
    include_counts
        Include per-metric sample counts in the statistics output.
    remove_nan
        Drop NaN pairs before computing metrics.
    min_samples
        Minimum number of valid pairs required to compute metrics.
    per_flight
        Also compute per-flight statistics when a ``flight`` coordinate exists.
    """

    stat_list: list[str] = Field(default_factory=lambda: ["MB", "NMB", "R2", "RMSE"])
    metrics: list[str] | None = None
    round_output: int = 3
    output_table: bool = False
    output_table_kwargs: OutputTableKwargs | dict[str, Any] = Field(default_factory=dict)
    domain_type: list[str] = Field(default_factory=lambda: ["all"])
    domain_name: list[str] = Field(default_factory=lambda: ["CONUS"])
    data: list[str] = Field(default_factory=list)
    data_proc: DataProcConfig | dict[str, Any] | None = None
    include_counts: bool = True
    remove_nan: bool = True
    min_samples: int = 3
    per_flight: bool = False


# =============================================================================
# AI Summary Configuration
# =============================================================================


class SummaryConfig(StrictSchema):
    """Configuration for the optional AI analysis summary stage.

    When ``enabled`` is true, a final pipeline stage sends the run's
    statistics, config metadata, and selected plot images to an AI model
    (via the Anthropic API directly, or via OpenRouter) and writes a markdown
    brief into the analysis output directory.
    """

    enabled: bool = False
    provider: Literal["anthropic", "openrouter"] = "anthropic"
    model: str = "claude-haiku-4-5"
    max_tokens: int = 2000
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_key_file: str | None = None
    plots: list[str] | None = None
    max_images: int = 8
    output_filename: str = "AI_summary.md"
    instructions: str | None = None
    templates: dict[str, dict] | None = None
    template_overrides: dict[str, str] | None = None

    @model_validator(mode="after")
    def _apply_provider_defaults(self) -> "SummaryConfig":
        """Flip Anthropic-default sentinels to OpenRouter equivalents.

        Only fields still holding the Anthropic default are changed, so an
        explicit user value is never overridden.
        """
        if self.provider == "openrouter":
            if self.model == "claude-haiku-4-5":
                self.model = "anthropic/claude-haiku-4.5"
            if self.api_key_env == "ANTHROPIC_API_KEY":
                self.api_key_env = "OPENROUTER_API_KEY"
        return self


# =============================================================================
# Derived-Analysis Specs
# =============================================================================


class PointReduce(StrictSchema):
    """Reduce a gridded field to a series at a single (lat, lon) point."""

    point: tuple[float, float]


class EOFSpec(StrictSchema):
    """EOF decomposition of one gridded source variable."""

    type: Literal["eof"]
    source: str
    variable: str
    n_modes: int = 10
    standardize: bool = False
    remove_seasonal_cycle: bool = False
    rotation: Literal["none", "varimax"] = "none"
    level: int | None = None


class WaveletSpec(StrictSchema):
    """Continuous wavelet transform of one source variable (a 1-D series)."""

    type: Literal["wavelet"]
    source: str
    variable: str
    mode: int | None = None
    reduce: Literal["area_mean"] | PointReduce | None = "area_mean"
    omega0: float = 6.0
    significance_level: float = 0.95
    dj: float = 0.25
    s0: float | None = None
    j: int | None = None

    @field_validator("reduce", mode="before")
    @classmethod
    def _parse_reduce(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return PointReduce(**v)
        return v


AnalysisSpec = EOFSpec | WaveletSpec


def build_analysis_spec(cfg: Any) -> AnalysisSpec:
    """Build the right AnalysisSpec submodel from a dict, dispatching on type."""
    if isinstance(cfg, (EOFSpec, WaveletSpec)):
        return cfg
    if not isinstance(cfg, dict):
        raise ValueError(f"analysis entry must be a mapping, got {type(cfg).__name__}")
    analysis_type = cfg.get("type")
    if analysis_type == "eof":
        return EOFSpec(**cfg)
    if analysis_type == "wavelet":
        return WaveletSpec(**cfg)
    raise ValueError(
        f"Unknown analysis type '{analysis_type}'. Available analysis types: eof, wavelet"
    )


# =============================================================================
# Root Configuration
# =============================================================================


class MonetConfig(StrictSchema):
    """Root configuration dataset for MELODIES-MONET.

    This is the top-level configuration that contains all sections.

    Parameters
    ----------
    analysis
        Analysis configuration (time window, output directory).
    sources
        Dictionary of unified data-source configurations keyed by source label.
    pairs
        Dictionary of binary pair definitions keyed by pair name.
    pairing
        Runtime options for the pipeline pairing stage.
    plots
        Dictionary of plot group configurations keyed by group name.
    stats
        Statistics configuration.

    Examples
    --------
    >>> from davinci_monet.config.schema import MonetConfig
    >>> config = MonetConfig(**{
    ...     "analysis": {"start_time": "2024-01-01", "end_time": "2024-01-02"},
    ...     "sources": {
    ...         "cam": {"type": "cesm_fv"},
    ...         "airnow": {"type": "pt_sfc"},
    ...     },
    ... })
    >>> config.analysis.start_time
    datetime.datetime(2024, 1, 1, 0, 0)
    """

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    # Data sources keyed by dataset label.
    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    pairs: dict[str, SourcePairConfig] = Field(default_factory=dict)
    pairing: PipelinePairingConfig | None = None
    plots: dict[str, PlotGroupConfig] = Field(default_factory=dict)
    analyses: dict[str, AnalysisSpec] = Field(default_factory=dict)
    stats: StatsConfig | None = None
    summary: SummaryConfig | None = None

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v: Any) -> dict[str, SourceConfig]:
        """Parse unified data-source configurations."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return {
                str(name): SourceConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
        return dict(v)

    @field_validator("pairs", mode="before")
    @classmethod
    def parse_pairs(cls, v: Any) -> dict[str, SourcePairConfig]:
        """Parse unified source-pair configurations."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return {
                str(name): SourcePairConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
        return dict(v)

    @field_validator("analyses", mode="before")
    @classmethod
    def parse_analyses(cls, v: Any) -> dict[str, AnalysisSpec]:
        """Parse derived-analysis configurations (dispatch on type)."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return {
                str(name): build_analysis_spec(cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
        return dict(v)

    @field_validator("plots", mode="before")
    @classmethod
    def parse_plots(cls, v: Any) -> dict[str, PlotGroupConfig]:
        """Parse plot configurations."""
        if v is None:
            return {}
        if isinstance(v, dict):
            result: dict[str, PlotGroupConfig] = {
                str(name): PlotGroupConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
            return result
        result = dict(v)
        return result

    @model_validator(mode="after")
    def validate_data_names(self) -> "MonetConfig":
        """Validate that pair, plot, and stats references resolve."""
        source_names = set(self.sources)
        pair_names = set(self.pairs)
        analysis_names = set(self.analyses)
        errors: list[str] = []

        for pair_name, pair in self.pairs.items():
            if source_names and pair.x.source not in source_names | analysis_names:
                errors.append(f"pairs.{pair_name}.x.source references unknown source")
            if source_names and pair.y.source not in source_names | analysis_names:
                errors.append(f"pairs.{pair_name}.y.source references unknown source")

        for plot_name, plot in self.plots.items():
            extra = getattr(plot, "__pydantic_extra__", None) or {}

            errors.extend(
                validate_plot_shape(
                    plot_name=plot_name,
                    plot_type=plot.type,
                    pairs=list(plot.pairs),
                    source=plot.source,
                    variable=plot.variable,
                )
            )

            pairs_refs = plot.pairs
            if isinstance(pairs_refs, str):
                pairs_refs = [pairs_refs]
            for ref in pairs_refs:
                if str(ref) not in pair_names:
                    errors.append(f"plots.{plot_name}.pairs references unknown pair '{ref}'")

            source_ref = plot.source
            if source_ref is not None and str(source_ref) not in source_names | analysis_names:
                errors.append(f"plots.{plot_name}.source references unknown source '{source_ref}'")

        if self.stats is not None:
            for ref in self.stats.data:
                if ref not in pair_names:
                    errors.append(f"stats.data references unknown pair '{ref}'")

        # Derived analyses become pseudo-sources; their keys must be unique.
        for name in analysis_names & source_names:
            errors.append(f"analyses.{name} collides with a source of the same name")

        # A valid source reference is a real source OR another analysis output.
        resolvable = source_names | analysis_names
        for a_name, a_spec in self.analyses.items():
            if a_spec.source not in resolvable:
                errors.append(
                    f"analyses.{a_name}.source references unknown source '{a_spec.source}'"
                )

        # Pairs may NOT reference a derived (analysis) source — not pairable.
        for pair_name, pair in self.pairs.items():
            for axis in ("x", "y"):
                ref = getattr(pair, axis).source
                if ref in analysis_names:
                    errors.append(
                        f"pairs.{pair_name}.{axis}.source '{ref}' is a derived analysis "
                        "output; derived sources are not pairable"
                    )

        # Detect cycles in the analysis dependency graph (topological sort).
        state: dict[str, int] = {}  # 0 unvisited, 1 visiting, 2 done

        def _visit(node: str) -> None:
            if state.get(node, 0) == 2:
                return
            if state.get(node, 0) == 1:
                errors.append(f"analyses dependency cycle detected at '{node}'")
                return
            state[node] = 1
            dep = self.analyses[node].source
            if dep in analysis_names:
                _visit(dep)
            state[node] = 2

        for a_name in analysis_names:
            _visit(a_name)

        if errors:
            raise ValueError("; ".join(errors))
        return self


# =============================================================================
# Convenience Aliases
# =============================================================================


Config = MonetConfig  # Alias for common usage
