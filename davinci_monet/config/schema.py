"""Pydantic schemas for DAVINCI configuration validation."""

from __future__ import annotations

import math
import re
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
    """Base model with strict validation settings."""


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


AnalysisTypeName = Literal[
    "eof",
    "wavelet",
    "aod_preprocess",
    "eof_projection",
    "wavelet_filter",
    "aod_scaling",
    "mmr_writer",
    "known_truth",
    "fable_v2_diagnostics",
    "gridded_analysis",
]


_RUN_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class RequiredArtifactSpec(StrictSchema):
    """One finalized analysis artifact required for production completion."""

    analysis: str = Field(min_length=1)
    role: str = Field(min_length=1)


class CompletionInspectionSpec(StrictSchema):
    """Inspection evidence required before production completion."""

    required: bool = True
    presets: list[str] = Field(min_length=1)


class RunCompletionSpec(StrictSchema):
    """Exact outputs and error policy required for a completed production run."""

    required_analyses: dict[str, AnalysisTypeName] = Field(default_factory=dict)
    required_artifacts: list[RequiredArtifactSpec] = Field(default_factory=list)
    required_saved_files: list[str] = Field(default_factory=list)
    required_plots: list[str] = Field(min_length=1)
    inspection: CompletionInspectionSpec
    allow_item_errors: bool = False

    @field_validator(
        "required_analyses",
        "required_saved_files",
        "required_plots",
    )
    @classmethod
    def _validate_named_contract_items(cls, value: Any) -> Any:
        names = value if isinstance(value, list) else value.keys()
        if any(not str(name).strip() for name in names):
            raise ValueError("completion contract names must not be blank")
        if isinstance(value, list) and len(value) != len(set(value)):
            raise ValueError("completion contract names must be unique")
        return value

    @model_validator(mode="after")
    def _validate_artifact_uniqueness(self) -> "RunCompletionSpec":
        identities = [(artifact.analysis, artifact.role) for artifact in self.required_artifacts]
        if len(identities) != len(set(identities)):
            raise ValueError("required artifacts must be unique by analysis and role")
        if not self.inspection.required:
            raise ValueError("production completion inspection.required must be true")
        return self


class RunConfig(StrictSchema):
    """Scheduled-run identity and optional production completion contract."""

    id: str = Field(min_length=1)
    kind: Literal["production", "preflight", "smoke", "example"]
    completion: RunCompletionSpec | None = None

    @field_validator("id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        if not _RUN_ID_PATTERN.fullmatch(value):
            raise ValueError("run.id must use lowercase kebab-case")
        return value

    @model_validator(mode="after")
    def _validate_kind_contract(self) -> "RunConfig":
        if self.kind == "production":
            if self.completion is None:
                raise ValueError("production runs require run.completion")
            if re.search(r"-r\d{2}$", self.id) is None:
                raise ValueError("production run.id must end in -rNN")
        elif self.completion is not None:
            raise ValueError("only production runs may declare completion")
        return self


class CheckpointRestoreConfig(StrictSchema):
    """Pinned prior-attempt stage boundary used to seed a new attempt."""

    source_attempt_root: Path
    through_stage: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_attempt_root")
    @classmethod
    def _validate_source_attempt_root(cls, value: Path) -> Path:
        if re.fullmatch(r"a\d{3,}", value.name) is None:
            raise ValueError("checkpoint restore source must end in aNNN notation")
        return value


class CheckpointConfig(StrictSchema):
    """Operational checkpoint policy for one pipeline attempt."""

    mode: Literal["required", "best_effort", "off"]
    granularity: Literal["item", "stage"]
    loaded_sources: bool
    retain: Literal["all", "failed", "none"]
    restore_from: CheckpointRestoreConfig | None = None

    @model_validator(mode="after")
    def _validate_restore_policy(self) -> "CheckpointConfig":
        if self.restore_from is not None and self.mode == "off":
            raise ValueError("checkpoint restore requires enabled checkpoints")
        return self


class ExecutionConfig(StrictSchema):
    """Operational execution paths and checkpoint policy."""

    attempt_root: Path
    checkpoints: CheckpointConfig

    @field_validator("attempt_root")
    @classmethod
    def _validate_attempt_root_name(cls, value: Path) -> Path:
        if re.fullmatch(r"a\d{3,}", value.name) is None:
            raise ValueError("execution.attempt_root must end in aNNN notation")
        return value


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
    workflow: Literal["standard", "synthetic_fit", "synthetic_evaluation"] = "standard"

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
            # Handle the legacy hyphenated format: '2019-08-02-12:00:00'
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

    @field_validator("output_dir", "log_dir", mode="before")
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
    style_preset
        Named renderer style preset, e.g. "geosit_aod" for AOD maps.
    levels_plot
        Explicit contour or color-bin boundaries for plots.
    cmap_plot
        Matplotlib colormap name for plots.
    extend_plot
        Colorbar extension mode: "neither", "both", "min", or "max".
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
    style_preset: str | None = None
    levels_plot: list[float] | None = None
    cmap_plot: str | None = None
    extend_plot: Literal["neither", "both", "min", "max"] | None = None
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


def _plotter_supports_artifact(plot_type: str) -> bool:
    """Return whether a registered renderer explicitly opts into ARTIFACT inputs."""
    from davinci_monet.plots.registry import get_plotter_class

    return bool(getattr(get_plotter_class(plot_type), "supports_artifact", False))


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
    time_padding: str | None = None
    evaluation_only: bool = False
    artifact_manifest: str | Path | None = None
    artifact_role: str | None = None
    artifact_analysis: str | None = None

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

    @field_validator("artifact_manifest", mode="before")
    @classmethod
    def _convert_artifact_manifest(cls, value: Any) -> str | None:
        return None if value is None else str(value)

    @field_validator("artifact_role", "artifact_analysis")
    @classmethod
    def _validate_artifact_label(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("artifact role/analysis labels must be nonempty")
        return value

    @model_validator(mode="after")
    def _validate_artifact_identity(self) -> "SourceConfig":
        fields = (self.artifact_manifest, self.artifact_role)
        if any(value is not None for value in fields) and not all(
            isinstance(value, str) and value.strip() for value in fields
        ):
            raise ValueError("artifact_manifest and artifact_role must be configured together")
        if self.artifact_analysis is not None and self.artifact_manifest is None:
            raise ValueError("artifact_analysis requires artifact_manifest and artifact_role")
        return self

    @field_validator("time_padding", mode="before")
    @classmethod
    def _validate_time_padding(cls, value: Any) -> str | None:
        if value is None:
            return None
        import pandas as pd

        try:
            duration = pd.Timedelta(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("time_padding must be a valid duration") from exc
        if pd.isna(duration) or duration < pd.Timedelta(0):
            raise ValueError("time_padding must be non-negative")
        return str(value)

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


def _source_looks_like_oracle(source: SourceConfig) -> bool:
    """Recognize the synthetic truth sidecar without matching ordinary names."""
    raw_paths: list[str] = []
    for raw_value in (source.files, source.filename):
        if isinstance(raw_value, list):
            raw_paths.extend(str(value) for value in raw_value)
        elif raw_value is not None:
            raw_paths.append(str(raw_value))
    has_oracle_component = any(
        "oracle" in [part.lower() for part in path.replace("\\", "/").split("/")]
        for path in raw_paths
    )
    has_truth_variable = any(str(name).endswith("_true") for name in source.variables)
    return has_oracle_component or has_truth_variable


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


class SourcePairConfig(StrictSchema):
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


class PlotSourceRef(StrictSchema):
    """One explicitly named input to a multi-source plot renderer."""

    source: str = Field(min_length=1)
    variable: str = Field(min_length=1)


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
    sources: list[PlotSourceRef] = Field(default_factory=list)
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


class PlotSuiteConfig(StrictSchema):
    """A named plot-suite expansion over one product source."""

    preset: str
    source: str
    group: str | None = None
    output_subdir: str | None = None
    fields: dict[str, str] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)


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


class AnalysisSpecBase(StrictSchema):
    """Shared execution policy for a derived-analysis specification."""

    required: bool = False

    def input_refs(self) -> dict[str, str]:
        """Return input role to raw-or-derived source reference mappings."""
        source = getattr(self, "source", None)
        if not isinstance(source, str) or not source:
            raise NotImplementedError(f"{type(self).__name__} must implement input_refs()")
        return {"source": source}


class EOFFitWindowSpec(StrictSchema):
    """Inclusive time window used to fit EOF preprocessing and modes."""

    start: datetime | str
    end: datetime | str

    @model_validator(mode="after")
    def _validate_order(self) -> "EOFFitWindowSpec":
        import pandas as pd

        try:
            start = pd.Timestamp(self.start)
            end = pd.Timestamp(self.end)
        except (TypeError, ValueError) as exc:
            raise ValueError("EOF fit_window bounds must be valid timestamps") from exc
        try:
            ordered = start <= end
        except TypeError as exc:
            raise ValueError("EOF fit_window bounds must use compatible time zones") from exc
        if not ordered:
            raise ValueError("EOF fit_window start must be at or before end")
        return self


class EOFSpec(AnalysisSpecBase):
    """EOF decomposition of one gridded source variable."""

    type: Literal["eof"]
    source: str
    variable: str
    n_modes: int = Field(default=10, ge=1)
    standardize: bool = False
    remove_seasonal_cycle: bool = False
    rotation: Literal["none", "varimax"] = "none"
    level: int | None = None
    solver: Literal["full", "randomized"] = "full"
    solver_seed: int = Field(default=0, ge=0, le=4_294_967_295)
    solver_oversampling: int = Field(default=10, ge=0)
    solver_iterations: int = Field(default=2, ge=0)
    fit_window: EOFFitWindowSpec | None = None
    fit_artifact: str | None = None
    fit_split: str = Field(default="basis_train", min_length=1)

    def input_refs(self) -> dict[str, str]:
        refs = {"source": self.source}
        if self.fit_artifact is not None:
            refs["fit_artifact"] = self.fit_artifact
        return refs

    @model_validator(mode="after")
    def _validate_fit_selection(self) -> "EOFSpec":
        if self.fit_window is not None and self.fit_artifact is not None:
            raise ValueError("EOF fit_window and fit_artifact are mutually exclusive")
        return self


class WaveletSpec(AnalysisSpecBase):
    """Continuous wavelet transform of one source variable (a 1-D series)."""

    type: Literal["wavelet"]
    source: str
    variable: str
    mode: int | None = Field(default=None, ge=1)
    reduce: Literal["area_mean"] | PointReduce | None = "area_mean"
    omega0: float = Field(default=6.0, gt=0.0)
    significance_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    dj: float = Field(default=0.25, gt=0.0)
    s0: float | None = Field(default=None, gt=0.0)
    j: int | None = Field(default=None, ge=0)

    @field_validator("reduce", mode="before")
    @classmethod
    def _parse_reduce(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return PointReduce(**v)
        return v

    @field_validator("omega0", "significance_level", "dj", "s0")
    @classmethod
    def _validate_finite_controls(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("wavelet numeric controls must be finite")
        return value


class LinearAODUncertaintySpec(StrictSchema):
    """Explicit linear-AOD uncertainty model transformed for projection."""

    type: Literal["linear_aod_rss"]
    name: str = Field(min_length=1)
    source_variable: str = Field(min_length=1)
    absolute_floor: float = Field(gt=0.0)
    relative_fraction: float = Field(default=0.0, ge=0.0)
    combination: Literal["root_sum_square"] = "root_sum_square"
    transform: Literal["delta_method"] = "delta_method"
    covariance: Literal["independent"] = "independent"

    @field_validator("absolute_floor", "relative_fraction")
    @classmethod
    def _validate_finite_uncertainty_controls(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("uncertainty model controls must be finite")
        return value


class AODPreprocessSpec(AnalysisSpecBase):
    """Screen, daily-sample, regrid, and shifted-log transform one AOD field."""

    type: Literal["aod_preprocess"]
    source: str
    variable: str
    target_grid: float | None = Field(default=None, gt=0.0, le=180.0)
    target_grid_from: str | None = None
    sample_local_time: float | None = None
    sample_tolerance: str | None = None
    day_anchor_hour: float = 12.0
    log_epsilon: float = Field(default=0.01, gt=0.0)
    uncertainty_variable: str | None = None
    uncertainty_covariance: Literal["independent"] | None = None
    uncertainty_model: LinearAODUncertaintySpec | None = None
    common_factor_variables: list[str] = Field(default_factory=list)

    def input_refs(self) -> dict[str, str]:
        refs = {"source": self.source}
        if self.target_grid_from is not None:
            refs["target_grid_from"] = self.target_grid_from
        return refs

    @field_validator("sample_local_time", "day_anchor_hour")
    @classmethod
    def _validate_hour(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not 0.0 <= value < 24.0):
            raise ValueError("hour values must be finite and in [0, 24)")
        return value

    @field_validator("log_epsilon")
    @classmethod
    def _validate_log_epsilon(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("log_epsilon must be finite")
        return value

    @field_validator("sample_tolerance")
    @classmethod
    def _validate_sample_tolerance(cls, value: str | None) -> str | None:
        if value is None:
            return None
        import pandas as pd

        try:
            duration = pd.Timedelta(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_tolerance must be a valid duration") from exc
        if pd.isna(duration) or duration < pd.Timedelta(0):
            raise ValueError("sample_tolerance must be non-negative")
        return value

    @field_validator("common_factor_variables")
    @classmethod
    def _validate_common_factors(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("common_factor_variables must be unique")
        return value

    @model_validator(mode="after")
    def _validate_target_grid(self) -> "AODPreprocessSpec":
        if self.target_grid is not None and self.target_grid_from is not None:
            raise ValueError("target_grid and target_grid_from are mutually exclusive")
        if self.uncertainty_model is not None and (
            self.uncertainty_variable is not None or self.uncertainty_covariance is not None
        ):
            raise ValueError(
                "uncertainty_model is mutually exclusive with uncertainty_variable "
                "and uncertainty_covariance"
            )
        if self.uncertainty_covariance is not None and self.uncertainty_variable is None:
            raise ValueError("uncertainty_covariance requires uncertainty_variable")
        if self.target_grid is not None:
            latitude_cells = 180.0 / self.target_grid
            longitude_cells = 360.0 / self.target_grid
            if not math.isclose(latitude_cells, round(latitude_cells), abs_tol=1.0e-9):
                raise ValueError("target_grid must divide 180 degrees exactly")
            if not math.isclose(longitude_cells, round(longitude_cells), abs_tol=1.0e-9):
                raise ValueError("target_grid must divide 360 degrees exactly")
        return self


class BiasFitWindowSpec(StrictSchema):
    """Inclusive time window used only to fit projection bias and support."""

    start: datetime | str
    end: datetime | str

    @model_validator(mode="after")
    def _validate_order(self) -> "BiasFitWindowSpec":
        import pandas as pd

        try:
            start = pd.Timestamp(self.start)
            end = pd.Timestamp(self.end)
        except (TypeError, ValueError) as exc:
            raise ValueError("bias_fit_window bounds must be valid timestamps") from exc
        if start > end:
            raise ValueError("bias_fit_window start must be at or before end")
        return self


class EOFProjectionObsSpec(StrictSchema):
    """One preprocessed observation input to an EOF projection."""

    source: str
    variable: str
    error_variable: str
    common_factor_variables: list[str] = Field(default_factory=list)

    @field_validator("common_factor_variables")
    @classmethod
    def _validate_common_factors(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("common_factor_variables must be unique")
        return value


class EOFProjectionSpec(AnalysisSpecBase):
    """Reduced-space ridge projection of AOD innovations onto an EOF basis."""

    type: Literal["eof_projection"]
    basis: str
    model: str
    model_variable: str
    obs: list[EOFProjectionObsSpec] = Field(min_length=1)
    ridge: float = Field(default=1.0, ge=0.0)
    bias_fit_window: BiasFitWindowSpec | None = None
    bias_fit_artifact: str | None = None
    bias_fit_method: Literal["monthly_mean", "joint_seasonal"] = "monthly_mean"
    sensor_offset_method: Literal["none", "overlap_zero_sum"] = "none"
    joint_bias_laplacian_strength: float = Field(default=1.0, ge=0.0)
    joint_bias_tolerance: float = Field(default=1.0e-6, gt=0.0)
    joint_bias_max_iterations: int = Field(default=20, ge=1)
    clim_bias: bool = True
    spatial_support: Literal["monthly_taper"] = "monthly_taper"
    support_min_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    support_full_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    support_smoothing_passes: int = Field(default=2, ge=0)
    delta_bounds: tuple[float, float] = (-1.6094379, 1.6094379)
    min_resolution: float = Field(default=0.3, ge=0.0, lt=1.0)
    time_chunk_size: int = Field(default=31, ge=1)

    def input_refs(self) -> dict[str, str]:
        refs = {"basis": self.basis, "model": self.model}
        refs.update({f"obs[{index}]": entry.source for index, entry in enumerate(self.obs)})
        if self.bias_fit_artifact is not None:
            refs["bias_fit_artifact"] = self.bias_fit_artifact
        return refs

    @field_validator(
        "ridge",
        "joint_bias_laplacian_strength",
        "joint_bias_tolerance",
    )
    @classmethod
    def _validate_projection_float(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("projection numeric controls must be finite")
        return value

    @field_validator("delta_bounds")
    @classmethod
    def _validate_delta_bounds(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not all(math.isfinite(bound) for bound in value) or value[0] >= value[1]:
            raise ValueError("delta_bounds must be finite and increasing")
        return value

    @model_validator(mode="after")
    def _validate_fit_and_support(self) -> "EOFProjectionSpec":
        if self.bias_fit_window is not None and self.bias_fit_artifact is not None:
            raise ValueError("bias_fit_window and bias_fit_artifact are mutually exclusive")
        if self.bias_fit_window is None and self.bias_fit_artifact is None:
            raise ValueError("eof_projection requires bias_fit_window or bias_fit_artifact")
        if self.support_min_fraction >= self.support_full_fraction:
            raise ValueError("support_min_fraction must be below support_full_fraction")
        if self.bias_fit_method == "monthly_mean" and self.sensor_offset_method != "none":
            raise ValueError(
                "sensor_offset_method='overlap_zero_sum' requires "
                "bias_fit_method='joint_seasonal'"
            )
        return self


class PeriodBandSpec(StrictSchema):
    """Finite daily period band retained by the segmented wavelet filter."""

    min: float = Field(gt=0.0)
    max: float = Field(gt=0.0)
    units: Literal["days"] = "days"

    @model_validator(mode="after")
    def _validate_order(self) -> "PeriodBandSpec":
        if not math.isfinite(self.min) or not math.isfinite(self.max) or self.min >= self.max:
            raise ValueError("wavelet band limits must be finite and increasing")
        return self


class WaveletFilterSpec(AnalysisSpecBase):
    """Bounded-gap, significance-gated CWT filter for projected EOF modes."""

    type: Literal["wavelet_filter"]
    source: str
    variable: str = "pc"
    resolution_variable: str = "resolution"
    min_resolution: float = Field(default=0.3, ge=0.0, lt=1.0)
    keep_significant: bool = True
    significance_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    band: PeriodBandSpec
    max_bridge_days: float = Field(default=7.0, ge=0.0)
    min_segment_days: float = Field(gt=0.0)
    omega0: float = Field(default=6.0, gt=0.0)
    dj: float = Field(default=0.25, gt=0.0)
    s0: float | None = Field(default=None, gt=0.0)

    @field_validator(
        "significance_level",
        "max_bridge_days",
        "min_segment_days",
        "omega0",
        "dj",
        "s0",
    )
    @classmethod
    def _validate_finite_controls(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("wavelet filter numeric controls must be finite")
        return value

    @model_validator(mode="after")
    def _validate_segment_length(self) -> "WaveletFilterSpec":
        if self.min_segment_days < 2.0 * self.band.max:
            raise ValueError("min_segment_days must be at least twice band.max")
        return self


class AODScalingSpec(AnalysisSpecBase):
    """Reconstruct and safely convert filtered log-AOD corrections to ratios."""

    type: Literal["aod_scaling"]
    basis: str
    projection: str
    coefficients: str
    model: str
    basis_variable: str = "eofs"
    bias_variable: str = "clim_bias_applied"
    support_variable: str = "spatial_support"
    coefficients_variable: str = "pc"
    model_variable: str = "aod"
    r_bounds: tuple[float, float] = (0.2, 5.0)
    aod_floor: float = Field(default=0.001, ge=0.0)
    time_chunk_days: int = Field(default=31, ge=1)

    def input_refs(self) -> dict[str, str]:
        return {
            "basis": self.basis,
            "projection": self.projection,
            "coefficients": self.coefficients,
            "model": self.model,
        }

    @field_validator("r_bounds")
    @classmethod
    def _validate_ratio_bounds(cls, value: tuple[float, float]) -> tuple[float, float]:
        if (
            not all(math.isfinite(bound) for bound in value)
            or value[0] <= 0.0
            or value[0] >= value[1]
        ):
            raise ValueError("r_bounds must be finite, positive, and increasing")
        return value

    @field_validator("aod_floor")
    @classmethod
    def _validate_aod_floor(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("aod_floor must be finite")
        return value


class MMRWriterSpec(AnalysisSpecBase):
    """Atomic per-file application of a scaling artifact to aerosol MMR fields."""

    type: Literal["mmr_writer"]
    scaling: str
    files: str = Field(min_length=1)
    species: list[str] | None = None
    output_dir: str = Field(min_length=1)
    time_interp: Literal["log_linear"] = "log_linear"
    outside_coverage: Literal["identity", "skip", "error"] = "identity"
    overwrite: bool = False
    resume: bool = False
    required: Literal[True] = True

    def input_refs(self) -> dict[str, str]:
        return {"scaling": self.scaling}

    @field_validator("species")
    @classmethod
    def _validate_species(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value or any(not species.strip() for species in value):
            raise ValueError("species must be a nonempty list of names")
        if len(value) != len(set(value)):
            raise ValueError("species names must be unique")
        return value


class KnownTruthSpec(AnalysisSpecBase):
    """Read-only synthetic recovery metrics against an explicit oracle input."""

    type: Literal["known_truth"]
    estimate: str
    truth: str
    estimate_delta_variable: str = "delta_log_applied"
    truth_delta_variable: str = "delta_filter_target_true"
    full_truth_delta_variable: str | None = "delta_applied_true"
    in_span_truth_delta_variable: str | None = "delta_in_span_true"
    perpendicular_truth_delta_variable: str | None = "delta_perp_true"
    estimate_aod_variable: str = "aod_target"
    truth_aod_variable: str = "aod_filter_target_true"
    full_truth_aod_variable: str | None = "aod_target_applied_true"
    model_aod_variable: str = "model_aod_overpass_true"
    estimate_basis_variable: str | None = "eofs"
    truth_basis_variable: str | None = "pattern_true"
    estimate_coefficient_variable: str | None = "pc"
    truth_coefficient_variable: str | None = "correction_pc_filter_target_true"
    support_variable: str | None = "spatial_support"
    resolution_variable: str | None = "resolution"
    observable_mode_variable: str | None = "mode_observable_true"
    primary_mask_variable: str | None = None
    split_variable: str | None = "split"
    evaluation_splits: list[str] = Field(default_factory=lambda: ["development_test"])
    best_representable_variable: str | None = "delta_best_representable_true"
    required: bool = True

    def input_refs(self) -> dict[str, str]:
        return {"estimate": self.estimate, "truth": self.truth}

    @field_validator("evaluation_splits")
    @classmethod
    def _validate_evaluation_splits(cls, value: list[str]) -> list[str]:
        if not value or any(not split.strip() for split in value):
            raise ValueError("evaluation_splits must contain at least one nonempty name")
        if len(value) != len(set(value)):
            raise ValueError("evaluation_splits must be unique")
        return value

    @model_validator(mode="after")
    def _validate_split_contract(self) -> "KnownTruthSpec":
        if self.evaluation_splits and self.split_variable is None:
            raise ValueError("split_variable is required when evaluation_splits are requested")
        return self


class FableV2DiagnosticsSpec(AnalysisSpecBase):
    """Evaluation-only stage decomposition for the FABLE v2 synthetic cycle."""

    type: Literal["fable_v2_diagnostics"]
    estimate: str
    projection: str
    truth: str
    projection_to_truth_sensor: dict[str, str] = Field(default_factory=dict)
    reported_common_factor_amplitude: float = Field(ge=0.0)
    evaluation_splits: list[str] = Field(default_factory=lambda: ["development_test"])
    required: bool = True

    def input_refs(self) -> dict[str, str]:
        return {
            "estimate": self.estimate,
            "projection": self.projection,
            "truth": self.truth,
        }

    @field_validator("evaluation_splits")
    @classmethod
    def _validate_evaluation_splits(cls, value: list[str]) -> list[str]:
        if not value or any(not split.strip() for split in value):
            raise ValueError("evaluation_splits must contain at least one nonempty name")
        if len(value) != len(set(value)):
            raise ValueError("evaluation_splits must be unique")
        return value

    @field_validator("projection_to_truth_sensor", mode="before")
    @classmethod
    def _validate_sensor_mapping(cls, value: Any) -> Any:
        if not isinstance(value, dict) or any(
            not isinstance(source, str) or not isinstance(target, str)
            for source, target in value.items()
        ):
            return value
        if any(
            not source or not target or source != source.strip() or target != target.strip()
            for source, target in value.items()
        ):
            raise ValueError(
                "projection_to_truth_sensor names must be nonempty and whitespace-trimmed"
            )
        if len(value.values()) != len(set(value.values())):
            raise ValueError("projection_to_truth_sensor must be bijective")
        return value

    @field_validator("reported_common_factor_amplitude")
    @classmethod
    def _validate_common_factor_amplitude(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reported_common_factor_amplitude must be finite")
        return value


class FormulaFieldSpec(StrictSchema):
    """One named field produced by a gridded analysis formula."""

    formula: str
    units: str | None = None
    long_name: str | None = None
    display_name: str | None = None
    style_preset: str | None = None


class CustomWindowSpec(StrictSchema):
    """Named inclusive time window for grouped products."""

    name: str
    start: datetime | str
    end: datetime | str


class GriddedAnalysisSpec(AnalysisSpecBase):
    """Role/formula-driven gridded analysis product."""

    type: Literal["gridded_analysis"]
    source: str
    groupby: Literal["day", "month", "season", "all"] | list[CustomWindowSpec] = "all"
    roles: dict[str, str]
    fields: dict[str, FormulaFieldSpec]
    output_group: str | None = None

    @field_validator("roles")
    @classmethod
    def _roles_nonempty(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("gridded_analysis roles must not be empty")
        return value

    @field_validator("fields")
    @classmethod
    def _fields_nonempty(cls, value: dict[str, FormulaFieldSpec]) -> dict[str, FormulaFieldSpec]:
        if not value:
            raise ValueError("gridded_analysis fields must not be empty")
        return value


AnalysisSpec = (
    EOFSpec
    | WaveletSpec
    | AODPreprocessSpec
    | EOFProjectionSpec
    | WaveletFilterSpec
    | AODScalingSpec
    | MMRWriterSpec
    | KnownTruthSpec
    | FableV2DiagnosticsSpec
    | GriddedAnalysisSpec
)


def build_analysis_spec(cfg: Any) -> AnalysisSpec:
    """Build the right AnalysisSpec submodel from a dict, dispatching on type."""
    if isinstance(
        cfg,
        (
            EOFSpec,
            WaveletSpec,
            AODPreprocessSpec,
            EOFProjectionSpec,
            WaveletFilterSpec,
            AODScalingSpec,
            MMRWriterSpec,
            KnownTruthSpec,
            FableV2DiagnosticsSpec,
            GriddedAnalysisSpec,
        ),
    ):
        return cfg
    if not isinstance(cfg, dict):
        raise ValueError(f"analysis entry must be a mapping, got {type(cfg).__name__}")
    analysis_type = cfg.get("type")
    if analysis_type == "eof":
        return EOFSpec(**cfg)
    if analysis_type == "wavelet":
        return WaveletSpec(**cfg)
    if analysis_type == "aod_preprocess":
        return AODPreprocessSpec(**cfg)
    if analysis_type == "eof_projection":
        return EOFProjectionSpec(**cfg)
    if analysis_type == "wavelet_filter":
        return WaveletFilterSpec(**cfg)
    if analysis_type == "aod_scaling":
        return AODScalingSpec(**cfg)
    if analysis_type == "mmr_writer":
        return MMRWriterSpec(**cfg)
    if analysis_type == "known_truth":
        return KnownTruthSpec(**cfg)
    if analysis_type == "fable_v2_diagnostics":
        return FableV2DiagnosticsSpec(**cfg)
    if analysis_type == "gridded_analysis":
        return GriddedAnalysisSpec(**cfg)
    raise ValueError(
        "Unknown analysis type "
        f"{analysis_type!r}. Available analysis types: eof, wavelet, "
        "aod_preprocess, eof_projection, wavelet_filter, aod_scaling, "
        "mmr_writer, known_truth, fable_v2_diagnostics, gridded_analysis"
    )


# =============================================================================
# Root Configuration
# =============================================================================


class InspectionConfig(StrictSchema):
    """Optional visual-inspection configuration."""

    enabled: bool = False
    required: bool = False
    presets: list[str] = Field(default_factory=list)
    preview_format: Literal["png"] = "png"


class MonetConfig(StrictSchema):
    """Root configuration model for DAVINCI.

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

    run: RunConfig | None = None
    execution: ExecutionConfig | None = None
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    # Data sources keyed by dataset label.
    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    pairs: dict[str, SourcePairConfig] = Field(default_factory=dict)
    pairing: PipelinePairingConfig | None = None
    plots: dict[str, PlotGroupConfig] = Field(default_factory=dict)
    plot_suites: dict[str, PlotSuiteConfig] = Field(default_factory=dict)
    analyses: dict[str, AnalysisSpec] = Field(default_factory=dict)
    stats: StatsConfig | None = None
    summary: SummaryConfig | None = None
    inspection: InspectionConfig | None = None

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

    @field_validator("plot_suites", mode="before")
    @classmethod
    def parse_plot_suites(cls, value: Any) -> dict[str, PlotSuiteConfig]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {
                str(name): PlotSuiteConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in value.items()
            }
        return dict(value)

    @field_validator("inspection", mode="before")
    @classmethod
    def parse_inspection(cls, value: Any) -> InspectionConfig | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return InspectionConfig(**value)
        return value

    @model_validator(mode="after")
    def validate_data_names(self) -> "MonetConfig":
        """Validate that pair, plot, and stats references resolve."""
        source_names = set(self.sources)
        pair_names = set(self.pairs)
        analysis_names = set(self.analyses)
        artifact_analysis_names = {
            name
            for name, spec in self.analyses.items()
            if isinstance(spec, (MMRWriterSpec, KnownTruthSpec, FableV2DiagnosticsSpec))
        }
        errors: list[str] = []

        completion = self.run.completion if self.run is not None else None
        if self.run is not None and self.run.kind == "production":
            if self.execution is None:
                errors.append("production runs require execution checkpoint policy")
            else:
                checkpoints = self.execution.checkpoints
                if checkpoints.mode != "required":
                    errors.append("production checkpoint mode must be 'required'")
                if checkpoints.granularity != "item":
                    errors.append("production checkpoint granularity must be 'item'")
                if not checkpoints.loaded_sources:
                    errors.append("production requires loaded-source checkpoints")
                if checkpoints.retain != "all":
                    errors.append("production checkpoint retention must be 'all'")

        if self.execution is not None:
            output_dir = self.analysis.output_dir
            log_dir = self.analysis.log_dir
            if output_dir is None or log_dir is None:
                errors.append("execution requires analysis.output_dir and analysis.log_dir")
            else:
                attempt_root = self.execution.attempt_root.expanduser().resolve()
                output_parent = Path(output_dir).expanduser().resolve().parent
                log_parent = Path(log_dir).expanduser().resolve().parent
                if attempt_root != output_parent or attempt_root != log_parent:
                    errors.append(
                        "execution.attempt_root must be the common parent of "
                        "analysis.output_dir and analysis.log_dir"
                    )

        if completion is not None:
            if self.analysis.output_dir is None:
                errors.append("production runs require analysis.output_dir")
            if self.analysis.log_dir is None:
                errors.append("production runs require analysis.log_dir")
            for required_name, required_type in completion.required_analyses.items():
                declared = self.analyses.get(required_name)
                if declared is None:
                    errors.append(
                        "run.completion requires "
                        f"analyses.{required_name} with type {required_type!r}"
                    )
                elif declared.type != required_type:
                    errors.append(
                        "run.completion requires "
                        f"analyses.{required_name}.type={required_type!r}, "
                        f"got {declared.type!r}"
                    )
                elif not declared.required:
                    errors.append(f"run.completion requires analyses.{required_name}.required=true")
            for artifact in completion.required_artifacts:
                if artifact.analysis not in completion.required_analyses:
                    errors.append(
                        "run.completion.required_artifacts "
                        f"references analysis '{artifact.analysis}' that is not required"
                    )
            for plot_name in completion.required_plots:
                if plot_name not in self.plots:
                    errors.append(
                        f"run.completion.required_plots references unknown plot '{plot_name}'"
                    )
            if self.inspection is None or not self.inspection.enabled:
                errors.append("production run completion requires inspection.enabled=true")
            elif not self.inspection.required:
                errors.append("production run completion requires inspection.required=true")
            elif set(self.inspection.presets) != set(completion.inspection.presets):
                errors.append(
                    "inspection.presets must exactly match run.completion.inspection.presets"
                )

        fit_types = {
            "aod_preprocess",
            "eof",
            "eof_projection",
            "wavelet_filter",
            "aod_scaling",
            "mmr_writer",
        }
        fit_analyses = {name for name, spec in self.analyses.items() if spec.type in fit_types}
        truth_sources = {
            name
            for name, source in self.sources.items()
            if source.evaluation_only or _source_looks_like_oracle(source)
        }
        known_truth = {
            name: spec for name, spec in self.analyses.items() if isinstance(spec, KnownTruthSpec)
        }
        v2_diagnostics = {
            name: spec
            for name, spec in self.analyses.items()
            if isinstance(spec, FableV2DiagnosticsSpec)
        }
        evaluation_analyses = {**known_truth, **v2_diagnostics}

        if known_truth and self.analysis.workflow != "synthetic_evaluation":
            errors.append("known_truth requires analysis.workflow: synthetic_evaluation")
        if v2_diagnostics and self.analysis.workflow != "synthetic_evaluation":
            errors.append("fable_v2_diagnostics requires analysis.workflow: synthetic_evaluation")
        if self.analysis.workflow == "synthetic_fit" and fit_analyses and truth_sources:
            names = ", ".join(sorted(truth_sources))
            errors.append(f"synthetic fitting analyses cannot load oracle truth sources: {names}")
        if self.analysis.workflow == "synthetic_fit":
            if evaluation_analyses:
                errors.append("synthetic_fit cannot contain truth-evaluation analyses")
            if truth_sources:
                errors.append("synthetic_fit sources may reference only fitting inputs")
        if self.analysis.workflow == "synthetic_evaluation":
            disallowed = sorted(
                name
                for name, spec in self.analyses.items()
                if not isinstance(spec, (KnownTruthSpec, FableV2DiagnosticsSpec))
            )
            if disallowed:
                errors.append(
                    "synthetic_evaluation may contain only known_truth or "
                    "fable_v2_diagnostics analyses: " + ", ".join(disallowed)
                )
            if not known_truth:
                errors.append("synthetic_evaluation requires at least one known_truth analysis")
            for name, source in self.sources.items():
                if name in truth_sources:
                    continue
                if source.artifact_manifest is None or source.artifact_role is None:
                    errors.append(
                        f"sources.{name} must be a finalized manifest-validated artifact "
                        "in synthetic_evaluation"
                    )
            for name, spec in known_truth.items():
                estimate = self.sources.get(spec.estimate)
                truth = self.sources.get(spec.truth)
                if estimate is not None and (
                    spec.estimate in truth_sources
                    or estimate.artifact_manifest is None
                    or estimate.artifact_role is None
                ):
                    errors.append(
                        f"analyses.{name}.estimate must reference a finalized fit artifact"
                    )
                if truth is not None and not truth.evaluation_only:
                    errors.append(
                        f"analyses.{name}.truth must reference an evaluation-only truth source"
                    )
            for name, diagnostic_spec in v2_diagnostics.items():
                for role in ("estimate", "projection"):
                    source_name = str(getattr(diagnostic_spec, role))
                    diagnostic_source = self.sources.get(source_name)
                    if diagnostic_source is not None and (
                        source_name in truth_sources
                        or diagnostic_source.artifact_manifest is None
                        or diagnostic_source.artifact_role is None
                    ):
                        errors.append(
                            f"analyses.{name}.{role} must reference a finalized fit artifact"
                        )
                diagnostic_truth = self.sources.get(diagnostic_spec.truth)
                if diagnostic_truth is not None and not diagnostic_truth.evaluation_only:
                    errors.append(
                        f"analyses.{name}.truth must reference an evaluation-only truth source"
                    )

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
                    sources=list(plot.sources),
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
            if (
                source_ref is not None
                and str(source_ref) in artifact_analysis_names
                and not _plotter_supports_artifact(plot.type)
            ):
                errors.append(
                    f"plots.{plot_name}.source '{source_ref}' is an ARTIFACT analysis output "
                    f"unsupported by plot type '{plot.type}'"
                )
            for input_ref in plot.sources:
                if input_ref.source not in source_names | analysis_names:
                    errors.append(
                        f"plots.{plot_name}.sources references unknown source "
                        f"'{input_ref.source}'"
                    )
                if input_ref.source in artifact_analysis_names and not _plotter_supports_artifact(
                    plot.type
                ):
                    errors.append(
                        f"plots.{plot_name}.sources source '{input_ref.source}' is an ARTIFACT "
                        f"analysis output unsupported by plot type '{plot.type}'"
                    )

        for suite_name, suite in self.plot_suites.items():
            if suite.source in artifact_analysis_names:
                errors.append(
                    f"plot_suites.{suite_name}.source '{suite.source}' is an ARTIFACT "
                    "analysis output"
                )

        if self.stats is not None:
            for ref in self.stats.data:
                if ref not in pair_names:
                    errors.append(f"stats.data references unknown pair '{ref}'")

        # Derived analyses become pseudo-sources; their keys must be unique.
        for name in analysis_names & source_names:
            errors.append(f"analyses.{name} collides with a source of the same name")

        # Every named input may reference a real source or another analysis output.
        resolvable = source_names | analysis_names

        def require_saved_fit_source(owner: str, ref: str, expected_role: str) -> None:
            source = self.sources.get(ref)
            if source is None:
                return
            if (
                source.artifact_manifest is None
                or source.artifact_role != expected_role
                or source.artifact_analysis is None
            ):
                errors.append(
                    f"analyses.{owner} saved fit source '{ref}' must configure "
                    f"artifact_manifest, artifact_role: {expected_role}, and artifact_analysis"
                )

        for a_name, a_spec in self.analyses.items():
            for role, ref in a_spec.input_refs().items():
                if ref not in resolvable:
                    errors.append(f"analyses.{a_name}.{role} references unknown source '{ref}'")
            if isinstance(a_spec, EOFProjectionSpec):
                if a_spec.bias_fit_artifact is not None:
                    require_saved_fit_source(a_name, a_spec.bias_fit_artifact, "projection_fit")
                if self.analysis.workflow == "synthetic_fit" and a_spec.basis in source_names:
                    require_saved_fit_source(a_name, a_spec.basis, "basis_fit")
                basis = self.analyses.get(a_spec.basis)
                if isinstance(basis, EOFSpec):
                    if basis.standardize:
                        errors.append(
                            f"analyses.{a_name}.basis '{a_spec.basis}' must use "
                            "standardize=false"
                        )
                    if basis.rotation != "none":
                        errors.append(
                            f"analyses.{a_name}.basis '{a_spec.basis}' must use rotation=none"
                        )
            if isinstance(a_spec, EOFSpec) and a_spec.fit_artifact is not None:
                require_saved_fit_source(a_name, a_spec.fit_artifact, "basis_fit")
            if isinstance(a_spec, WaveletFilterSpec):
                projection = self.analyses.get(a_spec.source)
                if isinstance(projection, EOFProjectionSpec) and not math.isclose(
                    a_spec.min_resolution,
                    projection.min_resolution,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                ):
                    errors.append(
                        f"analyses.{a_name}.min_resolution must match "
                        f"analyses.{a_spec.source}.min_resolution"
                    )

        # Physical gridded products may be evaluated in the same run after the
        # analyses stage. Mode/spectrum/artifact products remain non-pairable.
        pairable_analysis_types = {"aod_preprocess", "aod_scaling", "gridded_analysis"}
        for pair_name, pair in self.pairs.items():
            for axis in ("x", "y"):
                ref = getattr(pair, axis).source
                analysis = self.analyses.get(ref)
                if analysis is not None and analysis.type not in pairable_analysis_types:
                    errors.append(
                        f"pairs.{pair_name}.{axis}.source '{ref}' is a derived analysis "
                        f"output of type '{analysis.type}'; derived sources are not pairable"
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
            for dep in self.analyses[node].input_refs().values():
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
