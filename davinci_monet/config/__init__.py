"""Configuration module for DAVINCI.

This module provides Pydantic-based configuration validation and YAML parsing.
"""

from davinci_monet.config.parser import (
    ConfigBuilder,
    config_to_yaml,
    dump_config,
    load_config,
    load_yaml,
    merge_configs,
    validate_config,
)
from davinci_monet.config.schema import (
    AnalysisConfig,
    AxisRef,
    CheckpointConfig,
    CheckpointRestoreConfig,
    Config,
    DataProcConfig,
    ExecutionConfig,
    MonetConfig,
    PlotGroupConfig,
    PlotStyleConfig,
    SourceConfig,
    SourcePairConfig,
    StatsConfig,
    VariableConfig,
)

__all__ = [
    # Schema classes
    "MonetConfig",
    "Config",
    "AnalysisConfig",
    "CheckpointConfig",
    "CheckpointRestoreConfig",
    "ExecutionConfig",
    "SourceConfig",
    "AxisRef",
    "SourcePairConfig",
    "PlotGroupConfig",
    "PlotStyleConfig",
    "StatsConfig",
    "VariableConfig",
    "DataProcConfig",
    # Parser functions
    "load_config",
    "load_yaml",
    "validate_config",
    "dump_config",
    "config_to_yaml",
    "merge_configs",
    "ConfigBuilder",
]
