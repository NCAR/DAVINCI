"""Configuration module for DAVINCI.

This module provides Pydantic-based configuration validation,
YAML parsing, and migration utilities for MELODIES-MONET configs.

Imports are lazy (PEP 562 __getattr__) so that importing a submodule such as
``davinci_monet.config.schema`` does NOT eagerly pull in the parser/migration
modules (which drag in the heavy geo/data stack).  Callers that use the package
namespace directly (``from davinci_monet.config import load_config``) still work
identically — the attribute is resolved on first access.
"""

from __future__ import annotations

__all__ = [
    # Schema classes
    "MonetConfig",
    "Config",
    "AnalysisConfig",
    "ModelConfig",
    "ObservationConfig",
    "SourceConfig",
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
    # Migration
    "migrate_config",
    "migrate_to_sources",
    "expand_sources_to_legacy",
    "LegacyConfigWarning",
    "detect_config_version",
    "ConfigMigration",
    "CURRENT_VERSION",
]

# ---------------------------------------------------------------------------
# PEP 562 lazy loader — defers heavy submodule imports until first attribute
# access on this package.  This prevents `import davinci_monet.config.schema`
# (used by the daemon contracts to access StrictModel/FlexibleModel) from
# accidentally triggering the parser/migration heavy-import chain.
# ---------------------------------------------------------------------------

_SCHEMA_ATTRS = {
    "MonetConfig",
    "Config",
    "AnalysisConfig",
    "ModelConfig",
    "ObservationConfig",
    "SourceConfig",
    "SourcePairConfig",
    "PlotGroupConfig",
    "PlotStyleConfig",
    "StatsConfig",
    "VariableConfig",
    "DataProcConfig",
}

_PARSER_ATTRS = {
    "load_config",
    "load_yaml",
    "validate_config",
    "dump_config",
    "config_to_yaml",
    "merge_configs",
    "ConfigBuilder",
}

_MIGRATION_ATTRS = {
    "migrate_config",
    "migrate_to_sources",
    "expand_sources_to_legacy",
    "LegacyConfigWarning",
    "detect_config_version",
    "ConfigMigration",
    "CURRENT_VERSION",
}


def __getattr__(name: str) -> object:
    if name in _SCHEMA_ATTRS:
        from davinci_monet.config import schema as _schema

        return getattr(_schema, name)
    if name in _PARSER_ATTRS:
        from davinci_monet.config import parser as _parser

        return getattr(_parser, name)
    if name in _MIGRATION_ATTRS:
        from davinci_monet.config import migration as _migration

        return getattr(_migration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
