"""Smoke test: every daemon runtime primitive imports from contracts."""

from __future__ import annotations


def test_contracts_primitives_importable() -> None:
    from davinci_monet.daemon.contracts import (  # noqa: F401
        COMMANDS,
        PROTOCOL_VERSION,
        SCHEMA_DDL,
        STREAMING_COMMANDS,
        Clock,
        ControlHandler,
        ControlRequest,
        ControlResponse,
        JobRecord,
        JobSpec,
        JobStatus,
        ProgressEvent,
        SettleMode,
        StreamEvent,
        TriggerEvent,
        WatchStatusRecord,
    )

    # The runtime enum + literal aliases are present.
    assert JobStatus.QUEUED.value == "queued"
    assert PROTOCOL_VERSION == 1
    assert "subscribe" in STREAMING_COMMANDS


def test_contracts_does_not_own_config_models() -> None:
    """The config models live in daemon.config, NOT contracts."""
    import davinci_monet.daemon.contracts as contracts

    for forbidden in ("WatchRule", "DaemonConfig", "NotificationConfig", "WatchesFile"):
        assert not hasattr(
            contracts, forbidden
        ), f"{forbidden} must be owned by daemon.config, not contracts"


def test_config_package_exports_strict_flexible_model() -> None:
    """StrictModel and FlexibleModel must be importable from davinci_monet.config at runtime.

    The .pyi stub advertises them; the runtime __getattr__ must back that up.
    Regression guard for the stub/runtime divergence found in the Task 1 review.
    """
    # These must NOT raise ImportError at runtime.
    from davinci_monet.config import FlexibleModel, StrictModel  # noqa: F401

    # Sanity: they are the real Pydantic base classes.
    from davinci_monet.config.schema import FlexibleModel as SchemaFlexible
    from davinci_monet.config.schema import StrictModel as SchemaStrict

    assert StrictModel is SchemaStrict
    assert FlexibleModel is SchemaFlexible


def test_state_store_docstring_no_stale_constructor_reference() -> None:
    """StateStore docstring must not reference a constructor the Protocol no longer declares.

    The old text 'Opens/creates the DB ... on construction.' was left after
    __init__(self, db_path) was removed during the Protocol conversion.  Conforming
    implementations should still do that work, but the Protocol docstring should
    not imply a constructor it doesn't declare.
    """
    from davinci_monet.daemon.contracts import StateStore

    doc = StateStore.__doc__ or ""
    # The stale phrase tied the sentence to a concrete constructor.  It should be gone.
    assert "Opens/creates the DB" not in doc, (
        "StateStore docstring still contains the stale constructor phrase "
        "'Opens/creates the DB ... on construction.' — remove or rephrase it."
    )
