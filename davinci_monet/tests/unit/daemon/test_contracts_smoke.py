"""Smoke test: every daemon runtime primitive imports from contracts."""

from __future__ import annotations


def test_contracts_primitives_importable() -> None:
    from davinci_monet.daemon.contracts import (  # noqa: F401
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
    from davinci_monet.daemon.contracts import (  # noqa: F401
        COMMANDS,
        PROTOCOL_VERSION,
        SCHEMA_DDL,
        STREAMING_COMMANDS,
    )

    # The runtime enum + literal aliases are present.
    assert JobStatus.QUEUED.value == "queued"
    assert PROTOCOL_VERSION == 1
    assert "subscribe" in STREAMING_COMMANDS


def test_contracts_does_not_own_config_models() -> None:
    """The config models live in daemon.config, NOT contracts."""
    import davinci_monet.daemon.contracts as contracts

    for forbidden in ("WatchRule", "DaemonConfig", "NotificationConfig", "WatchesFile"):
        assert not hasattr(contracts, forbidden), (
            f"{forbidden} must be owned by daemon.config, not contracts"
        )
