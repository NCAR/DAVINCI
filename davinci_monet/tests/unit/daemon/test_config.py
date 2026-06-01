"""Unit tests for davinci_monet.daemon.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from davinci_monet.daemon.config import NotificationConfig, parse_duration


class TestParseDuration:
    def test_none_returns_none(self) -> None:
        assert parse_duration(None) is None

    def test_bare_int_is_seconds(self) -> None:
        assert parse_duration(30) == 30.0

    def test_bare_float_is_seconds(self) -> None:
        assert parse_duration(1.5) == 1.5

    def test_unsuffixed_string_is_seconds(self) -> None:
        assert parse_duration("45") == 45.0

    def test_seconds_suffix(self) -> None:
        assert parse_duration("30s") == 30.0

    def test_minutes_suffix(self) -> None:
        assert parse_duration("5m") == 300.0

    def test_hours_suffix(self) -> None:
        assert parse_duration("2h") == 7200.0

    def test_days_suffix(self) -> None:
        assert parse_duration("1d") == 86400.0

    def test_case_insensitive_suffix(self) -> None:
        assert parse_duration("2H") == 7200.0

    def test_whitespace_tolerated(self) -> None:
        assert parse_duration(" 5m ") == 300.0

    def test_fractional_value(self) -> None:
        assert parse_duration("1.5h") == 5400.0

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("-5s")

    def test_unparseable_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("abc")

    def test_unknown_suffix_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("10y")


class TestNotificationConfig:
    def test_defaults(self) -> None:
        cfg = NotificationConfig()
        assert cfg.desktop is True
        assert cfg.icloud_copy is True
        assert "CloudDocs/Claude" in str(cfg.icloud_dir)

    def test_default_icloud_dir_user_expanded(self) -> None:
        cfg = NotificationConfig()
        # ~ must be expanded at construction (no literal tilde remains)
        assert not str(cfg.icloud_dir).startswith("~")

    def test_icloud_dir_tilde_expanded(self) -> None:
        cfg = NotificationConfig(icloud_dir="~/somewhere")
        assert str(cfg.icloud_dir) == str(Path.home() / "somewhere")

    def test_icloud_dir_env_expanded(self, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.setenv("ICLOUD_ROOT", "/tmp/icloud")
        cfg = NotificationConfig(icloud_dir="${ICLOUD_ROOT}/sub")
        assert str(cfg.icloud_dir) == "/tmp/icloud/sub"

    def test_extra_keys_allowed(self) -> None:
        # FlexibleModel -> forward-compat extra keys tolerated
        cfg = NotificationConfig(slack=True)
        assert cfg.desktop is True


from davinci_monet.daemon.config import WatchRule


class TestWatchRule:
    def test_minimal(self) -> None:
        rule = WatchRule(name="cam", watch="/in/*.nc", run="/cfg.yaml")
        assert rule.name == "cam"
        assert rule.watch == "/in/*.nc"
        assert rule.run == "/cfg.yaml"
        assert rule.on_fire == "whole_config"
        assert rule.settle == 30.0
        assert rule.enabled is True
        assert rule.env == {}
        assert rule.notify is None
        assert rule.inject_into is None
        assert rule.sentinel is None

    def test_settle_string_parsed(self) -> None:
        rule = WatchRule(name="r", watch="/x", run="/c", settle="5m")
        assert rule.settle == 300.0

    def test_settle_numeric_passthrough(self) -> None:
        rule = WatchRule(name="r", watch="/x", run="/c", settle=45)
        assert rule.settle == 45.0

    def test_settle_mode_quiescence_default(self) -> None:
        rule = WatchRule(name="r", watch="/x", run="/c")
        assert rule.settle_mode == "quiescence"

    def test_settle_mode_sentinel_when_set(self) -> None:
        rule = WatchRule(name="r", watch="/x", run="/c", sentinel="/in/DONE")
        assert rule.settle_mode == "sentinel"

    def test_on_fire_new_files_only_with_inject(self) -> None:
        rule = WatchRule(
            name="r", watch="/x", run="/c",
            on_fire="new_files_only", inject_into="cam",
        )
        assert rule.on_fire == "new_files_only"
        assert rule.inject_into == "cam"

    def test_bad_on_fire_rejected(self) -> None:
        with pytest.raises(Exception):  # pydantic ValidationError
            WatchRule(name="r", watch="/x", run="/c", on_fire="bogus")

    def test_notify_channels(self) -> None:
        rule = WatchRule(name="r", watch="/x", run="/c", notify=["desktop", "log"])
        assert rule.notify == ["desktop", "log"]

    def test_bad_notify_channel_rejected(self) -> None:
        with pytest.raises(Exception):
            WatchRule(name="r", watch="/x", run="/c", notify=["pager"])

    def test_env_overlay_not_expanded(self) -> None:
        # env is the layer-2 worker overlay; values are stored verbatim
        rule = WatchRule(name="r", watch="/x", run="/c", env={"DATA": "${HOME}/d"})
        assert rule.env == {"DATA": "${HOME}/d"}
