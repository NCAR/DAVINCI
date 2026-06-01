"""Unit tests for davinci_monet.daemon.config."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from davinci_monet.daemon.config import (
    ConfigurationError,
    DaemonConfig,
    NotificationConfig,
    WatchesFile,
    WatchRule,
    load_watches,
    parse_duration,
)

# ---------------------------------------------------------------------------
# Supervisor import invariant — daemon.config must NOT drag in xarray/pandas
# ---------------------------------------------------------------------------


class TestSupervisorImportInvariant:
    """Guard the architecture invariant: importing daemon.config must not
    transitively load xarray or pandas into the supervisor process.

    We use a subprocess with a fresh Python interpreter so that xarray/pandas
    loaded by the test-suite itself (via other imports) cannot produce a false
    pass.
    """

    def _check_heavy_imports(self, module: str) -> subprocess.CompletedProcess:
        """Run a fresh Python process that imports ``module`` and reports
        whether xarray / pandas ended up in sys.modules."""
        script = (
            f"import {module}\n"
            "import sys\n"
            "print('xarray' in sys.modules)\n"
            "print('pandas' in sys.modules)\n"
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_daemon_config_does_not_load_xarray(self) -> None:
        result = self._check_heavy_imports("davinci_monet.daemon.config")
        assert result.returncode == 0, f"Import failed:\n{result.stderr}"
        lines = result.stdout.strip().splitlines()
        xarray_loaded = lines[0].strip() == "True"
        assert not xarray_loaded, (
            "importing davinci_monet.daemon.config loaded xarray — "
            "the supervisor-imports-nothing-heavy invariant is broken"
        )

    def test_daemon_config_does_not_load_pandas(self) -> None:
        result = self._check_heavy_imports("davinci_monet.daemon.config")
        assert result.returncode == 0, f"Import failed:\n{result.stderr}"
        lines = result.stdout.strip().splitlines()
        pandas_loaded = lines[1].strip() == "True"
        assert not pandas_loaded, (
            "importing davinci_monet.daemon.config loaded pandas — "
            "the supervisor-imports-nothing-heavy invariant is broken"
        )


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

    def test_icloud_dir_env_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ICLOUD_ROOT", "/tmp/icloud")
        cfg = NotificationConfig(icloud_dir="${ICLOUD_ROOT}/sub")
        assert str(cfg.icloud_dir) == "/tmp/icloud/sub"

    def test_extra_keys_allowed(self) -> None:
        # FlexibleModel -> forward-compat extra keys tolerated
        cfg = NotificationConfig(slack=True)
        assert cfg.desktop is True


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
            name="r",
            watch="/x",
            run="/c",
            on_fire="new_files_only",
            inject_into="cam",
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


class TestDaemonConfig:
    def test_defaults(self) -> None:
        cfg = DaemonConfig()
        assert cfg.poll_interval == 5.0
        assert cfg.max_concurrent == 1
        assert cfg.hdf5_file_locking is False
        assert cfg.max_settle_wait == 1800.0
        assert cfg.worker_timeout is None
        assert isinstance(cfg.notifications, NotificationConfig)

    def test_state_dir_user_expanded(self) -> None:
        cfg = DaemonConfig()
        assert not str(cfg.state_dir).startswith("~")
        assert str(cfg.state_dir) == str(Path.home() / ".davinci" / "daemon")

    def test_state_dir_env_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAEMON_ROOT", "/tmp/dmn")
        cfg = DaemonConfig(state_dir="${DAEMON_ROOT}/state")
        assert str(cfg.state_dir) == "/tmp/dmn/state"

    def test_poll_interval_duration_string(self) -> None:
        cfg = DaemonConfig(poll_interval="10s")
        assert cfg.poll_interval == 10.0

    def test_max_settle_wait_duration_string(self) -> None:
        cfg = DaemonConfig(max_settle_wait="30m")
        assert cfg.max_settle_wait == 1800.0

    def test_max_settle_wait_none_disables(self) -> None:
        cfg = DaemonConfig(max_settle_wait=None)
        assert cfg.max_settle_wait is None

    def test_worker_timeout_duration_string(self) -> None:
        cfg = DaemonConfig(worker_timeout="2h")
        assert cfg.worker_timeout == 7200.0

    def test_derived_paths(self) -> None:
        cfg = DaemonConfig(state_dir="/tmp/dstate")
        assert cfg.db_path == Path("/tmp/dstate/history.db")
        assert cfg.socket_path == Path("/tmp/dstate/control.sock")
        assert cfg.pid_path == Path("/tmp/dstate/daemon.pid")
        assert cfg.lock_path == Path("/tmp/dstate/daemon.lock")
        assert cfg.log_path == Path("/tmp/dstate/daemon.log")

    def test_nested_notifications_dict(self) -> None:
        cfg = DaemonConfig(notifications={"desktop": False})
        assert cfg.notifications.desktop is False


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "watches.yaml"
    p.write_text(text)
    return p


class TestLoadWatches:
    def test_minimal_file(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
daemon:
  poll_interval: 5s
watches:
  cam:
    watch: /in/cam/*.nc
    run: /cfg/asia-aq.yaml
""",
        )
        wf = load_watches(path)
        assert isinstance(wf, WatchesFile)
        assert wf.daemon.poll_interval == 5.0
        assert set(wf.watches) == {"cam"}
        rule = wf.watches["cam"]
        assert rule.name == "cam"  # key injected as name
        assert rule.watch == "/in/cam/*.nc"
        assert rule.run == "/cfg/asia-aq.yaml"
        assert rule.on_fire == "whole_config"

    def test_layer1_env_expansion_on_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATA", "/scratch/cam")
        path = _write(
            tmp_path,
            """
watches:
  cam:
    watch: ${DATA}/incoming/*.nc
    run: ${DATA}/cfg.yaml
    sentinel: ${DATA}/DONE
""",
        )
        rule = load_watches(path).watches["cam"]
        assert rule.watch == "/scratch/cam/incoming/*.nc"
        assert rule.run == "/scratch/cam/cfg.yaml"
        assert rule.sentinel == "/scratch/cam/DONE"

    def test_per_rule_env_not_layer1_expanded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATA", "/scratch/cam")
        path = _write(
            tmp_path,
            """
watches:
  cam:
    watch: /in/*.nc
    run: /cfg.yaml
    env:
      DATA: ${OTHER}/root
""",
        )
        rule = load_watches(path).watches["cam"]
        # env overlay is layer-2 (worker-side) -> stays verbatim
        assert rule.env == {"DATA": "${OTHER}/root"}

    def test_new_files_only_requires_inject_into(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
watches:
  modis:
    watch: /in/*.hdf
    run: /cfg.yaml
    on_fire: new_files_only
""",
        )
        with pytest.raises(ConfigurationError, match="inject_into"):
            load_watches(path)

    def test_new_files_only_with_inject_into_ok(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
watches:
  modis:
    watch: /in/*.hdf
    run: /cfg.yaml
    on_fire: new_files_only
    inject_into: modis_src
""",
        )
        rule = load_watches(path).watches["modis"]
        assert rule.on_fire == "new_files_only"
        assert rule.inject_into == "modis_src"

    def test_bad_on_fire_value_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
watches:
  r:
    watch: /in/*.nc
    run: /cfg.yaml
    on_fire: sometimes
""",
        )
        with pytest.raises(ConfigurationError):
            load_watches(path)

    def test_empty_file_defaults(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "watches: {}\n")
        wf = load_watches(path)
        assert wf.watches == {}
        assert wf.daemon.poll_interval == 5.0


from davinci_monet.daemon.config import merge_rules


def _rule(name: str, **kw: object) -> WatchRule:
    base = {"name": name, "watch": f"/in/{name}/*.nc", "run": f"/cfg/{name}.yaml"}
    base.update(kw)
    return WatchRule(**base)


class TestMergeRules:
    def test_declared_only(self) -> None:
        declared = {"a": _rule("a"), "b": _rule("b")}
        merged = merge_rules(declared, {}, set())
        assert set(merged) == {"a", "b"}
        assert all(r.enabled for r in merged.values())

    def test_live_preserved(self) -> None:
        declared = {"a": _rule("a")}
        live = {"z": _rule("z")}
        merged = merge_rules(declared, live, set())
        assert set(merged) == {"a", "z"}

    def test_declared_wins_on_collision(self) -> None:
        declared = {"a": _rule("a", run="/cfg/declared.yaml")}
        live = {"a": _rule("a", run="/cfg/live.yaml")}
        merged = merge_rules(declared, live, set())
        assert merged["a"].run == "/cfg/declared.yaml"

    def test_disabled_applied_to_declared(self) -> None:
        declared = {"a": _rule("a"), "b": _rule("b")}
        merged = merge_rules(declared, {}, {"a"})
        assert merged["a"].enabled is False
        assert merged["b"].enabled is True

    def test_disabled_applied_to_live(self) -> None:
        live = {"z": _rule("z")}
        merged = merge_rules({}, live, {"z"})
        assert merged["z"].enabled is False

    def test_disabled_name_not_present_is_noop(self) -> None:
        declared = {"a": _rule("a")}
        merged = merge_rules(declared, {}, {"ghost"})
        assert set(merged) == {"a"}
        assert merged["a"].enabled is True

    def test_does_not_mutate_inputs(self) -> None:
        a = _rule("a", enabled=True)
        declared = {"a": a}
        merge_rules(declared, {}, {"a"})
        # original object must be untouched (returns new instances)
        assert a.enabled is True
