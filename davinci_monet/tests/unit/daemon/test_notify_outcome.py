"""Unit tests for notify_outcome routing across desktop + iCloud + log."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from davinci_monet.daemon.config import DaemonConfig, NotificationConfig, WatchRule
from davinci_monet.daemon.contracts import JobRecord, JobStatus
from davinci_monet.daemon.notify import notify_outcome


class _RecordingDesktop:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, title: str, message: str) -> bool:
        self.calls.append((title, message))
        return True


class _RecordingIcloud:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *, plots, summary_text, summary_name) -> list[str]:
        self.calls.append({"plots": list(plots), "summary_name": summary_name})
        return [f"/ic/{summary_name}"]


def _rule(name: str = "cam_realtime", notify=None) -> WatchRule:
    return WatchRule(
        name=name,
        watch="/data/*.nc",
        run="/cfg/x.yaml",
        notify=notify,
    )


def _job(status: JobStatus, plots=None, output_dir=None) -> JobRecord:
    summary = {}
    if plots is not None:
        summary["plots"] = plots
    if output_dir is not None:
        summary["output_dir"] = output_dir
    return JobRecord(
        id=7,
        watch_name="cam_realtime",
        config_path="/cfg/x.yaml",
        on_fire="whole_config",
        files=["/data/a.nc"],
        status=status,
        submitted_at=datetime(2026, 5, 31, 12, 0, 0),
        result_summary=summary or None,
    )


def test_success_fires_desktop_and_icloud(tmp_path: Path) -> None:
    desktop = _RecordingDesktop()
    icloud = _RecordingIcloud()
    cfg = DaemonConfig(
        notifications=NotificationConfig(desktop=True, icloud_copy=True, icloud_dir=tmp_path)
    )
    job = _job(JobStatus.COMPLETED, plots=["/out/scatter.png"])
    notify_outcome(job, cfg, _rule(), desktop=desktop, icloud=icloud)
    assert len(desktop.calls) == 1
    title, msg = desktop.calls[0]
    assert "cam_realtime" in msg
    assert "completed" in msg.lower()
    assert len(icloud.calls) == 1
    assert icloud.calls[0]["plots"] == ["/out/scatter.png"]
    assert icloud.calls[0]["summary_name"].endswith(".md")


def test_failure_fires_desktop_but_not_icloud(tmp_path: Path) -> None:
    desktop = _RecordingDesktop()
    icloud = _RecordingIcloud()
    cfg = DaemonConfig(
        notifications=NotificationConfig(desktop=True, icloud_copy=True, icloud_dir=tmp_path)
    )
    job = _job(JobStatus.FAILED, plots=["/out/scatter.png"])
    notify_outcome(job, cfg, _rule(), desktop=desktop, icloud=icloud)
    assert len(desktop.calls) == 1
    assert "failed" in desktop.calls[0][1].lower()
    assert icloud.calls == []  # no iCloud copy on failure


def test_desktop_disabled_globally_suppresses_desktop(tmp_path: Path) -> None:
    desktop = _RecordingDesktop()
    icloud = _RecordingIcloud()
    cfg = DaemonConfig(
        notifications=NotificationConfig(desktop=False, icloud_copy=True, icloud_dir=tmp_path)
    )
    notify_outcome(
        _job(JobStatus.COMPLETED, plots=[]),
        cfg,
        _rule(),
        desktop=desktop,
        icloud=icloud,
    )
    assert desktop.calls == []


def test_per_rule_notify_overrides_to_desktop_only(tmp_path: Path) -> None:
    desktop = _RecordingDesktop()
    icloud = _RecordingIcloud()
    cfg = DaemonConfig(
        notifications=NotificationConfig(desktop=True, icloud_copy=True, icloud_dir=tmp_path)
    )
    # Rule overrides channels to desktop-only: no iCloud even on success.
    rule = _rule(notify=["desktop"])
    notify_outcome(
        _job(JobStatus.COMPLETED, plots=["/out/p.png"]),
        cfg,
        rule,
        desktop=desktop,
        icloud=icloud,
    )
    assert len(desktop.calls) == 1
    assert icloud.calls == []


def test_per_rule_notify_log_only_suppresses_desktop(tmp_path: Path) -> None:
    desktop = _RecordingDesktop()
    icloud = _RecordingIcloud()
    cfg = DaemonConfig(
        notifications=NotificationConfig(desktop=True, icloud_copy=True, icloud_dir=tmp_path)
    )
    rule = _rule(notify=["log"])
    notify_outcome(
        _job(JobStatus.COMPLETED, plots=["/out/p.png"]),
        cfg,
        rule,
        desktop=desktop,
        icloud=icloud,
    )
    assert desktop.calls == []
    assert icloud.calls == []


def test_outcome_always_logs(tmp_path: Path, caplog) -> None:
    cfg = DaemonConfig(
        notifications=NotificationConfig(desktop=False, icloud_copy=False, icloud_dir=tmp_path)
    )
    with caplog.at_level("INFO"):
        notify_outcome(
            _job(JobStatus.COMPLETED, plots=[]),
            cfg,
            _rule(),
            desktop=_RecordingDesktop(),
            icloud=_RecordingIcloud(),
        )
    assert any("cam_realtime" in r.getMessage() for r in caplog.records)


def test_notifier_notify_result_delegates_to_notify_outcome(tmp_path: Path, monkeypatch) -> None:
    """The Notifier facade forwards (job, cfg, rule) to notify_outcome verbatim."""
    from davinci_monet.daemon import notify as notify_mod
    from davinci_monet.daemon.notify import Notifier

    seen: dict = {}

    def fake_notify_outcome(job, cfg, rule=None, *, desktop=None, icloud=None) -> None:
        seen.update({"job": job, "cfg": cfg, "rule": rule})

    monkeypatch.setattr(notify_mod, "notify_outcome", fake_notify_outcome)

    cfg = DaemonConfig(
        notifications=NotificationConfig(desktop=True, icloud_copy=True, icloud_dir=tmp_path)
    )
    rule = _rule()
    job = _job(JobStatus.COMPLETED, plots=["/out/p.png"])
    notifier = Notifier(cfg)
    notifier.notify_result(job, rule)

    assert seen["job"] is job
    assert seen["cfg"] is cfg
    assert seen["rule"] is rule
