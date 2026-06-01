"""Unit tests for daemon iCloud copy of plots + run summary."""

from __future__ import annotations

from pathlib import Path

from davinci_monet.daemon.notify import IcloudCopier, copy_to_icloud


def _make_plot(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x89PNG fake")
    return p


def test_copy_creates_icloud_dir_and_copies_plots(tmp_path: Path) -> None:
    plots = [
        str(_make_plot(tmp_path, "scatter.png")),
        str(_make_plot(tmp_path, "timeseries.pdf")),
    ]
    icloud = tmp_path / "iCloud" / "Claude"  # does not exist yet
    copied = copy_to_icloud(
        icloud_dir=icloud,
        plots=plots,
        summary_text="# Run cam_realtime\nstatus: completed\n",
        summary_name="cam_realtime_job7.md",
    )
    assert icloud.is_dir()
    names = {Path(c).name for c in copied}
    assert "scatter.png" in names
    assert "timeseries.pdf" in names
    # The summary markdown is written into icloud_dir.
    assert (icloud / "cam_realtime_job7.md").read_text().startswith("# Run")


def test_copy_skips_missing_plot_files(tmp_path: Path) -> None:
    good = str(_make_plot(tmp_path, "ok.png"))
    missing = str(tmp_path / "gone.png")
    icloud = tmp_path / "iCloud"
    copied = copy_to_icloud(
        icloud_dir=icloud,
        plots=[good, missing],
        summary_text="x",
        summary_name="s.md",
    )
    names = {Path(c).name for c in copied}
    assert "ok.png" in names
    assert "gone.png" not in names


def test_copy_uses_injected_copyfn(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fake_copy(src: str, dst: str) -> None:
        calls.append((src, dst))

    src = str(_make_plot(tmp_path, "a.png"))
    icloud = tmp_path / "ic"
    copy_to_icloud(
        icloud_dir=icloud,
        plots=[src],
        summary_text="t",
        summary_name="s.md",
        copyfn=fake_copy,
    )
    assert len(calls) == 1
    assert calls[0][0] == src
    assert calls[0][1].endswith("a.png")


def test_icloud_copier_class_binds_dir(tmp_path: Path) -> None:
    icloud = tmp_path / "ic"
    src = str(_make_plot(tmp_path, "b.png"))
    copier = IcloudCopier(icloud_dir=icloud)
    copied = copier(plots=[src], summary_text="t", summary_name="s.md")
    assert (icloud / "s.md").exists()
    assert any(Path(c).name == "b.png" for c in copied)
