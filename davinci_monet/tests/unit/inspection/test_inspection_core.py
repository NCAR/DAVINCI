from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from davinci_monet.inspection.core import inspect_run_directory


def _install_fake_pdftoppm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_pdftoppm = bin_dir / "pdftoppm"
    fake_pdftoppm.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[-1] + '.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')\n"
    )
    fake_pdftoppm.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path('/usr/bin')}")


def test_inspect_run_directory_writes_json_and_markdown(tmp_path) -> None:
    plots = tmp_path / "plots" / "daily"
    plots.mkdir(parents=True)
    (plots / "cam-analyzed-aod.pdf").write_bytes(b"%PDF-1.4\n")
    result = inspect_run_directory(tmp_path, presets=["gridded_aod_diagnostics"])
    assert result.passed is True
    assert (tmp_path / "inspection" / "inspection.json").exists()
    assert (tmp_path / "inspection" / "inspection.md").exists()
    data = json.loads((tmp_path / "inspection" / "inspection.json").read_text())
    assert data["passed"] is True
    assert [check["name"] for check in data["checks"]] == [
        "final_pdf_products_exist",
        "known_preset_selected",
    ]
    assert all("detail" in check for check in data["checks"])


def test_inspect_run_directory_uses_explicit_plot_paths(tmp_path) -> None:
    source_dir = tmp_path / "cam"
    source_dir.mkdir()
    pdf = source_dir / "cam-analyzed-aod.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    result = inspect_run_directory(
        tmp_path,
        presets=["gridded_aod_diagnostics"],
        plot_paths=[pdf],
    )

    assert result.passed is True
    data = json.loads((tmp_path / "inspection" / "inspection.json").read_text())
    assert data["checks"][0]["pdfs"] == ["cam/cam-analyzed-aod.pdf"]


def test_inspect_run_directory_writes_png_previews_for_final_pdfs(tmp_path, monkeypatch) -> None:
    _install_fake_pdftoppm(tmp_path, monkeypatch)
    plots = tmp_path / "plots" / "daily"
    plots.mkdir(parents=True)
    (plots / "cam-analyzed-aod.pdf").write_bytes(b"%PDF-1.4\n")

    result = inspect_run_directory(
        tmp_path, presets=["gridded_aod_diagnostics"], preview_format="png"
    )

    preview = tmp_path / "inspection" / "previews" / "daily" / "cam-analyzed-aod.png"
    assert result.passed is True
    assert result.preview_paths == [preview]
    assert preview.read_bytes().startswith(b"\x89PNG")
    data = json.loads((tmp_path / "inspection" / "inspection.json").read_text())
    assert data["previews"] == ["inspection/previews/daily/cam-analyzed-aod.png"]
    preview_check = next(
        check for check in data["checks"] if check["name"] == "inspection_previews_exist"
    )
    assert preview_check["passed"] is True
    assert preview_check["previews"] == ["inspection/previews/daily/cam-analyzed-aod.png"]


def test_inspect_run_directory_fails_for_unknown_preset(tmp_path) -> None:
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "aod.pdf").write_bytes(b"%PDF-1.4\n")

    result = inspect_run_directory(tmp_path, presets=["missing"])

    assert result.passed is False
    data = json.loads((tmp_path / "inspection" / "inspection.json").read_text())
    assert data["checks"][0]["passed"] is True
    assert data["checks"][1]["passed"] is False
    assert data["checks"][1]["detail"] == "missing"


def test_inspect_run_directory_ignores_pdf_named_directories(tmp_path) -> None:
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "not-a-product.pdf").mkdir()

    result = inspect_run_directory(tmp_path, presets=["gridded_aod_diagnostics"])

    assert result.passed is False
    data = json.loads((tmp_path / "inspection" / "inspection.json").read_text())
    assert data["checks"][0]["passed"] is False
    assert data["checks"][0]["detail"] == "0 PDF plot(s) found"


def test_inspect_run_directory_rejects_missing_run_directory(tmp_path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="run directory does not exist"):
        inspect_run_directory(missing, presets=["gridded_aod_diagnostics"])

    assert not missing.exists()


def test_inspect_run_directory_rejects_file_run_directory(tmp_path) -> None:
    run_file = tmp_path / "not-a-directory"
    run_file.write_text("not a run")

    with pytest.raises(NotADirectoryError, match="run directory is not a directory"):
        inspect_run_directory(run_file, presets=["gridded_aod_diagnostics"])

    assert not (tmp_path / "not-a-directory" / "inspection").exists()
