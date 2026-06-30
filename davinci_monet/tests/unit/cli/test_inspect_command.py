from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

from davinci_monet.cli.app import app


def _install_fake_pdftoppm(tmp_path: Path, monkeypatch) -> None:
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


def test_inspect_command_exits_nonzero_for_missing_pdfs(tmp_path) -> None:
    result = CliRunner().invoke(
        app, ["inspect", str(tmp_path), "--preset", "gridded_aod_diagnostics"]
    )
    assert result.exit_code == 1
    assert "Inspection failed" in result.output


def test_inspect_command_passes_when_pdf_products_exist(tmp_path, monkeypatch) -> None:
    _install_fake_pdftoppm(tmp_path, monkeypatch)
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "aod.pdf").write_bytes(b"%PDF-1.4\n")

    result = CliRunner().invoke(
        app, ["inspect", str(tmp_path), "--preset", "gridded_aod_diagnostics"]
    )

    assert result.exit_code == 0
    assert "Inspection passed" in result.output
    assert (tmp_path / "inspection" / "previews" / "aod.png").exists()


def test_inspect_command_exits_nonzero_for_missing_run_dir(tmp_path) -> None:
    missing = tmp_path / "missing"

    result = CliRunner().invoke(app, ["inspect", str(missing)])

    assert result.exit_code == 1
    assert "run directory does not exist" in result.output
    assert not missing.exists()
