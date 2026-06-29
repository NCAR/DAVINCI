from __future__ import annotations

from typer.testing import CliRunner

from davinci_monet.cli.app import app


def test_inspect_command_exits_nonzero_for_missing_pdfs(tmp_path) -> None:
    result = CliRunner().invoke(
        app, ["inspect", str(tmp_path), "--preset", "gridded_aod_diagnostics"]
    )
    assert result.exit_code == 1
    assert "Inspection failed" in result.output


def test_inspect_command_passes_when_pdf_products_exist(tmp_path) -> None:
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "aod.pdf").write_bytes(b"%PDF-1.4\n")

    result = CliRunner().invoke(
        app, ["inspect", str(tmp_path), "--preset", "gridded_aod_diagnostics"]
    )

    assert result.exit_code == 0
    assert "Inspection passed" in result.output


def test_inspect_command_exits_nonzero_for_missing_run_dir(tmp_path) -> None:
    missing = tmp_path / "missing"

    result = CliRunner().invoke(app, ["inspect", str(missing)])

    assert result.exit_code == 1
    assert "run directory does not exist" in result.output
    assert not missing.exists()
