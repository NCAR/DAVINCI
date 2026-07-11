"""Safety and delivery tests for the FABLE diagnostic plotting CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    repository = Path(__file__).resolve().parents[4]
    path = repository / "analyses/aerosol-tuning/scripts/plot_acceptance_diagnostics.py"
    spec = importlib.util.spec_from_file_location("plot_acceptance_diagnostics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def test_output_replacement_requires_matching_ownership(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(ValueError, match="unowned diagnostic output"):
        SCRIPT._prepare_output_root(acceptance, output, overwrite=True)


def test_owned_output_can_be_replaced_without_touching_inputs(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()
    output = tmp_path / "output"
    SCRIPT._prepare_output_root(acceptance, output, overwrite=False)
    (output / "old.txt").write_text("old", encoding="utf-8")

    SCRIPT._prepare_output_root(acceptance, output, overwrite=True)

    assert not (output / "old.txt").exists()
    assert (output / SCRIPT.OUTPUT_MARKER).is_file()
    assert acceptance.is_dir()


def test_output_must_not_overlap_acceptance_inputs(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()

    with pytest.raises(ValueError, match="must not overlap"):
        SCRIPT._prepare_output_root(acceptance, acceptance / "plots", overwrite=False)


def test_icloud_delivery_contains_only_pdfs(tmp_path: Path) -> None:
    plot_root = tmp_path / "plots"
    plot_root.mkdir()
    pdf = plot_root / "wavelet_seed_1179_mode_2.pdf"
    png = plot_root / "wavelet_seed_1179_mode_2.png"
    pdf.write_bytes(b"pdf")
    png.write_bytes(b"png")
    destination = tmp_path / "icloud/FABLE_acceptance_diagnostics"
    destination.mkdir(parents=True)
    (destination / png.name).write_bytes(b"legacy png")
    (destination / "index.html").write_text("legacy", encoding="utf-8")
    (destination / "diagnostic-record.json").write_text("{}", encoding="utf-8")

    acceptance = tmp_path / "acceptance"
    delivered = SCRIPT._deliver(tmp_path / "icloud", acceptance, [pdf, png])

    assert delivered == destination
    assert {path.name for path in destination.iterdir()} == {pdf.name}
    assert (destination / pdf.name).read_bytes() == b"pdf"


def test_delivery_must_not_overlap_acceptance_inputs(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()
    pdf = tmp_path / "diagnostic.pdf"
    pdf.write_bytes(b"pdf")

    with pytest.raises(ValueError, match="must not overlap"):
        SCRIPT._deliver(acceptance, acceptance, [pdf])
