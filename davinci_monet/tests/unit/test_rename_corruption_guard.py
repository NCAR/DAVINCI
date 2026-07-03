"""Guard against broad model/dataset rename corruption in public text."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_FILE = Path(__file__).resolve()

SCAN_ROOTS = ("davinci_monet", "scripts", "examples")
ROOT_FILES = ("CLAUDE.md", "README.md")
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml"}

FORBIDDEN_PHRASES = (
    "Data Dataset",
    "dataset name",
    "dataset output",
    "datasets against datasets",
    "datasets/datasets",
    "mental dataset",
    "Cross-Dataset",
    "Base dataset with strict",
    "the next dataset",
)


def _scanned_files() -> Iterable[Path]:
    for rel_path in ROOT_FILES:
        path = REPO_ROOT / rel_path
        if path.exists():
            yield path
    for root in SCAN_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*")):
            if path == THIS_FILE or path.suffix not in TEXT_SUFFIXES:
                continue
            yield path


def test_rename_corruption_phrases_stay_out_of_public_text() -> None:
    offenders: list[str] = []
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8")
        rel_path = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(text.splitlines(), start=1):
            for phrase in FORBIDDEN_PHRASES:
                if phrase in line:
                    offenders.append(f"{rel_path}:{lineno}: {phrase}")

    assert not offenders, "Rename-corruption phrase(s) found:\n" + "\n".join(offenders)
