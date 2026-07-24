"""Tests for package metadata needed by optional features."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def test_ai_extra_declares_openrouter_http_client() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    data = tomllib.loads((repo_root / "pyproject.toml").read_text())

    ai_deps = data["project"]["optional-dependencies"]["ai"]
    assert any(dep.lower().startswith("httpx") for dep in ai_deps)


def test_version_is_consistent_across_sources() -> None:
    """``__version__``, pyproject ``version``, and the CHANGELOG top entry agree.

    Guards against the drift where ``__init__`` declared a stale version while
    pyproject/CHANGELOG advanced, so ``davinci --version`` printed the
    wrong number. ``importlib.metadata`` is intentionally not consulted: the
    calendar version ``"26.06"`` PEP 440-normalizes to ``"26.6"`` and editable
    installs can carry stale egg-info, so it is not a faithful display source.
    """
    from davinci_monet import __version__

    repo_root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    pyproject_version = pyproject["project"]["version"]

    changelog = (repo_root / "CHANGELOG.md").read_text()
    match = re.search(r"^##\s+(\S+)", changelog, re.MULTILINE)
    assert match, "Could not find a version heading (## <version>) in CHANGELOG.md"
    changelog_version = match.group(1)

    assert (
        __version__ == pyproject_version
    ), f"__init__.__version__={__version__!r} != pyproject version={pyproject_version!r}"
    assert (
        changelog_version == pyproject_version
    ), f"CHANGELOG top entry={changelog_version!r} != pyproject version={pyproject_version!r}"
