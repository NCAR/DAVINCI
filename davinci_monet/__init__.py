"""DAVINCI: Data Analysis and Visual Intelligence for Climate/Chemistry.

A modern, type-safe toolkit for evaluating climate and atmospheric
composition models and datasets against observations.
"""

from __future__ import annotations

# Single display version for the toolkit. Kept in lock-step with the
# ``version`` field in pyproject.toml and the top entry of CHANGELOG.md; the
# packaging metadata test (tests/unit/test_packaging_metadata.py) fails if they
# drift. We intentionally do NOT derive this from importlib.metadata: the
# calendar version "26.06" PEP 440-normalizes to "26.6" and an editable install
# can carry stale egg-info, so metadata would misreport the user-facing version.
__version__ = "26.06"
__all__ = ["__version__"]
