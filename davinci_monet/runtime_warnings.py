"""Runtime warning filters for harmless third-party dependency warnings."""

from __future__ import annotations

import warnings


def apply_runtime_warning_filters() -> None:
    """Suppress known harmless deprecation warnings from dependencies."""
    warnings.filterwarnings(
        "ignore",
        message=r"_rrfs_cmaq_mm module is deprecated.*",
        category=DeprecationWarning,
        module=r"monetio\.models\._rrfs_cmaq_mm",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The locs attribute was deprecated in Matplotlib 3\.11.*",
        category=DeprecationWarning,
        module=r"cartopy\.mpl\.ticker",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Please import `hermitenorm` from the `scipy\.special` namespace.*",
        category=DeprecationWarning,
        module=r"pycwt\.mothers",
    )
