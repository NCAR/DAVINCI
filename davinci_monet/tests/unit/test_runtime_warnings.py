"""Runtime warning policy tests."""

from __future__ import annotations

import warnings


def test_runtime_warning_policy_suppresses_known_dependency_deprecations() -> None:
    from davinci_monet.runtime_warnings import apply_runtime_warning_filters

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        apply_runtime_warning_filters()

        warnings.warn_explicit(
            "_rrfs_cmaq_mm module is deprecated. Use ufs instead.",
            DeprecationWarning,
            filename="monetio/models/_rrfs_cmaq_mm.py",
            lineno=5,
            module="monetio.models._rrfs_cmaq_mm",
        )
        warnings.warn_explicit(
            "The locs attribute was deprecated in Matplotlib 3.11 and will be removed in 3.13.",
            DeprecationWarning,
            filename="cartopy/mpl/ticker.py",
            lineno=151,
            module="cartopy.mpl.ticker",
        )
        warnings.warn_explicit(
            "Please import `hermitenorm` from the `scipy.special` namespace; "
            "the `scipy.special.orthogonal` namespace is deprecated and will "
            "be removed in SciPy 2.0.0.",
            DeprecationWarning,
            filename="pycwt/mothers.py",
            lineno=8,
            module="pycwt.mothers",
        )
        warnings.warn_explicit(
            "new unrelated deprecation",
            DeprecationWarning,
            filename="somewhere.py",
            lineno=1,
            module="somewhere",
        )

    assert [str(item.message) for item in caught] == ["new unrelated deprecation"]
