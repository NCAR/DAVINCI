from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.analysis.formula import FormulaError, evaluate_formula


def _dataset() -> xr.Dataset:
    time = np.array(["2008-07-01T00:00", "2008-07-01T03:00"], dtype="datetime64[ns]")
    lat = np.array([-1.0, 1.0])
    lon = np.array([0.0, 90.0])
    shape = (2, 2, 2)
    return xr.Dataset(
        {
            "analysis": (("time", "lat", "lon"), np.arange(8, dtype=float).reshape(shape)),
            "observation": (("time", "lat", "lon"), np.ones(shape)),
            "mask": (
                ("time", "lat", "lon"),
                np.array(
                    [
                        [[1.0, 0.0], [1.0, 0.0]],
                        [[0.0, 1.0], [1.0, 0.0]],
                    ]
                ),
            ),
            "count": (("time", "lat", "lon"), np.ones(shape) * 2.0),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_arithmetic_expression_uses_role_names() -> None:
    out = evaluate_formula("analysis - observation", _dataset())
    np.testing.assert_allclose(out.values, _dataset()["analysis"].values - 1.0)


def test_whitelisted_mean_and_active_mean() -> None:
    ds = _dataset()
    mean = evaluate_formula('mean(analysis, dim="time")', ds)
    active = evaluate_formula('active_mean(analysis - observation, mask > 0.5, dim="time")', ds)
    assert mean.dims == ("lat", "lon")
    assert active.dims == ("lat", "lon")
    np.testing.assert_allclose(mean.values, [[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(active.values[0, 0], -1.0)
    np.testing.assert_allclose(active.values[0, 1], 4.0)


def test_weighted_mean_and_where() -> None:
    ds = _dataset()
    out = evaluate_formula('weighted_mean(where(mask > 0.5, analysis, nan), count, dim="time")', ds)
    assert out.dims == ("lat", "lon")
    assert np.isfinite(out.values[0, 0])
    assert np.isnan(out.values[1, 1])


def test_scalar_not_uses_logical_bool_semantics() -> None:
    ds = _dataset()
    out = evaluate_formula("where(not True, analysis, observation)", ds)
    np.testing.assert_allclose(out.values, ds["observation"].values)


def test_direct_variable_result_does_not_mutate_source_attrs() -> None:
    ds = _dataset()
    ds["analysis"].attrs["units"] = "1"

    out = evaluate_formula("analysis", ds)

    assert out.attrs["units"] == "1"
    assert out.attrs["formula"] == "analysis"
    assert ds["analysis"].attrs == {"units": "1"}


def test_unknown_function_kwargs_are_formula_errors() -> None:
    with pytest.raises(FormulaError):
        evaluate_formula("mean(analysis, bogus=1)", _dataset())


def test_dangerous_integer_exponentiation_is_rejected_before_large_int_compute() -> None:
    with pytest.raises(FormulaError, match="exponent"):
        evaluate_formula("analysis + 10**10**4", _dataset())


def test_small_integer_exponentiation_is_allowed() -> None:
    out = evaluate_formula("analysis + 2**8", _dataset())
    np.testing.assert_allclose(out.values, _dataset()["analysis"].values + 256.0)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('date')",
        "analysis.values",
        "open('x')",
        "lambda x: x",
        "[x for x in [1]]",
        'mean(analysis, **{"dim": "time"})',
    ],
)
def test_unsafe_expressions_are_rejected(expr: str) -> None:
    with pytest.raises(FormulaError):
        evaluate_formula(expr, _dataset())
