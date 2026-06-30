from __future__ import annotations

import ast
from typing import Any

import numpy as np
import xarray as xr


class FormulaError(ValueError):
    """Raised when a formula is invalid or unsafe."""


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARY = (ast.UAdd, ast.USub, ast.Not)
_ALLOWED_COMPARE = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_ALLOWED_BOOL = (ast.And, ast.Or)


def _mean(value: xr.DataArray, *, dim: str | list[str] | None = None) -> xr.DataArray:
    return value.mean(dim=dim, skipna=True)


def _sum(value: xr.DataArray, *, dim: str | list[str] | None = None) -> xr.DataArray:
    return value.sum(dim=dim, skipna=True)


def _count(value: xr.DataArray, *, dim: str | list[str] | None = None) -> xr.DataArray:
    return value.count(dim=dim)


def _where(condition: xr.DataArray, x: Any, y: Any) -> xr.DataArray:
    return xr.where(condition, x, y)


def _weighted_mean(
    value: xr.DataArray, weight: xr.DataArray, *, dim: str | list[str] | None = None
) -> xr.DataArray:
    valid = value.notnull() & weight.notnull() & (weight > 0)
    numer = value.where(valid).fillna(0.0) * weight.where(valid).fillna(0.0)
    denom = weight.where(valid).sum(dim=dim, skipna=True)
    return numer.sum(dim=dim, skipna=True) / denom.where(denom > 0)


def _active_mean(
    value: xr.DataArray, active: xr.DataArray, *, dim: str | list[str] | None = None
) -> xr.DataArray:
    masked = value.where(active)
    return masked.mean(dim=dim, skipna=True)


def _fraction(active: xr.DataArray, *, dim: str | list[str] | None = None) -> xr.DataArray:
    return active.astype(float).mean(dim=dim, skipna=True)


def _percentile(
    value: xr.DataArray, q: float, *, dim: str | list[str] | None = None
) -> xr.DataArray:
    return value.quantile(q / 100.0, dim=dim, skipna=True)


_FUNCTIONS = {
    "mean": _mean,
    "sum": _sum,
    "count": _count,
    "fraction": _fraction,
    "weighted_mean": _weighted_mean,
    "active_mean": _active_mean,
    "where": _where,
    "percentile": _percentile,
}


class _Evaluator(ast.NodeVisitor):
    def __init__(self, env: dict[str, Any]) -> None:
        self.env = env

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id == "nan":
            return np.nan
        if node.id not in self.env:
            raise FormulaError(f"unknown formula name: {node.id}")
        return self.env[node.id]

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            return node.value
        raise FormulaError(f"unsupported constant: {node.value!r}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise FormulaError("unsupported arithmetic operator")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        return left % right

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, _ALLOWED_UNARY):
            raise FormulaError("unsupported unary operator")
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(operand, (bool, np.bool_)):
            return not bool(operand)
        return ~operand

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if not isinstance(node.op, _ALLOWED_BOOL):
            raise FormulaError("unsupported boolean operator")
        values = [self.visit(value) for value in node.values]
        out = values[0]
        for value in values[1:]:
            out = (out & value) if isinstance(node.op, ast.And) else (out | value)
        return out

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        result = None
        for op, comp in zip(node.ops, node.comparators):
            if not isinstance(op, _ALLOWED_COMPARE):
                raise FormulaError("unsupported comparison operator")
            right = self.visit(comp)
            if isinstance(op, ast.Eq):
                current = left == right
            elif isinstance(op, ast.NotEq):
                current = left != right
            elif isinstance(op, ast.Lt):
                current = left < right
            elif isinstance(op, ast.LtE):
                current = left <= right
            elif isinstance(op, ast.Gt):
                current = left > right
            else:
                current = left >= right
            result = current if result is None else (result & current)
            left = right
        return result

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise FormulaError("formula calls are limited to whitelisted functions")
        args = [self.visit(arg) for arg in node.args]
        if any(kw.arg is None for kw in node.keywords):
            raise FormulaError("keyword unpacking is not supported in formulas")
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return _FUNCTIONS[node.func.id](*args, **kwargs)

    def generic_visit(self, node: ast.AST) -> Any:
        raise FormulaError(f"unsupported formula syntax: {type(node).__name__}")


def evaluate_formula(expression: str, env: xr.Dataset | dict[str, Any]) -> xr.DataArray:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(str(exc)) from exc
    if isinstance(env, xr.Dataset):
        eval_env = {name: env[name] for name in env.data_vars}
    else:
        eval_env = dict(env)
    try:
        result = _Evaluator(eval_env).visit(parsed)
    except FormulaError:
        raise
    except Exception as exc:
        raise FormulaError(str(exc)) from exc
    if not isinstance(result, xr.DataArray):
        raise FormulaError("formula must produce an xarray.DataArray")
    result = result.copy(deep=False)
    result.attrs = dict(result.attrs)
    result.attrs["formula"] = expression
    return result
