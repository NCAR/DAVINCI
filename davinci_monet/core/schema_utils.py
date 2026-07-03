"""Helpers for Pydantic schema objects."""

from __future__ import annotations

from typing import Any


def dump_schema(value: Any, **kwargs: Any) -> dict[str, Any]:
    """Return a dictionary from a Pydantic object."""
    kwargs.setdefault("mode", "json")
    return dict(value.model_dump(**kwargs))


def is_schema_object(value: Any) -> bool:
    """Return whether ``value`` looks like a Pydantic schema object."""
    return hasattr(value, "model_dump")


def validate_schema(schema_cls: Any, data: Any) -> Any:
    """Validate data with a Pydantic schema class."""
    return schema_cls.model_validate(data)
