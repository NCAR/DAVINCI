"""Tests for davinci_monet.util.system."""

from __future__ import annotations

from davinci_monet.util.system import _cpu_name_from_cpuinfo


def test_cpu_name_from_cpuinfo_extracts_model_name() -> None:
    text = "processor\t: 0\nmodel name\t: Apple M2\ncache size\t: 8192 KB\n"
    assert _cpu_name_from_cpuinfo(text) == "Apple M2"


def test_cpu_name_from_cpuinfo_missing_field_returns_none() -> None:
    text = "processor\t: 0\ncache size\t: 8192 KB\n"
    assert _cpu_name_from_cpuinfo(text) is None
