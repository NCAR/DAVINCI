from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from davinci_monet.analysis.base import DerivedAnalysis
from davinci_monet.analysis.formula import FormulaError, evaluate_formula
from davinci_monet.analysis.gridded_reductions import group_dataset
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry

if TYPE_CHECKING:
    from davinci_monet.config.schema import GriddedAnalysisSpec


def _role_dataset(data: xr.Dataset, roles: dict[str, str]) -> xr.Dataset:
    missing = [var for var in roles.values() if var not in data.data_vars]
    if missing:
        raise ValueError(f"gridded_analysis source is missing variables: {missing}")
    return xr.Dataset(
        {role: data[var] for role, var in roles.items()},
        attrs=dict(data.attrs),
    )


@analysis_registry.register("gridded_analysis")
class GriddedAnalysis(DerivedAnalysis):
    """Role/formula-driven gridded analysis product."""

    name = "gridded_analysis"
    long_name = "Gridded Analysis Product"
    output_geometry = DataGeometry.GRID

    def analyze(self, data: xr.Dataset, spec: GriddedAnalysisSpec) -> xr.Dataset:
        role_ds = _role_dataset(data, spec.roles)
        groups = group_dataset(role_ds, spec.groupby)
        per_group: list[xr.Dataset] = []
        labels: list[str] = []
        for label, subset in groups.items():
            fields: dict[str, xr.DataArray] = {}
            env = subset.copy()
            for field_name, field_spec in spec.fields.items():
                try:
                    field = evaluate_formula(field_spec.formula, env)
                except FormulaError as exc:
                    raise ValueError(f"formula failed for field '{field_name}': {exc}") from exc
                field = field.rename(field_name)
                if field_spec.units:
                    field.attrs["units"] = field_spec.units
                if field_spec.long_name:
                    field.attrs["long_name"] = field_spec.long_name
                if field_spec.display_name:
                    field.attrs["display_name"] = field_spec.display_name
                if field_spec.style_preset:
                    field.attrs["style_preset"] = field_spec.style_preset
                fields[field_name] = field
                env[field_name] = field
            per_group.append(xr.Dataset(fields))
            labels.append(label)
        out = xr.concat(per_group, dim=xr.IndexVariable("group", np.asarray(labels, dtype=object)))
        out.attrs.update(
            {
                "geometry": "grid",
                "analysis_type": "gridded_analysis",
                "source": spec.source,
                "groupby": str(spec.groupby),
            }
        )
        if spec.output_group:
            out.attrs["source_label"] = spec.output_group
        return out
