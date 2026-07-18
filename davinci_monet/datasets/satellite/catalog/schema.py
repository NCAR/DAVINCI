from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VariableEntry(BaseModel, extra="forbid"):
    display_name: str
    sds_name: str
    units: str = "1"
    wavelength_nm: float | None = None
    long_name: str | None = None
    valid_min: float | None = None
    valid_max: float | None = None
    qa_screening: Literal["level2_usefulness_flag"] | None = None


class ProductEntry(BaseModel, extra="forbid"):
    product_id: str
    instrument: str
    platform: str
    daac: str
    collection: str
    level: str
    geometry: str
    file_format: str
    time_parse: str  # strptime pattern applied to the filename "A%Y%j" token
    cadence: Literal["daily", "monthly", "granule"]
    dim_aliases: dict[str, str] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    variables: list[VariableEntry] = Field(default_factory=list)

    def variable_by_display(self, name: str) -> VariableEntry | None:
        return next((v for v in self.variables if v.display_name == name), None)

    def variable_by_sds(self, name: str) -> VariableEntry | None:
        return next((v for v in self.variables if v.sds_name == name), None)
