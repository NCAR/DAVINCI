"""Terminal-independent ASCII binning, viewport navigation, and screen layout."""

from __future__ import annotations

import textwrap
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import DisplayGrid, MapDataset, interpolate, is_land_water_ice


def ascii_text(value: object) -> str:
    """Keep file metadata printable and safe for an ASCII character-cell display."""
    text = str(value).replace("µ", "u").replace("°", "deg").replace("−", "-")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "replace").decode("ascii")
    return "".join(char if char.isprintable() else " " for char in text)


@dataclass(frozen=True)
class BinScale:
    """Up to ten intervals; lower bounds and the final maximum are included."""

    edges: np.ndarray
    empty: bool = False

    @classmethod
    def from_values(
        cls,
        values: np.ndarray,
        *,
        limits: tuple[float, float] | None = None,
        mask_below: float | None = None,
        bins: int = 10,
    ) -> BinScale:
        if not 1 <= bins <= 10:
            raise ValueError("ASCII maps support between 1 and 10 bins.")
        if mask_below is not None and not np.isfinite(mask_below):
            raise ValueError("The masking cutoff must be finite.")
        valid = np.isfinite(values)
        if mask_below is not None:
            valid &= values >= mask_below
        finite = values[valid]
        if limits is not None:
            low, high = limits
            if not np.isfinite(limits).all() or low >= high:
                raise ValueError("Scale limits must be finite with MIN < MAX.")
        elif finite.size:
            low, high = float(finite.min()), float(finite.max())
        else:
            low, high = 0.0, 1.0
        return cls(np.linspace(low, high, bins + 1), empty=not finite.size)

    def legend(self, units: str = "") -> str:
        if self.empty:
            return "No finite, unmasked values; -=missing/masked"
        suffix = f" ({ascii_text(units)})" if units else ""
        if self.edges[0] == self.edges[-1]:
            return f"Bins{suffix}: 0={self.edges[0]:.6g} (constant); -=missing/masked"
        last = len(self.edges) - 2
        bins = [
            f"{i}=[{self.edges[i]:.6g},{self.edges[i + 1]:.6g}{']' if i == last else ')'}"
            for i in range(last + 1)
        ]
        return f"Bins{suffix}: " + " ".join(bins) + "; tails clipped; -=missing/masked"


def ascii_cells(
    values: np.ndarray,
    scale: BinScale,
    *,
    mask_below: float | None = None,
    land_water_ice: bool = False,
) -> np.ndarray:
    """Encode finite values as digits, reserving '-' for missing or masked cells."""
    chars = np.full(values.shape, "-", dtype="<U1")
    valid = np.isfinite(values)
    if mask_below is not None:
        valid &= values >= mask_below
    if land_water_ice:
        # Packed MERRA-2 flags can be slightly different from exact integers.
        for code in (1, 2):
            chars[valid & np.isclose(values, code, atol=1e-3, rtol=0)] = str(code)
        return chars
    if scale.edges[0] == scale.edges[-1]:
        chars[valid] = "0"
    else:
        bins = np.searchsorted(scale.edges, values[valid], side="right") - 1
        chars[valid] = np.asarray(list("0123456789"))[np.clip(bins, 0, len(scale.edges) - 2)]
    return chars


@dataclass
class Viewport:
    """North/south movement clamps at the poles; east/west movement wraps."""

    top: int = 0
    left: int = 0
    initialized: bool = False

    def constrain(self, grid: DisplayGrid, height: int, width: int) -> None:
        if not self.initialized:
            self.top = max(0, (grid.rows - height) // 2)
            self.left = max(0, (grid.columns - width) // 2)
            self.initialized = True
        self.top = max(0, min(self.top, max(0, grid.rows - height)))
        self.left %= grid.columns

    def pan(self, dy: int, dx: int, grid: DisplayGrid, height: int, width: int) -> None:
        self.constrain(grid, height, width)
        self.top += dy
        self.left += dx
        self.constrain(grid, height, width)

    def recenter(self) -> None:
        self.initialized = False


class MapView:
    """Cached map state shared by curses and noninteractive text snapshots."""

    def __init__(
        self,
        dataset: MapDataset,
        variable: str | None = None,
        *,
        grid: DisplayGrid | None = None,
        indices: dict[str, int] | None = None,
        method: str = "linear",
        limits: tuple[float, float] | None = None,
        mask_below: float | None = None,
        bins: int = 10,
    ) -> None:
        self.dataset = dataset
        self.variable = variable or dataset.variables[0]
        self.grid = grid or DisplayGrid()
        self.indices: dict[str, int] = {}
        self.method = method
        self.limits = limits
        self.mask_below = mask_below
        self.bins = bins
        self.viewport = Viewport()
        self.change(indices=indices or {})

    def change(
        self,
        *,
        variable: str | None = None,
        indices: dict[str, int] | None = None,
        grid: DisplayGrid | None = None,
        method: str | None = None,
    ) -> None:
        """Update atomically, leaving the last working map intact on a read error."""
        name = variable or self.variable
        selection = self.dataset.default_indices(name)
        selection.update(
            indices if indices is not None else (self.indices if name == self.variable else {})
        )
        source = self.dataset.select(name, selection)
        target = grid or self.grid
        interpolation = method or self.method
        categorical = is_land_water_ice(source)
        values = interpolate(source, target, interpolation)
        # Scale from the complete native slice, independent of viewport/resolution.
        scale = BinScale.from_values(
            source.values, limits=self.limits, mask_below=self.mask_below, bins=self.bins
        )
        chars = ascii_cells(values, scale, mask_below=self.mask_below, land_water_ice=categorical)
        selection = {
            dim: idx % self.dataset.dataset[name].sizes[dim] for dim, idx in selection.items()
        }
        self.variable, self.indices, self.grid = name, selection, target
        self.method, self.categorical = interpolation, categorical
        self.source, self.values, self.scale, self.chars = source, values, scale, chars

    def selection_labels(self) -> list[str]:
        field = self.dataset.dataset[self.variable]
        labels = []
        for dim, index in self.indices.items():
            coord = field.coords.get(dim)
            if coord is not None and coord.ndim == 1:
                value = f" ({ascii_text(coord.values[index])}"
                units = str(coord.attrs.get("units", ""))
                value += f" {ascii_text(units)})" if units else ")"
            else:
                value = ""
            labels.append(f"{dim}={index}/{field.sizes[dim] - 1}{value}")
        return labels

    def legend_lines(self, columns: int) -> list[str]:
        if self.categorical:
            legend = "LWI categories: 1=land 2=ice -=water/missing (nearest interpolation)"
        else:
            legend = self.scale.legend(str(self.source.attrs.get("units", "")))
        if self.mask_below is not None:
            legend += f"; values < {self.mask_below:.6g} masked"
        return textwrap.wrap(legend, width=max(1, columns - 1))

    def map_size(self, rows: int, columns: int) -> tuple[int, int]:
        return (
            min(self.grid.rows, max(0, rows - 6 - len(self.legend_lines(columns)))),
            min(self.grid.columns, max(0, columns - 9)),
        )


def render_frame(view: MapView, rows: int, columns: int, message: str = "") -> list[str]:
    """Compose a complete screen that fits the actual terminal dimensions.

    One column is reserved to avoid writing curses' bottom-right cell. No GUI,
    image rendering, or pipeline dependencies are involved.
    """
    if rows < 12 or columns < 45:
        return [
            text[: max(0, columns - 1)]
            for text in ("Terminal too small: resize to at least 12 rows x 45 columns.", "q: quit")
        ][:rows]
    height, width = view.map_size(rows, columns)
    if height < 1:
        return ["Resize terminal to leave room for the map and legend."[: columns - 1]]
    view.viewport.constrain(view.grid, height, width)
    top, left = view.viewport.top, view.viewport.left
    col_indices = (left + np.arange(width)) % view.grid.columns
    latitudes, longitudes = view.grid.latitudes, view.grid.longitudes
    north, south = latitudes[top], latitudes[top + height - 1]
    west, east = longitudes[col_indices[0]], longitudes[col_indices[-1]]
    center_row, center_col = top + height // 2, int(col_indices[width // 2])
    value = view.values[center_row, center_col]
    method = "nearest" if view.categorical else view.method
    units = ascii_text(view.source.attrs.get("units", ""))
    lines = [
        f"{view.variable} [{units}] | grid {view.grid.rows}x{view.grid.columns} | "
        f"terminal {rows}x{columns} | {method} | {ascii_text(Path(view.dataset.filename).name)}",
        " | ".join(view.selection_labels()) or "2-D field (no time or level dimension)",
        f"N {north:.2f} S {south:.2f} W {west:.2f} E {east:.2f} | "
        f"center ({latitudes[center_row]:.2f},{longitudes[center_col]:.2f})={value:.6g}",
    ]
    for row in range(top, top + height):
        lines.append(f"{latitudes[row]:6.1f}| " + "".join(view.chars[row, col_indices]))
    lines.append(f"        lon {west:.2f} -> {east:.2f} (east; wraps at 180)")
    lines.extend(view.legend_lines(columns))
    lines.append("arrows/hjkl:pan v:var t:time z:level g:grid f:fit b:bins ?:help q:quit")
    lines.append(message)
    return [ascii_text(line)[: columns - 1] for line in lines[:rows]]
