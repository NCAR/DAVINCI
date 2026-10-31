"""Curses input and menus, isolated from the importable map/data API."""

from __future__ import annotations

import curses
from typing import Any

from davinci_monet.core.coordinates import vertical_dim_name

from .data import DisplayGrid, time_dimension
from .render import BinScale, MapView, ascii_text, render_frame

HELP = [
    "ASCII geographic maps - one character per display-grid cell",
    "",
    "Arrows / h j k l    Pan west/south/north/east one cell",
    "PageUp / PageDown   Pan north/south half a screen",
    "Home               Center the viewport on (0, 0)",
    "v                  Select a variable (type / to filter the menu)",
    "t / z / d          Select time / vertical level / another dimension",
    ", / .              Previous / next time",
    "[ / ]              Previous / next native vertical level",
    "g                  Enter grid rows columns, e.g. 180 360",
    "+ / -              Double / halve the logical grid resolution",
    "f                  Toggle whole-globe fit and the requested grid",
    "i                  Toggle linear / nearest interpolation",
    "b                  Set MIN MAX for bins; empty input restores automatic",
    "m                  Set masking cutoff; empty input disables it",
    "?                  This help (scroll with arrows or PageUp/PageDown)",
    "q / Esc            Quit; Esc cancels a menu or prompt",
    "",
    "The scale uses the full native slice and stays fixed while panning.",
    "Native levels are selected by index; values and units appear in menus.",
    "LWI is categorical: 1=land, 2=ice, -=water/missing; always nearest.",
    "Regional coverage and missing cells remain '-'; no extrapolation.",
]


class CursesViewer:
    def __init__(self, view: MapView, *, fit: bool = False) -> None:
        self.view = view
        self.requested_grid = view.grid
        self.fit = fit
        self.message = "?: help; all displayed time/level indices are zero-based"

    @staticmethod
    def _write(screen: Any, row: int, text: str) -> None:
        rows, columns = screen.getmaxyx()
        if row < rows and columns > 1:
            try:
                screen.addnstr(row, 0, ascii_text(text), columns - 1)
            except curses.error:
                # A resize may occur after measuring this row. The next
                # KEY_RESIZE redraws it without consuming another input event.
                pass

    def _prompt(self, screen: Any, label: str, initial: str = "") -> str | None:
        text = initial
        while True:
            screen.erase()
            self._write(screen, 0, label)
            self._write(screen, 1, "> " + text)
            self._write(screen, 3, "Enter: accept | Esc: cancel")
            screen.refresh()
            key = screen.getch()
            if key in (10, 13, curses.KEY_ENTER):
                return text.strip()
            if key == 27:
                return None
            if key in (curses.KEY_BACKSPACE, 8, 127):
                text = text[:-1]
            elif 32 <= key < 127:
                text += chr(key)

    def _choose(self, screen: Any, title: str, options: list[str], current: int = 0) -> int | None:
        visible = list(range(len(options)))
        position = min(current, max(0, len(visible) - 1))
        while True:
            rows, _ = screen.getmaxyx()
            page = max(1, rows - 3)
            screen.erase()
            self._write(screen, 0, title)
            start = (position // page) * page
            for row, idx in enumerate(visible[start : start + page], 1):
                mark = "> " if start + row - 1 == position else "  "
                self._write(screen, row, mark + options[idx])
            self._write(
                screen, rows - 1, "arrows/PgUp/PgDn: move | /:filter | Enter:select | Esc:cancel"
            )
            screen.refresh()
            key = screen.getch()
            if key in (27, ord("q")):
                return None
            if key in (10, 13, curses.KEY_ENTER) and visible:
                return visible[position]
            if key == ord("/"):
                text = self._prompt(screen, "Filter (empty input shows all):")
                if text is not None:
                    visible = [
                        i for i, value in enumerate(options) if text.lower() in value.lower()
                    ]
                    position = 0
            elif key in (curses.KEY_UP, ord("k")):
                position -= 1
            elif key in (curses.KEY_DOWN, ord("j")):
                position += 1
            elif key == curses.KEY_PPAGE:
                position -= page
            elif key == curses.KEY_NPAGE:
                position += page
            elif key == curses.KEY_HOME:
                position = 0
            elif key == curses.KEY_END:
                position = len(visible) - 1
            position = max(0, min(position, len(visible) - 1))

    def _select_dimension(self, screen: Any, dim: str | None) -> None:
        view = self.view
        if dim is None or dim not in view.indices:
            self.message = "This variable has no matching dimension."
            return
        field = view.dataset.dataset[view.variable]
        coord = field.coords.get(dim)
        units = str(coord.attrs.get("units", "")) if coord is not None else ""
        options = [
            (
                f"{i}: {ascii_text(coord.values[i])} {units}"
                if coord is not None and coord.ndim == 1
                else str(i)
            )
            for i in range(field.sizes[dim])
        ]
        picked = self._choose(screen, f"Select {dim} (native indices)", options, view.indices[dim])
        if picked is not None:
            view.change(indices={**view.indices, dim: picked})

    def _step(self, dim: str | None, amount: int) -> None:
        view = self.view
        if dim is None or dim not in view.indices:
            self.message = "This variable has no matching dimension."
            return
        size = view.dataset.dataset[view.variable].sizes[dim]
        picked = max(0, min(size - 1, view.indices[dim] + amount))
        if picked != view.indices[dim]:
            view.change(indices={**view.indices, dim: picked})

    def _fit_to_window(self, rows: int, columns: int) -> None:
        if self.fit and rows >= 12 and columns >= 45:
            height = rows - 6 - len(self.view.legend_lines(columns))
            if height > 0:
                grid = DisplayGrid(height, columns - 9)
                if grid != self.view.grid:
                    self.view.change(grid=grid)
                self.view.viewport.top, self.view.viewport.left = 0, 0
                self.view.viewport.initialized = True

    def handle_key(self, screen: Any, key: int) -> bool:
        """Process one event; returning False exits the viewer."""
        view = self.view
        rows, columns = screen.getmaxyx()
        height, width = view.map_size(rows, columns)
        field = view.dataset.dataset[view.variable]
        if key in (ord("q"), 27):
            return False
        pans = {
            curses.KEY_UP: (-1, 0),
            ord("k"): (-1, 0),
            curses.KEY_DOWN: (1, 0),
            ord("j"): (1, 0),
            curses.KEY_LEFT: (0, -1),
            ord("h"): (0, -1),
            curses.KEY_RIGHT: (0, 1),
            ord("l"): (0, 1),
            curses.KEY_PPAGE: (-max(1, height // 2), 0),
            curses.KEY_NPAGE: (max(1, height // 2), 0),
        }
        if key in pans:
            if not self.fit:
                view.viewport.pan(*pans[key], view.grid, height, width)
        elif key == curses.KEY_HOME:
            view.viewport.recenter()
        elif key == ord("v"):
            names = view.dataset.variables
            options = [
                f"{name} [{view.dataset.dataset[name].attrs.get('units', '')}] "
                f"{dict(view.dataset.dataset[name].sizes)}"
                for name in names
            ]
            picked = self._choose(
                screen, "Select map variable", options, names.index(view.variable)
            )
            if picked is not None:
                view.change(variable=names[picked])
                self.message = "Variable changed; native surface level selected when available."
        elif key in (ord("t"), ord("z")):
            self._select_dimension(
                screen, time_dimension(field) if key == ord("t") else vertical_dim_name(field)
            )
        elif key == ord("d"):
            dims = list(view.indices)
            picked = self._choose(screen, "Select dimension", dims)
            if picked is not None:
                self._select_dimension(screen, dims[picked])
        elif key in (ord(","), ord(".")):
            self._step(time_dimension(field), 1 if key == ord(".") else -1)
        elif key in (ord("["), ord("]")):
            self._step(vertical_dim_name(field), 1 if key == ord("]") else -1)
        elif key == ord("f"):
            self.fit = not self.fit
            if not self.fit:
                view.change(grid=self.requested_grid)
                view.viewport.recenter()
        elif key == ord("g"):
            value = self._prompt(screen, "Grid latitude rows longitude columns:")
            if value:
                sizes = [int(part) for part in value.replace("x", " ").split()]
                if len(sizes) != 2:
                    raise ValueError("Enter two positive integers, e.g. 180 360.")
                grid = DisplayGrid(*sizes)
                view.change(grid=grid)
                self.requested_grid, self.fit = grid, False
                view.viewport.recenter()
        elif key in (ord("+"), ord("="), ord("-")):
            factor = 2.0 if key != ord("-") else 0.5
            grid = DisplayGrid(
                max(1, int(view.grid.rows * factor)), max(1, int(view.grid.columns * factor))
            )
            view.change(grid=grid)
            self.requested_grid, self.fit = grid, False
            view.viewport.recenter()
        elif key == ord("i"):
            view.change(method="nearest" if view.method == "linear" else "linear")
        elif key in (ord("b"), ord("m")):
            self._change_scale(screen, key)
        elif key == ord("?"):
            self._choose(screen, "Help (Esc returns to the map)", HELP)
        return True

    def _change_scale(self, screen: Any, key: int) -> None:
        view = self.view
        if key == ord("b"):
            text = self._prompt(screen, "Bin limits MIN MAX; empty input: automatic:")
            if text is None:
                return
            limits = tuple(float(part) for part in text.split()) if text else None
            if limits is not None and len(limits) != 2:
                raise ValueError("Enter two numbers, MIN MAX.")
            new_limits = (limits[0], limits[1]) if limits else None
            BinScale.from_values(view.source.values, limits=new_limits, bins=view.bins)
            view.limits = new_limits
        else:
            text = self._prompt(screen, "Mask values strictly below cutoff; empty input: disable:")
            if text is None:
                return
            cutoff = float(text) if text else None
            BinScale.from_values(view.source.values, mask_below=cutoff, bins=view.bins)
            view.mask_below = cutoff
        view.change()

    def run(self, screen: Any) -> None:
        screen.keypad(True)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        running = True
        while running:
            try:
                rows, columns = screen.getmaxyx()
                self._fit_to_window(rows, columns)
                screen.erase()
                for row, line in enumerate(render_frame(self.view, rows, columns, self.message)):
                    self._write(screen, row, line)
                screen.refresh()
                key = screen.getch()
                if key in (curses.KEY_RESIZE, 12):
                    screen.clear()
                else:
                    running = self.handle_key(screen, key)
            except (ValueError, OSError, IndexError) as exc:
                self.message = ascii_text(exc)
            except curses.error:
                self.message = "Terminal resized; press a key to redraw."
                key = screen.getch()
                if key in (ord("q"), 27):
                    running = False


def run(view: MapView, *, fit: bool = False) -> None:
    """Restore terminal modes even on an exception or keyboard interruption."""
    curses.wrapper(CursesViewer(view, fit=fit).run)
