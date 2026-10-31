"""Standalone argparse entry point; optional YAML and macOS Terminal launcher."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml

from davinci_monet.core.coordinates import vertical_dim_name

from .config import ViewerConfig, load_config, source_path
from .data import DisplayGrid, MapDataset, time_dimension
from .render import MapView, ascii_text, render_frame

_APPLE_SCRIPT = """on run argv
    set requestedRows to (item 1 of argv) as integer
    set requestedColumns to (item 2 of argv) as integer
    set viewerCommand to item 3 of argv
    set requestedFontSize to (item 4 of argv) as integer
    tell application "Terminal"
        set viewerTab to do script ""
        set custom title of viewerTab to "DAVINCI ASCII Map"
        set font size of viewerTab to requestedFontSize
        repeat 10 times
            set number of columns of viewerTab to requestedColumns
            set number of rows of viewerTab to requestedRows
            delay 0.1
            if (number of rows of viewerTab is requestedRows) and (number of columns of viewerTab is requestedColumns) then exit repeat
        end repeat
        do script viewerCommand in viewerTab
        activate
        return ((number of rows of viewerTab) as text) & "x" & ((number of columns of viewerTab) as text)
    end tell
end run
"""


def open_terminal(arguments: Sequence[str], rows: int, columns: int, font_size: int = 12) -> str:
    """Open a dedicated Terminal tab/window using argument-safe AppleScript.

    Use this Python interpreter and package root, independent of the new login
    shell's conda activation and working directory. Never reuse an existing tab.
    """
    if sys.platform != "darwin":
        raise ValueError("--new-window currently supports macOS Terminal only.")
    if rows < 12 or columns < 45:
        raise ValueError("A new window needs at least 12 rows and 45 columns.")
    if not 6 <= font_size <= 72:
        raise ValueError("Terminal font size must be between 6 and 72 points.")
    root = str(Path(__file__).absolute().parents[2])
    command = shlex.join(
        [
            "env",
            f"PYTHONPATH={root}",
            sys.executable,
            "-m",
            "davinci_monet.terminal_map",
            *arguments,
        ]
    )
    try:
        result = subprocess.run(
            ["osascript", "-", str(rows), str(columns), command, str(font_size)],
            input=_APPLE_SCRIPT,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Terminal could not open the viewer: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Timed out opening Terminal; check macOS Automation permission.") from exc
    return result.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Browse NetCDF geographic fields as ASCII digits in your terminal.",
        epilog="YAML is optional. CLI options override YAML; defaults are a 180x360 grid and 10 bins.",
    )
    parser.add_argument("file", nargs="?", help="NetCDF file (or source.filename in YAML)")
    parser.add_argument("--config", "-c", help="Standalone viewer YAML configuration")
    parser.add_argument("--variable", "-v", help="Initial variable (v opens the picker)")
    parser.add_argument("--grid", nargs=2, type=int, metavar=("LAT", "LON"))
    parser.add_argument(
        "--level", type=int, help="Native vertical index, zero-based; negatives allowed"
    )
    parser.add_argument("--time", type=int, help="Time index, zero-based; negatives allowed")
    parser.add_argument("--index", action="append", default=[], metavar="DIM=INDEX")
    parser.add_argument("--method", choices=("linear", "nearest"))
    parser.add_argument("--bins", type=int, help="Number of numeric bins, 1-10 (default: 10)")
    parser.add_argument("--limits", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument(
        "--auto-limits", action="store_true", help="Override YAML limits with automatic"
    )
    parser.add_argument("--mask-below", type=float, help="Render values below this cutoff as '-'")
    parser.add_argument("--no-mask", action="store_true", help="Disable the YAML masking cutoff")
    parser.add_argument("--fit", action=argparse.BooleanOptionalAction, default=None)
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--new-window", nargs=2, type=int, metavar=("ROWS", "COLS"))
    window.add_argument("--current-window", action="store_true", help="Override YAML new_window")
    parser.add_argument(
        "--font-size", type=int, help="Font size for a new macOS Terminal (default: 12)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List map variables without loading fields"
    )
    parser.add_argument(
        "--print", dest="snapshot", action="store_true", help="Print one frame without curses"
    )
    parser.add_argument(
        "--size", nargs=2, type=int, metavar=("ROWS", "COLS"), help="Snapshot dimensions"
    )
    return parser


def _controls(args: argparse.Namespace) -> tuple[ViewerConfig, Path]:
    config = load_config(args.config)
    data = config.model_dump()
    source, display, terminal = data["source"], data["display"], data["terminal"]
    if args.file is not None:
        source["filename"] = args.file
    if args.variable is not None:
        source["variable"] = args.variable
    for item in args.index:
        try:
            dim, index = item.split("=", 1)
            if not dim:
                raise ValueError
            source["indices"][dim] = int(index)
        except ValueError as exc:
            raise ValueError(f"Invalid --index {item!r}; use DIM=INTEGER.") from exc
    if args.grid is not None:
        display["grid"] = dict(zip(("lat", "lon"), args.grid))
    for argument, key in (
        ("method", "interpolation"),
        ("bins", "bins"),
        ("fit", "fit"),
        ("limits", "limits"),
        ("mask_below", "mask_below"),
    ):
        value = getattr(args, argument)
        if value is not None:
            display[key] = value
    if args.auto_limits:
        display["limits"] = None
    if args.no_mask:
        display["mask_below"] = None
    if args.new_window is not None:
        terminal.update(new_window=True, rows=args.new_window[0], columns=args.new_window[1])
    elif args.current_window:
        terminal["new_window"] = False
    if args.font_size is not None:
        terminal["font_size"] = args.font_size
    config = ViewerConfig.model_validate(data)
    if not config.source.filename:
        raise ValueError("Provide a NetCDF file or set source.filename in --config YAML.")
    path = source_path(config.source.filename, args.config if args.file is None else None)
    if not path.is_file():
        raise ValueError(f"NetCDF file not found: {path}")
    return config, path


def _child_arguments(config: ViewerConfig, path: Path, args: argparse.Namespace) -> list[str]:
    display = config.display
    child = [
        str(path),
        "--grid",
        str(display.grid.lat),
        str(display.grid.lon),
        "--method",
        display.interpolation,
        "--bins",
        str(display.bins),
        "--current-window",
    ]
    if config.source.variable is not None:
        child.extend(["--variable", config.source.variable])
    for dim, index in config.source.indices.items():
        child.extend(["--index", f"{dim}={index}"])
    if display.fit:
        child.append("--fit")
    if display.limits is not None:
        child.extend(["--limits", *(str(value) for value in display.limits)])
    if display.mask_below is not None:
        child.extend(["--mask-below", str(display.mask_below)])
    for key in ("time", "level"):
        if getattr(args, key) is not None:
            child.extend([f"--{key}", str(getattr(args, key))])
    if args.snapshot:
        child.append("--print")
    if args.list:
        child.append("--list")
    if args.size is not None:
        child.extend(["--size", *(str(value) for value in args.size)])
    return child


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config, path = _controls(args)
        display, terminal = config.display, config.terminal
        grid = DisplayGrid(display.grid.lat, display.grid.lon)
        if args.size is not None and (args.size[0] < 12 or args.size[1] < 45):
            raise ValueError("Snapshot dimensions need at least 12 rows and 45 columns.")
        if terminal.new_window:
            size = open_terminal(
                _child_arguments(config, path, args),
                terminal.rows,
                terminal.columns,
                terminal.font_size,
            )
            requested = f"{terminal.rows}x{terminal.columns}"
            detail = f"requested {requested}; Terminal reported {size} rows x columns"
            print(f"Opened map in macOS Terminal ({detail}).")
            return 0
        with MapDataset.open(path) as dataset:
            if args.list:
                for variable in dataset.variables:
                    field = dataset.dataset[variable]
                    print(
                        ascii_text(
                            f"{variable}: {dict(field.sizes)} [{field.attrs.get('units', '')}]"
                        )
                    )
                return 0
            variable = config.source.variable or dataset.variables[0]
            indices = dict(config.source.indices)
            if variable not in dataset.variables:
                raise ValueError(f"{variable!r} is not a map variable; use --list to see choices.")
            for key, dim in (
                ("time", time_dimension(dataset.dataset[variable])),
                ("level", vertical_dim_name(dataset.dataset[variable])),
            ):
                index = getattr(args, key)
                if index is not None:
                    if dim is None:
                        raise ValueError(f"{variable} has no {key} dimension.")
                    indices[dim] = index
            view = MapView(
                dataset,
                variable,
                grid=grid,
                indices=indices,
                method=display.interpolation,
                limits=display.limits,
                mask_below=display.mask_below,
                bins=display.bins,
            )
            if args.snapshot:
                detected = shutil.get_terminal_size(fallback=(100, 35))
                rows, columns = args.size or (detected.lines, detected.columns)
                if display.fit:
                    height = rows - 6 - len(view.legend_lines(columns))
                    if height < 1:
                        raise ValueError("Snapshot is too small to fit the legend and map.")
                    view.change(grid=DisplayGrid(height, columns - 9))
                print("\n".join(render_frame(view, rows, columns)))
            else:
                if not sys.stdin.isatty() or not sys.stdout.isatty():
                    raise ValueError(
                        "Interactive mode requires a terminal; use --print or --new-window."
                    )
                from .tui import run

                run(view, fit=display.fit)
        return 0
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
