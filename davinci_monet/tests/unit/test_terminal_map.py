"""Standalone map unit/CLI tests; these do not exercise the DAVINCI pipeline."""

from __future__ import annotations

import curses
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from davinci_monet.terminal_map import (
    BinScale,
    DisplayGrid,
    MapDataset,
    MapView,
    Viewport,
    ascii_cells,
    render_frame,
)
from davinci_monet.terminal_map.cli import main, open_terminal
from davinci_monet.terminal_map.config import ViewerConfig, load_config, source_path
from davinci_monet.terminal_map.data import interpolate
from davinci_monet.terminal_map.tui import CursesViewer


@pytest.fixture
def netcdf(tmp_path: Path) -> Path:
    lat = np.array([90, 45, 0, -45, -90])  # descending native latitude
    lon = np.array([0, 90, 180, 270])
    base = lat[:, None] / 45 + np.cos(np.deg2rad(lon))[None, :]
    volume = base[None, None, :, :] + np.array([0, 100])[:, None, None, None]
    volume = volume + np.array([0, 20])[None, :, None, None]
    ds = xr.Dataset(
        {
            "RH": (("time", "lev", "latitude", "longitude"), volume, {"units": "%"}),
            "T2M": (("longitude", "latitude"), (base + 280).T, {"units": "K"}),
            "LWI": (("latitude", "longitude"), np.tile([0, 1.00007, 1.9999, 0], (5, 1))),
            "other": ("time", [1, 2]),
        },
        coords={
            "latitude": lat,
            "longitude": lon,
            "time": np.array(["2026-04-01", "2026-04-02"], dtype="datetime64[ns]"),
            "lev": ("lev", [1, 72], {"positive": "down", "units": "layer"}),
        },
    )
    path = tmp_path / "synthetic.nc"
    ds.to_netcdf(path)
    return path


def test_native_file_selection_transpose_surface_and_cache(netcdf: Path) -> None:
    with MapDataset.open(netcdf) as dataset:
        assert dataset.variables == ["RH", "T2M", "LWI"]
        assert dataset.default_indices("RH") == {"time": 0, "lev": 1}
        selected = dataset.select("RH", {"time": -1, "lev": -1})
        assert selected.dims == ("lat", "lon")
        assert selected.sel(lat=0, lon=0).item() == 121
        assert selected is dataset.select("RH", {"time": 1, "lev": 1})
        assert dataset.select("T2M").sel(lat=45, lon=-180).item() == 280


@pytest.mark.parametrize("levels,expected", [([1000, 500, 100], 0), ([100, 500, 1000], 2)])
def test_pressure_surface_uses_maximum_pressure(levels: list[int], expected: int) -> None:
    ds = xr.Dataset(
        {"RH": (("plev", "lat", "lon"), np.ones((3, 2, 2)))},
        coords={
            "plev": ("plev", levels, {"units": "hPa", "standard_name": "air_pressure"}),
            "lat": [-90, 90],
            "lon": [-90, 90],
        },
    )
    assert MapDataset(ds).default_indices("RH") == {"plev": expected}


def test_cyclic_linear_interpolation_is_analytic_at_seam(netcdf: Path) -> None:
    with MapDataset.open(netcdf) as dataset:
        values = interpolate(dataset.select("RH", {"time": 0, "lev": 0}), DisplayGrid(1, 4))
        np.testing.assert_allclose(values, [[-0.5, 0.5, 0.5, -0.5]], atol=1e-12)
        assert np.isfinite(interpolate(dataset.select("T2M"), DisplayGrid(180, 360))).all()


@pytest.mark.parametrize(
    "longitudes,inside,outside",
    [
        ([10, 15, 20], 15, -175),
        ([10, 20], 15, -175),
        ([170, 175, 180, 185, 190], -175, 5),
    ],
)
def test_regional_grids_leave_unsampled_regions_missing(
    longitudes: list[int], inside: int, outside: int
) -> None:
    ds = xr.Dataset(
        {"height": (("lat", "lon"), np.ones((2, len(longitudes))))},
        coords={"lat": [-10, 10], "lon": longitudes},
    )
    dataset = MapDataset(ds)
    grid = DisplayGrid(9, 36)
    values = interpolate(dataset.select("height"), grid)
    row = int(np.argmin(abs(grid.latitudes)))
    assert values[row, int(np.argmin(abs(grid.longitudes - inside)))] == 1
    assert np.isnan(values[row, int(np.argmin(abs(grid.longitudes - outside)))])
    assert np.isnan(values[0]).all()


def test_cf_auxiliary_coordinates_and_unknown_extra_dimension() -> None:
    ds = xr.Dataset(
        {"field": (("member", "y", "x"), np.arange(8).reshape(2, 2, 2))},
        coords={
            "nav_lat": ("y", [-90, 90], {"standard_name": "latitude"}),
            "nav_lon": ("x", [-90, 90], {"standard_name": "longitude"}),
        },
    )
    dataset = MapDataset(ds)
    assert dataset.select("field", {"member": 1}).values.tolist() == [[4, 5], [6, 7]]
    with pytest.raises(ValueError, match="outside"):
        dataset.select("field", {"member": 2})
    with pytest.raises(ValueError, match="no selectable"):
        dataset.select("field", {"missing": 0})


def test_rejects_curvilinear_and_resource_fork(netcdf: Path) -> None:
    ds = xr.Dataset(
        {"field": (("y", "x"), np.ones((2, 2)))},
        coords={"lat": (("y", "x"), np.zeros((2, 2))), "lon": (("y", "x"), np.zeros((2, 2)))},
    )
    with pytest.raises(ValueError, match="rectilinear"):
        MapDataset(ds)
    with pytest.raises(ValueError, match="resource-fork"):
        MapDataset.open(netcdf.with_name("._synthetic.nc"))


def test_ten_bin_boundaries_missing_clipping_and_constant_field() -> None:
    values = np.array([-1, 0, 0.9, 1, 2, 9, 10, 11, np.nan, np.inf])
    scale = BinScale.from_values(values, limits=(0, 10))
    assert "".join(ascii_cells(values, scale)) == "00012999--"
    assert "9=[9,10]" in scale.legend()
    assert "".join(ascii_cells(np.array([-1.0, 0.0, 1.0]), scale, mask_below=0)) == "-01"
    constant = np.full((2, 2), 42.0)
    assert (ascii_cells(constant, BinScale.from_values(constant)) == "0").all()
    assert BinScale.from_values(np.array([np.nan])).empty
    assert BinScale.from_values(values, bins=5).edges.size == 6
    with pytest.raises(ValueError, match="MIN < MAX"):
        BinScale.from_values(values, limits=(1, 1))


def test_lwi_preserves_flags_and_forces_nearest(netcdf: Path) -> None:
    with MapDataset.open(netcdf) as dataset:
        view = MapView(dataset, "LWI", grid=DisplayGrid(9, 36))
        assert set(np.unique(view.chars)) == {"-", "1", "2"}
        np.testing.assert_allclose(np.unique(view.values), [0, 1.00007, 1.9999])
        assert "nearest" in " ".join(view.legend_lines(80))


def test_pan_wraps_poles_clamp_and_scale_is_stable(
    netcdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MapDataset.open(netcdf) as dataset:
        view = MapView(dataset, "T2M")
        source, edges = view.source, view.scale.edges.copy()
        height, width = view.map_size(24, 80)
        view.viewport = Viewport(0, 359, initialized=True)
        view.viewport.pan(-100, 2, view.grid, height, width)
        assert (view.viewport.top, view.viewport.left) == (0, 1)
        view.viewport.pan(10000, 0, view.grid, height, width)
        assert view.viewport.top == view.grid.rows - height
        with monkeypatch.context() as patch:
            patch.setattr(
                dataset, "select", lambda *a, **kw: pytest.fail("Panning reread the file")
            )
            assert render_frame(view, 24, 80)
        view.change(grid=DisplayGrid(90, 180))
        assert view.source is source
        np.testing.assert_array_equal(view.scale.edges, edges)


@pytest.mark.parametrize("size", [(12, 45), (24, 80), (40, 120), (6, 20)])
def test_rendered_frame_fits_ascii_terminal(netcdf: Path, size: tuple[int, int]) -> None:
    with MapDataset.open(netcdf) as dataset:
        frame = render_frame(MapView(dataset, "RH"), *size)
        assert len(frame) <= size[0]
        assert all(len(line) < size[1] and line.isascii() for line in frame)


def test_yaml_defaults_strictness_env_and_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = ViewerConfig()
    assert (defaults.display.grid.lat, defaults.display.grid.lon, defaults.display.bins) == (
        180,
        360,
        10,
    )
    path = tmp_path / "viewer.yaml"
    path.write_text("display:\n  bins: 11\n")
    with pytest.raises(ValueError, match="bins"):
        load_config(path)
    path.write_text("display:\n  grid: {lat: 90, lno: 180}\n")
    with pytest.raises(ValueError, match="lno"):
        load_config(path)
    path.write_text("display:\n  bins: 4\n  bins: 5\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(path)
    monkeypatch.setenv("ASCII_DATA", str(tmp_path))
    assert source_path("${ASCII_DATA}/field.nc") == tmp_path / "field.nc"
    assert source_path("field.nc", str(path)) == tmp_path / "field.nc"
    with pytest.raises(ValueError, match="Unresolved"):
        source_path("${ASCII_MAP_UNSET_VARIABLE}/field.nc")


def test_cli_yaml_and_override_through_real_script(netcdf: Path, tmp_path: Path) -> None:
    config = tmp_path / "viewer.yaml"
    config.write_text(
        "source:\n  filename: synthetic.nc\n  variable: RH\n  indices: {time: 1, lev: 0}\n"
        "display:\n  grid: {lat: 18, lon: 36}\n  bins: 5\n"
    )
    script = Path(__file__).absolute().parents[3] / "scripts" / "terminal_map.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--print",
            "--size",
            "24",
            "90",
            "--variable",
            "T2M",
            "--index",
            "time=0",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    # YAML indices remain explicit selections: switching to a dimensionless
    # variable must diagnose them rather than silently hiding a configuration error.
    assert result.returncode == 2
    assert "no selectable dimension" in result.stderr
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--print",
            "--size",
            "24",
            "90",
            "--grid",
            "9",
            "18",
            "--level",
            "1",
            "--bins",
            "3",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "grid 9x18" in result.stdout
    assert "time=1/1" in result.stdout and "lev=1/1" in result.stdout
    assert "2=[" in result.stdout and "3=[" not in result.stdout
    assert len(result.stdout.splitlines()) <= 24


def test_cli_list_does_not_load_slices(
    netcdf: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(MapDataset, "select", lambda *a, **kw: pytest.fail("--list loaded data"))
    assert main([str(netcdf), "--list"]) == 0
    assert "RH:" in capsys.readouterr().out


def test_cli_new_window_serializes_resolved_yaml(
    netcdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "window.yaml"
    config.write_text(
        "source:\n  filename: ${MISSING_BUT_OVERRIDDEN}/missing.nc\n  variable: LWI\n"
        "terminal:\n  new_window: true\n  rows: 35\n  columns: 110\n"
    )
    called: list[tuple[list[str], int, int, int]] = []

    def capture(arguments: list[str], rows: int, columns: int, font_size: int) -> str:
        called.append((arguments, rows, columns, font_size))
        return "35x110"

    monkeypatch.setattr("davinci_monet.terminal_map.cli.open_terminal", capture)
    assert main([str(netcdf), "--config", str(config), "--fit"]) == 0
    arguments, rows, columns, font_size = called[0]
    assert (rows, columns) == (35, 110)
    assert font_size == 12
    assert arguments[0] == str(netcdf)
    assert "--fit" in arguments and "--new-window" not in arguments and "--config" not in arguments


def test_terminal_launcher_keeps_shell_metacharacters_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    called = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        called.append((command, kwargs))
        return SimpleNamespace(stdout="40x120\n")

    monkeypatch.setattr(subprocess, "run", run)
    filename = "/Volumes/Io/a file ' $(not-a-command).nc4"
    assert open_terminal([filename, "--variable", "T2M"], 40, 120) == "40x120"
    command, kwargs = called[0]
    assert command[:4] == ["osascript", "-", "40", "120"]
    assert command[5] == "12"
    shell_args = shlex.split(command[4])
    assert shell_args[-3:] == [filename, "--variable", "T2M"]
    assert filename not in str(kwargs["input"])


class FakeScreen:
    def __init__(self, events: list[int | tuple[int, tuple[int, int]]]) -> None:
        self.events = iter(events)
        self.size = (30, 100)
        self.frames: list[list[str]] = []
        self.lines: list[str] = []

    def getmaxyx(self) -> tuple[int, int]:
        return self.size

    def keypad(self, enabled: bool) -> None:
        pass

    def erase(self) -> None:
        self.lines = []

    def clear(self) -> None:
        self.erase()

    def addnstr(self, row: int, column: int, text: str, count: int) -> None:
        assert 0 <= row < self.size[0] and len(text[:count]) < self.size[1]
        self.lines.append(text[:count])

    def refresh(self) -> None:
        self.frames.append(self.lines.copy())

    def getch(self) -> int:
        event = next(self.events)
        if isinstance(event, tuple):
            key, self.size = event
            return key
        return event


def test_curses_event_loop_level_time_resize_and_fit(
    netcdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(curses, "curs_set", lambda value: None)
    with MapDataset.open(netcdf) as dataset:
        view = MapView(dataset, "RH")
        screen = FakeScreen(
            [
                ord("z"),
                curses.KEY_HOME,
                10,
                ord("."),
                curses.KEY_RIGHT,
                (curses.KEY_RESIZE, (22, 75)),
                ord("f"),
                ord("q"),
            ]
        )
        CursesViewer(view).run(screen)
        assert view.indices == {"time": 1, "lev": 0}
        assert view.grid.columns == 66
        assert view.grid.rows < 22
        assert any("Select lev" in " ".join(frame) for frame in screen.frames)
        assert "terminal 22x75" in " ".join(screen.frames[-1])
