from __future__ import annotations

import hashlib
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.analysis import mmr_writer, mmr_writer_io
from davinci_monet.analysis.base import AnalysisExecutionError
from davinci_monet.analysis.mmr_writer import (
    DEFAULT_AEROSOL_SPECIES,
    interpolate_native_ratio,
    write_corrected_mmr_files,
)
from davinci_monet.analysis.mmr_writer_scaling import (
    ValidatedScaling,
    interpolate_validated_native_ratio,
)
from davinci_monet.config.schema import MMRWriterSpec
from davinci_monet.tests.synthetic.aerosol_tuning import (
    SyntheticTuningSpec,
    generate_aerosol_tuning_bundle,
    optical_aod_oracle,
    write_aerosol_tuning_bundle,
)


def _scaling(*, value: float = 2.0) -> xr.Dataset:
    time = pd.date_range("2000-01-01 12:00", periods=3, freq="1D")
    lat = np.array([-45.0, 45.0])
    lon = np.array([-135.0, -45.0, 45.0, 135.0])
    ratio = np.full((time.size, lat.size, lon.size), value, dtype=np.float64)
    return xr.Dataset(
        {
            "r": (("time", "lat", "lon"), ratio),
            "spatial_support": (("time", "lat", "lon"), np.ones_like(ratio)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"r_min": 0.2, "r_max": 5.0, "spec_hash": "synthetic-scenario"},
    )


def test_sparse_native_times_load_only_unique_source_brackets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = pd.date_range("2000-01-01", periods=1001, freq="1D")
    ratio = xr.DataArray(
        np.ones((times.size, 2, 4)),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [-45.0, 45.0], "lon": [-135.0, -45.0, 45.0, 135.0]},
    )
    validated = ValidatedScaling(
        ratio=ratio,
        support=xr.ones_like(ratio),
        time_ns=times.values.astype("datetime64[ns]").astype(np.int64),
    )
    loaded_time_sizes: list[int] = []
    original = xr.DataArray.load

    def observed_load(data: xr.DataArray, **kwargs: object) -> xr.DataArray:
        loaded_time_sizes.append(int(data.sizes.get("time", 0)))
        return original(data, **kwargs)

    monkeypatch.setattr(xr.DataArray, "load", observed_load)

    output = interpolate_validated_native_ratio(
        validated,
        np.array([times[0], times[-1]], dtype="datetime64[ns]"),
        np.array([-45.0, 45.0]),
        np.array([-135.0, -45.0, 45.0, 135.0]),
        outside_coverage="identity",
    )

    assert loaded_time_sizes == [2, 2]
    np.testing.assert_allclose(output["ratio"], 1.0)


def _write_native(
    path: Path,
    *,
    species: tuple[str, ...] = DEFAULT_AEROSOL_SPECIES,
    times: pd.DatetimeIndex | None = None,
) -> None:
    resolved_time = (
        times if times is not None else pd.date_range("2000-01-02", periods=2, freq="3h")
    )
    coords = {
        "time": resolved_time,
        "lev": np.array([20000.0, 100000.0]),
        "lat": np.array([45.0, -45.0]),
        "lon": np.array([45.0, -135.0, 135.0, -45.0]),
    }
    shape = (len(resolved_time), 2, 2, 4)
    variables: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, str]]] = {}
    for index, name in enumerate(species):
        values = np.full(shape, 1.0e-9 * (index + 1), dtype=np.float32)
        variables[name] = (
            ("time", "lev", "lat", "lon"),
            values,
            {"units": "kg kg-1", "source_attr": name},
        )
    if "DU001" in variables:
        variables["DU001"][1][0, 0, 0, 0] = np.float32(-9999.0)
        variables["DU001"][1][0, 0, 0, 1] = np.nan
    variables["SO2"] = (
        ("time", "lev", "lat", "lon"),
        np.full(shape, 3.0e-10, dtype=np.float32),
        {"units": "kg kg-1", "synthetic_role": "gas"},
    )
    variables["ORO"] = (
        ("lat", "lon"),
        np.arange(8, dtype=np.float32).reshape(2, 4),
        {"units": "m", "synthetic_role": "static"},
    )
    dataset = xr.Dataset(variables, coords=coords, attrs={"original": "preserve-me"})
    encoding = {
        name: {
            "zlib": True,
            "complevel": 2,
            "shuffle": True,
            "chunksizes": (1, 2, 2, 2),
            "_FillValue": np.float32(-9999.0),
        }
        for name in species
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
    if "DU001" in species:
        with netCDF4.Dataset(path, "r+") as serialized:
            raw = serialized.variables["DU001"]
            raw.set_auto_maskandscale(False)
            raw[0, 0, 0, 1] = np.float32(np.nan)


def _spec(input_path: Path, output_dir: Path, **updates: object) -> MMRWriterSpec:
    values: dict[str, object] = {
        "type": "mmr_writer",
        "scaling": "scaling",
        "files": str(input_path),
        "output_dir": str(output_dir),
    }
    values.update(updates)
    return MMRWriterSpec.model_validate(values)


def _raw(path: Path, variable: str) -> np.ndarray:
    with netCDF4.Dataset(path) as dataset:
        data = dataset.variables[variable]
        data.set_auto_maskandscale(False)
        return np.asarray(data[:]).copy()


def test_interpolation_is_periodic_log_linear_and_never_edge_holds() -> None:
    scaling = _scaling(value=1.0)
    scaling["r"][0] = 1.0
    scaling["r"][1] = 4.0
    scaling["r"][2] = 9.0
    target_time = np.array(
        ["1999-12-31T12", "2000-01-02T00", "2000-01-02T12"], dtype="datetime64[h]"
    )
    result = interpolate_native_ratio(
        scaling,
        target_time,
        [45.0, -45.0],
        [225.0, 45.0],
    )

    np.testing.assert_array_equal(result["lat"], [45.0, -45.0])
    np.testing.assert_array_equal(result["lon"], [225.0, 45.0])
    np.testing.assert_allclose(result["ratio"].isel(time=0), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result["ratio"].isel(time=1), 2.0, rtol=1.0e-12)
    np.testing.assert_allclose(result["ratio"].isel(time=2), 4.0, rtol=1.0e-12)
    np.testing.assert_array_equal(result["inside_coverage"], [False, True, True])
    with pytest.raises(ValueError, match="outside scaling coverage"):
        interpolate_native_ratio(
            scaling,
            target_time,
            [-45.0],
            [-135.0],
            outside_coverage="error",
        )


def test_writer_scales_all_species_and_preserves_native_file_contract(tmp_path: Path) -> None:
    input_path = tmp_path / "input" / "native.nc4"
    output_dir = tmp_path / "corrected"
    _write_native(input_path)
    source_species = {name: _raw(input_path, name) for name in DEFAULT_AEROSOL_SPECIES}
    assert source_species["DU001"][0, 0, 0, 0] == np.float32(-9999.0)
    assert np.isnan(source_species["DU001"][0, 0, 0, 1])
    source_gas = _raw(input_path, "SO2")
    source_static = _raw(input_path, "ORO")
    with netCDF4.Dataset(input_path) as source:
        metadata = {
            name: (
                source.variables[name].dtype,
                source.variables[name].dimensions,
                source.variables[name].chunking(),
                source.variables[name].filters(),
                {
                    attr: source.variables[name].getncattr(attr)
                    for attr in source.variables[name].ncattrs()
                },
            )
            for name in DEFAULT_AEROSOL_SPECIES
        }

    result = write_corrected_mmr_files(_scaling(), _spec(input_path, output_dir))
    output_path = output_dir / input_path.name

    assert result.dataset["status"].item() == "written"
    assert result.dataset.attrs["analysis_type"] == "mmr_writer"
    assert result.manifest_entries[0]["status"] == "written"
    assert len(result.manifest_entries[0]["checksums"]["payload_sha256"]) == 64
    assert (
        result.manifest_entries[0]["checksums"]["output_sha256"]
        == hashlib.sha256(output_path.read_bytes()).hexdigest()
    )
    assert (
        result.manifest_entries[0]["checksums"]["input_sha256"]
        == hashlib.sha256(input_path.read_bytes()).hexdigest()
    )
    np.testing.assert_array_equal(_raw(input_path, "SO2"), source_gas)
    np.testing.assert_array_equal(_raw(output_path, "SO2"), source_gas)
    np.testing.assert_array_equal(_raw(output_path, "ORO"), source_static)

    with netCDF4.Dataset(output_path) as output:
        assert output.getncattr("original") == "preserve-me"
        assert output.getncattr("davinci_scenario_hash") == "synthetic-scenario"
        assert (
            output.getncattr("davinci_payload_sha256")
            == result.manifest_entries[0]["checksums"]["payload_sha256"]
        )
        for name in DEFAULT_AEROSOL_SPECIES:
            variable = output.variables[name]
            dtype, dimensions, chunking, filters, attributes = metadata[name]
            assert variable.dtype == dtype
            assert variable.dimensions == dimensions
            assert variable.chunking() == chunking
            assert variable.filters() == filters
            assert {attr: variable.getncattr(attr) for attr in variable.ncattrs()} == attributes
            actual = _raw(output_path, name)
            source_values = source_species[name]
            valid = np.isfinite(source_values) & (source_values != np.float32(-9999.0))
            np.testing.assert_allclose(actual[valid], source_values[valid] * 2.0, rtol=5.0e-6)
            np.testing.assert_array_equal(actual[~valid], source_values[~valid])


def test_preflight_rejects_incomplete_species_before_any_output(tmp_path: Path) -> None:
    first = tmp_path / "input" / "a.nc4"
    second = tmp_path / "input" / "b.nc4"
    output_dir = tmp_path / "corrected"
    _write_native(first)
    _write_native(second, species=DEFAULT_AEROSOL_SPECIES[:-1])

    with pytest.raises(ValueError, match="missing configured aerosol species"):
        write_corrected_mmr_files(_scaling(), _spec(first.parent / "*.nc4", output_dir))
    assert not output_dir.exists()


def test_writer_validates_scaling_collection_once_for_multiple_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    _write_native(input_dir / "a.nc4")
    _write_native(input_dir / "b.nc4")
    validation_count = 0
    original = mmr_writer._validated_scaling

    def counted(scaling: xr.Dataset) -> object:
        nonlocal validation_count
        validation_count += 1
        return original(scaling)

    monkeypatch.setattr(mmr_writer, "_validated_scaling", counted)

    result = write_corrected_mmr_files(
        _scaling(),
        _spec(input_dir / "*.nc4", tmp_path / "corrected"),
    )

    assert result.dataset.sizes["file"] == 2
    assert validation_count == 1


def test_collision_resume_overwrite_and_alias_policies(tmp_path: Path) -> None:
    input_path = tmp_path / "input" / "native.nc4"
    output_dir = tmp_path / "corrected"
    _write_native(input_path)
    base = _spec(input_path, output_dir)
    write_corrected_mmr_files(_scaling(), base)

    with pytest.raises(FileExistsError, match="output already exists"):
        write_corrected_mmr_files(_scaling(), base)
    resumed = write_corrected_mmr_files(_scaling(), _spec(input_path, output_dir, resume=True))
    assert resumed.dataset["status"].item() == "resumed"

    with netCDF4.Dataset(input_path, "r+") as dataset:
        dataset.variables["SO2"][0, 0, 0, 0] = np.float32(7.0e-10)
    with pytest.raises(ValueError, match="resume hashes do not match"):
        write_corrected_mmr_files(_scaling(), _spec(input_path, output_dir, resume=True))
    rewritten = write_corrected_mmr_files(
        _scaling(), _spec(input_path, output_dir, resume=True, overwrite=True)
    )
    assert rewritten.dataset["status"].item() == "written"
    assert _raw(output_dir / input_path.name, "SO2")[0, 0, 0, 0] == pytest.approx(7.0e-10)

    with pytest.raises(ValueError, match="input and output paths alias"):
        write_corrected_mmr_files(_scaling(), _spec(input_path, input_path.parent))


def test_resume_rejects_payload_tampering_even_when_provenance_matches(tmp_path: Path) -> None:
    input_path = tmp_path / "input" / "native.nc4"
    output_dir = tmp_path / "corrected"
    _write_native(input_path)
    write_corrected_mmr_files(_scaling(), _spec(input_path, output_dir))
    output_path = output_dir / input_path.name
    with netCDF4.Dataset(output_path, "r+") as dataset:
        variable = dataset.variables["DU002"]
        variable.set_auto_maskandscale(False)
        variable[0, 0, 0, 0] = variable[0, 0, 0, 0] * np.float32(7.0)

    with pytest.raises(ValueError, match="payload checksum does not match"):
        write_corrected_mmr_files(_scaling(), _spec(input_path, output_dir, resume=True))

    rewritten = write_corrected_mmr_files(
        _scaling(), _spec(input_path, output_dir, resume=True, overwrite=True)
    )
    assert rewritten.dataset["status"].item() == "written"


def test_multi_file_failure_carries_finalized_receipts(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    first = input_dir / "a.nc4"
    second = input_dir / "b.nc4"
    output_dir = tmp_path / "corrected"
    _write_native(first)
    _write_native(second)
    original = mmr_writer_io._mutate_aerosols
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-file failure")
        original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mmr_writer_io, "_mutate_aerosols", fail_second)

    with pytest.raises(AnalysisExecutionError, match="second-file failure") as raised:
        write_corrected_mmr_files(_scaling(), _spec(input_dir / "*.nc4", output_dir))

    entries = raised.value.manifest_entries
    assert len(entries) == 1
    assert entries[0]["path"] == str(output_dir / first.name)
    assert entries[0]["status"] == "written"
    assert len(entries[0]["checksums"]["output_sha256"]) == 64
    assert len(entries[0]["checksums"]["payload_sha256"]) == 64
    assert (output_dir / first.name).is_file()
    assert not (output_dir / second.name).exists()


def test_skip_policy_and_atomic_failure_leave_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "input" / "outside.nc4"
    _write_native(
        outside,
        times=pd.date_range("1999-01-01", periods=2, freq="3h"),
    )
    skipped_dir = tmp_path / "skipped"
    skipped = write_corrected_mmr_files(
        _scaling(), _spec(outside, skipped_dir, outside_coverage="skip")
    )
    assert skipped.dataset["status"].item() == "skipped_outside_coverage"
    assert not (skipped_dir / outside.name).exists()

    inside = tmp_path / "input" / "inside.nc4"
    output_dir = tmp_path / "corrected"
    _write_native(inside)
    output_dir.mkdir()
    existing = output_dir / inside.name
    existing.write_bytes(b"existing-output")

    def fail_mutation(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected mutation failure")

    monkeypatch.setattr(mmr_writer_io, "_mutate_aerosols", fail_mutation)
    with pytest.raises(RuntimeError, match="injected mutation failure"):
        write_corrected_mmr_files(_scaling(), _spec(inside, output_dir, overwrite=True))
    assert existing.read_bytes() == b"existing-output"
    assert list(output_dir.glob(f".{inside.name}.*.tmp")) == []


def test_writer_ci_matches_independent_native_optical_oracle(tmp_path: Path) -> None:
    bundle = generate_aerosol_tuning_bundle(SyntheticTuningSpec.writer_ci(20260710))
    bundle_root = tmp_path / "bundle"
    write_aerosol_tuning_bundle(bundle_root, bundle)
    truth = bundle.truth
    time = truth["r_applied_true"]["time"]
    month_index = xr.DataArray(time.dt.month.values, dims=("time",), coords={"time": time})
    support = truth["spatial_support_true"].sel(month=month_index)
    scaling = xr.Dataset(
        {
            "r": truth["r_applied_true"].rename(mode_lat="lat", mode_lon="lon"),
            "spatial_support": support.rename(mode_lat="lat", mode_lon="lon"),
        },
        attrs={
            "r_min": 0.2,
            "r_max": 5.0,
            "spec_hash": bundle.provenance["spec_hash"],
        },
    )
    output_dir = tmp_path / "corrected"
    result = write_corrected_mmr_files(
        scaling,
        _spec(bundle_root / "inputs" / "mmr" / "*.nc4", output_dir),
    )

    corrected_days: list[xr.Dataset] = []
    for path in sorted(output_dir.glob("*.nc4")):
        with xr.open_dataset(path) as dataset:
            corrected_days.append(dataset.load())
    corrected = xr.concat(corrected_days, dim="time", data_vars="all")
    optical_aod = optical_aod_oracle(corrected)
    expected = truth["scaled_optical_aod"].rename(
        mmr_time="time", native_lat="lat", native_lon="lon"
    )
    ratio_expected = truth["r_3hour_true"].rename(
        mmr_time="time", native_lat="lat", native_lon="lon"
    )
    baseline = truth["baseline_optical_aod"].rename(
        mmr_time="time", native_lat="lat", native_lon="lon"
    )

    xr.testing.assert_allclose(
        optical_aod.where(np.isfinite(expected)),
        expected.where(np.isfinite(expected)),
        rtol=5.0e-6,
        atol=1.0e-10,
    )
    xr.testing.assert_allclose(
        (optical_aod / baseline).where(np.isfinite(baseline)),
        ratio_expected.where(np.isfinite(baseline)),
        rtol=5.0e-6,
        atol=1.0e-10,
    )
    assert result.dataset.sizes["file"] == 2
    assert set(result.dataset["status"].values.tolist()) == {"written"}
