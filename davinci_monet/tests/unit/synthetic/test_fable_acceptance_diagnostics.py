"""Plot-source and pipeline tests for FABLE acceptance diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

import davinci_monet.tests.synthetic.fable_acceptance_diagnostics as diagnostics  # noqa: E402
import davinci_monet.tests.synthetic.fable_acceptance_provenance as provenance  # noqa: E402
from davinci_monet.pipeline.runner import PipelineRunner  # noqa: E402
from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.registry import has_plotter  # noqa: E402
from davinci_monet.tests.synthetic._aerosol_io import scientific_dataset_hash  # noqa: E402
from davinci_monet.tests.synthetic.fable_acceptance_plots import (  # noqa: E402
    FableEOFComparisonPlotter,
    FablePCReconstructionPlotter,
    FableSpatialRecoveryPlotter,
    FableWaveletScalogramPlotter,
    registered_acceptance_plotters,
)


def _diagnostic_dataset() -> xr.Dataset:
    seed = [101]
    mode = [1, 2, 3]
    time = pd.date_range("2007-01-01", periods=24, freq="D")
    lat = [-45.0, 0.0, 45.0]
    lon = [-120.0, 0.0, 120.0]
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    pattern = np.stack(
        (
            np.sin(np.deg2rad(xx)),
            np.sin(np.deg2rad(yy)),
            np.cos(np.deg2rad(xx)) * np.cos(np.deg2rad(yy)),
        )
    )
    phase = np.arange(len(time), dtype=float)
    raw = np.stack(
        (
            0.08 * np.sin(2.0 * np.pi * phase / 8.0),
            0.05 * np.sin(2.0 * np.pi * phase / 12.0),
            np.zeros_like(phase),
        ),
        axis=1,
    )
    reconstructed = 0.85 * raw
    truth_target = 0.9 * raw
    spatial = np.sin(np.deg2rad(xx)) * np.cos(np.deg2rad(yy))
    scalar_seed = ("seed", [0.25])
    dataset = xr.Dataset(
        {
            "truth_eof": (("seed", "mode", "lat", "lon"), pattern[None]),
            "learned_eof_aligned": (("seed", "mode", "lat", "lon"), 0.98 * pattern[None]),
            "eof_residual": (("seed", "mode", "lat", "lon"), -0.02 * pattern[None]),
            "truth_snapshot": (("seed", "lat", "lon"), spatial[None]),
            "estimate_snapshot": (("seed", "lat", "lon"), 0.8 * spatial[None]),
            "residual_snapshot": (("seed", "lat", "lon"), -0.2 * spatial[None]),
            "truth_correction_rms": (("seed", "lat", "lon"), np.abs(spatial)[None]),
            "estimate_correction_rms": (
                ("seed", "lat", "lon"),
                0.8 * np.abs(spatial)[None],
            ),
            "residual_correction_rms": (
                ("seed", "lat", "lon"),
                0.2 * np.abs(spatial)[None],
            ),
            "raw_projected_pc": (("seed", "time", "mode"), raw[None]),
            "raw_truth_pc": (("seed", "time", "mode"), raw[None]),
            "wavelet_reconstruction_pc": (
                ("seed", "time", "mode"),
                reconstructed[None],
            ),
            "wavelet_truth_target_pc": (
                ("seed", "time", "mode"),
                truth_target[None],
            ),
            "raw_eligible": (("seed", "time", "mode"), np.ones((1, 24, 3), dtype=bool)),
            "wavelet_valid_segment": (
                ("seed", "time", "mode"),
                np.ones((1, 24, 3), dtype=bool),
            ),
            "wavelet_coi_safe": (
                ("seed", "time", "mode"),
                np.ones((1, 24, 3), dtype=bool),
            ),
            "mode_observable": (("seed", "mode"), [[True, True, False]]),
            "mode_similarity": (("seed", "mode"), [[0.99, 0.98, 0.95]]),
            "explained_variance": (("seed", "mode"), [[0.6, 0.3, 0.1]]),
            "coefficient_correlation": (("seed", "mode"), [[0.95, 0.75, np.nan]]),
            "coefficient_origin_slope": (("seed", "mode"), [[0.9, 0.6, np.nan]]),
            "coefficient_nrmse": (("seed", "mode"), [[0.3, 0.6, np.nan]]),
            "subspace_angle_mean_degrees": scalar_seed,
            "subspace_angle_max_degrees": ("seed", [0.5]),
            "subspace_projector_error": ("seed", [0.02]),
            "snapshot_valid_count": ("seed", [9]),
            "snapshot_nrmse": ("seed", [0.2]),
            "primary_valid_count": ("seed", [216]),
            "field_nrmse": ("seed", [0.25]),
        },
        coords={"seed": seed, "mode": mode, "time": time, "lat": lat, "lon": lon},
        attrs={
            "snapshot_time": "2007-01-12T00:00:00",
            "selected_policy_id": "fable-v1-all-band",
        },
    )
    return dataset


def test_matching_uses_reconstruction_preserving_scale_direction() -> None:
    truth = xr.DataArray(
        np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0,
        dims=("mode", "lat", "lon"),
        coords={"mode": [1, 2], "lat": [-30.0, 30.0], "lon": [0.0, 120.0, 240.0]},
    )
    scales = np.array([2.0, -3.0])
    estimate = truth * xr.DataArray(scales, dims="mode", coords={"mode": truth["mode"]})

    selected_truth, aligned = diagnostics._align_matched_basis(
        estimate,
        truth,
        np.array([0, 1]),
        np.array([0, 1]),
        scales,
    )

    xr.testing.assert_allclose(aligned, selected_truth)


def test_matched_estimate_values_follow_nonidentity_permutation() -> None:
    values = xr.DataArray(
        [0.7, 0.2, 0.1],
        dims="mode",
        coords={"mode": [1, 2, 3]},
    )
    truth_mode = xr.DataArray([1, 2, 3], dims="mode", coords={"mode": [1, 2, 3]})

    matched = diagnostics._matched_estimate_modes(
        values,
        np.array([1, 0, 2]),
        truth_mode,
    )

    np.testing.assert_allclose(matched.values, [0.2, 0.7, 0.1])
    np.testing.assert_array_equal(matched["mode"].values, [1, 2, 3])


def test_primary_mask_applies_day_support_and_finite_contract() -> None:
    time = pd.date_range("2007-01-01", periods=2, freq="D")
    truth = xr.DataArray(
        np.ones((2, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-30.0, 30.0], "lon": [0.0, 180.0]},
    )
    estimate = truth.copy()
    estimate[0, 0, 0] = np.nan
    support = xr.ones_like(truth)
    support[0, 1, 1] = 0.0
    score_day = xr.DataArray([True, False], dims="time", coords={"time": time})

    mask = diagnostics._acceptance_primary_mask(score_day, support, truth, estimate)

    assert int(mask.sum().item()) == 2


def test_seed_receipts_bind_plotted_inputs_to_frozen_bytes(tmp_path: Path, monkeypatch) -> None:
    seed = 1179
    seed_root = tmp_path / f"seed-{seed}"
    fitting_manifest = seed_root / "output/manifest.json"
    evaluation_manifest = seed_root / "evaluation/manifest.json"
    recovery = seed_root / "evaluation/artifacts/recovery"
    truth = seed_root / "oracle/truth.nc"
    for path in (fitting_manifest.parent, evaluation_manifest.parent, recovery, truth.parent):
        path.mkdir(parents=True, exist_ok=True)
    fitting_manifest.write_text("{}", encoding="utf-8")
    evaluation_manifest.write_text("{}", encoding="utf-8")
    xr.Dataset({"truth": ("sample", [1.0, 2.0])}).to_netcdf(truth)
    for analysis in ("aod_basis", "obs_pcs", "scaling"):
        artifact = seed_root / f"output/artifacts/{analysis}/chunk-00000.nc"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(analysis.encode("ascii"))
    recovery_file = recovery / "chunk-00000.nc"
    recovery_file.write_bytes(b"recovery")
    with xr.open_dataset(truth) as truth_dataset:
        scientific_hash = scientific_dataset_hash(truth_dataset)
    scenario = seed_root / "scenario.json"
    scenario.write_text(
        json.dumps(
            {
                "root_seed": seed,
                "scenario": "synthetic_osse",
                "files": {
                    "oracle/truth.nc": {
                        "role": "evaluation_only:oracle",
                        "byte_sha256": hashlib.sha256(truth.read_bytes()).hexdigest(),
                        "scientific_sha256": scientific_hash,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def identity(path: Path) -> dict[str, str]:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    run = {
        "scenario": "synthetic_osse",
        "scenario_manifest": identity(scenario),
        "fitting": {"manifest": identity(fitting_manifest)},
        "evaluation": {
            "manifest": identity(evaluation_manifest),
            "recovery_artifact": {
                "artifact_dir": str(recovery),
                "checksums": {"files": {recovery_file.name: "0" * 64}},
            },
        },
    }
    validated: list[tuple[str, str]] = []

    def validate_artifact(manifest, paths, *, role, analysis):
        assert Path(manifest) == fitting_manifest
        assert list(paths)
        validated.append((analysis, role))
        return {}

    monkeypatch.setattr(
        provenance,
        "validate_finalized_artifact_manifest",
        validate_artifact,
    )

    provenance.validate_seed_receipts(tmp_path, seed, run)

    assert validated == [
        ("aod_basis", "basis_fit"),
        ("obs_pcs", "projection_fit"),
        ("scaling", "scaling"),
    ]
    truth.write_bytes(b"mutated truth")
    with pytest.raises(ValueError, match="oracle checksum"):
        provenance.validate_seed_receipts(tmp_path, seed, run)


def test_diagnostic_renderers_return_expected_pages(monkeypatch) -> None:
    dataset = _diagnostic_dataset()
    spatial = FableSpatialRecoveryPlotter()
    eof = FableEOFComparisonPlotter()
    monkeypatch.setattr(spatial, "add_map_features", lambda *args, **kwargs: None)
    monkeypatch.setattr(eof, "add_map_features", lambda *args, **kwargs: None)

    spatial_pages = spatial.render(build_series(dataset, "truth_snapshot"))
    eof_pages = eof.render(build_series(dataset, "truth_eof"))
    pc_pages = FablePCReconstructionPlotter().render(
        build_series(dataset, "wavelet_reconstruction_pc")
    )

    assert [label for label, _ in spatial_pages] == ["snapshot", "temporal_rms"]
    assert [label for label, _ in eof_pages] == ["seed_101"]
    assert [label for label, _ in pc_pages] == ["seed_101"]
    assert len(pc_pages[0][1].axes) == 4
    for _, figure in (*spatial_pages, *eof_pages, *pc_pages):
        plt.close(figure)


def test_acceptance_wavelet_title_does_not_duplicate_subtitle() -> None:
    time = pd.date_range("2007-01-01", periods=16, freq="D")
    period = np.array([2.0, 4.0, 8.0, 16.0])
    spectrum = xr.Dataset(
        {"power": (("time", "period"), np.ones((len(time), len(period))))},
        coords={"time": time, "period": ("period", period, {"units": "days"})},
        attrs={"wavelet_quantity": "pc"},
    )
    plotter = FableWaveletScalogramPlotter()
    plotter.config.title = "FABLE projected PC wavelet power | seed 1179, mode 2"
    plotter.config.subtitle = "2007-01-01 to 2007-01-16"

    figure = plotter.render(build_series(spectrum, "power"))

    assert figure.axes[0].get_title() == plotter.config.title
    subtitles = [
        text for text in figure.axes[0].texts if text.get_text() == plotter.config.subtitle
    ]
    assert len(subtitles) == 1
    plt.close(figure)


def test_acceptance_plotter_registration_cleans_up_after_error() -> None:
    names = (
        "fable_spatial_recovery",
        "fable_eof_comparison",
        "fable_pc_reconstruction",
        "fable_wavelet_scalogram",
    )
    assert not any(has_plotter(name) for name in names)

    with pytest.raises(RuntimeError, match="registration probe"):
        with registered_acceptance_plotters():
            assert all(has_plotter(name) for name in names)
            raise RuntimeError("registration probe")

    assert not any(has_plotter(name) for name in names)


def test_pc_diagnostic_plot_runs_through_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "diagnostics.nc"
    _diagnostic_dataset().to_netcdf(source)
    variables = {name: {"units": "1"} for name in _diagnostic_dataset().data_vars}
    config = {
        "analysis": {
            "start_time": "2007-01-01",
            "end_time": "2007-01-24 23:59:59",
            "output_dir": str(tmp_path / "out"),
        },
        "sources": {
            "diagnostics": {
                "type": "generic",
                "files": str(source),
                "variables": variables,
            }
        },
        "plots": {
            "pc": {
                "type": "fable_pc_reconstruction",
                "source": "diagnostics",
                "variable": "wavelet_reconstruction_pc",
                "formats": ["png"],
            }
        },
    }

    with registered_acceptance_plotters():
        result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    plots = result.context.results["plotting"].data["plots_generated"]
    assert len(plots) == 1
    assert Path(plots[0]).is_file()
