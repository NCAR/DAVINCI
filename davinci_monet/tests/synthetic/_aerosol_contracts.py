"""Contracts, locked scenario profiles, and deterministic RNG for FABLE synthetic data."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.tests.synthetic.generators import Domain, TimeConfig

SCHEMA_VERSION = "fable-synthetic-v1"
SERIALIZER_VERSION = "1"
SENSORS = ("sensor_a", "sensor_b")
SPLIT_NAMES = ("basis_train", "bias_fit", "calibration", "development_test")
DEFAULT_AEROSOL_SPECIES = (
    "DU001",
    "DU002",
    "DU003",
    "DU004",
    "DU005",
    "SS001",
    "SS002",
    "SS003",
    "SS004",
    "SS005",
    "SO4",
    "BCPHOBIC",
    "BCPHILIC",
    "OCPHOBIC",
    "OCPHILIC",
)
NAMED_STREAMS = (
    "model_residual",
    "correction_residual",
    "common_error",
    "sensor_a_cloud",
    "sensor_a_noise",
    "sensor_a_outages",
    "sensor_a_qa",
    "sensor_b_cloud",
    "sensor_b_noise",
    "sensor_b_outages",
    "sensor_b_qa",
    "holdout_noise",
    "mmr_perturbations",
)
SCENARIOS = {
    "exact_micro",
    "masked_chain_ci",
    "multi_sensor_ci",
    "writer_ci",
    "null_ci",
    "calibration_null",
    "low_aod_ci",
    "synthetic_osse",
    "synthetic_osse_null",
}
GRAVITY = 9.80665


@dataclass(frozen=True)
class SyntheticTuningSpec:
    """Immutable controls for one coupled synthetic aerosol-tuning case."""

    scenario: str = "exact_micro"
    master_seed: int = 20260710
    native_domain: Domain = field(default_factory=lambda: Domain(-180.0, 180.0, -90.0, 90.0, 12, 6))
    mode_domain: Domain = field(default_factory=lambda: Domain(-180.0, 180.0, -90.0, 90.0, 6, 3))
    time_config: TimeConfig = field(
        default_factory=lambda: TimeConfig("2001-01-01", "2001-01-12", "1h")
    )
    split_windows: tuple[tuple[str, str, str], ...] = ()
    n_modes: int = 3
    model_periods_days: tuple[float, ...] = (5.0, 8.0, 11.0)
    correction_periods_days: tuple[float, ...] = (4.0, 6.5, 9.0)
    local_overpass_hour: float = 13.5
    log_epsilon: float = 0.01
    r_bounds: tuple[float, float] = (0.2, 5.0)
    aod_floor: float = 0.001
    sensor_error_sigma: tuple[float, float] = (0.02, 0.03)
    common_error_sigma: float = 0.01
    sensor_bias_log: tuple[float, float] = (0.0, 0.0)
    heteroscedastic_strength: float = 0.0
    error_temporal_correlation: float = 0.0
    error_spatial_correlation: float = 0.0
    cloud_fraction: float = 0.25
    mnar_cloud_strength: float = 0.0
    qa_failure_fraction: float = 0.08
    basis_drift_amplitude: float = 0.0
    off_basis_amplitude: float = 0.018
    out_of_band_amplitude: float = 0.0
    out_of_band_period_days: float = 2.0
    correction_trend_per_year: float = 0.0
    filter_band_days: tuple[float, float] = (4.0, 180.0)
    filter_max_bridge_days: int = 7
    filter_min_segment_days: int = 360
    support_min_fraction: float = 0.2
    support_full_fraction: float = 0.5
    short_gap_days: int = 2
    long_gap_days: int = 8
    mmr_days: int = 2
    species: tuple[str, ...] = DEFAULT_AEROSOL_SPECIES

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {sorted(SCENARIOS)}")
        if (
            not isinstance(self.master_seed, int)
            or isinstance(self.master_seed, bool)
            or not 0 <= self.master_seed <= 2**63 - 1
        ):
            raise ValueError("master_seed must be an integer in [0, 2**63 - 1]")
        if not 0.0 <= self.local_overpass_hour < 24.0:
            raise ValueError("local_overpass_hour must be in [0, 24)")
        if self.log_epsilon <= 0.0:
            raise ValueError("log_epsilon must be positive")
        r_min, r_max = self.r_bounds
        if not 0.0 < r_min <= 1.0 <= r_max:
            raise ValueError("r_bounds must be positive and contain identity")
        if self.aod_floor < 0.0:
            raise ValueError("aod_floor must be nonnegative")
        if len(self.sensor_error_sigma) != len(SENSORS) or min(self.sensor_error_sigma) < 0.0:
            raise ValueError("sensor_error_sigma must contain two nonnegative values")
        if self.common_error_sigma < 0.0:
            raise ValueError("common_error_sigma must be nonnegative")
        if len(self.sensor_bias_log) != len(SENSORS) or not np.all(
            np.isfinite(self.sensor_bias_log)
        ):
            raise ValueError("sensor_bias_log must contain two finite values")
        for name, value in (
            ("heteroscedastic_strength", self.heteroscedastic_strength),
            ("basis_drift_amplitude", self.basis_drift_amplitude),
            ("off_basis_amplitude", self.off_basis_amplitude),
            ("out_of_band_amplitude", self.out_of_band_amplitude),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("error_temporal_correlation", self.error_temporal_correlation),
            ("error_spatial_correlation", self.error_spatial_correlation),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if not np.isfinite(self.mnar_cloud_strength) or self.mnar_cloud_strength < 0.0:
            raise ValueError("mnar_cloud_strength must be finite and nonnegative")
        if not np.isfinite(self.out_of_band_period_days) or self.out_of_band_period_days <= 0.0:
            raise ValueError("out_of_band_period_days must be positive and finite")
        if not np.isfinite(self.correction_trend_per_year):
            raise ValueError("correction_trend_per_year must be finite")
        band_min, band_max = self.filter_band_days
        if not 0.0 < band_min < band_max:
            raise ValueError("filter_band_days must be positive and increasing")
        if self.filter_max_bridge_days < 0:
            raise ValueError("filter_max_bridge_days must be nonnegative")
        if self.filter_min_segment_days < 2.0 * band_max:
            raise ValueError("filter_min_segment_days must be at least twice the maximum period")
        for name, value in (
            ("cloud_fraction", self.cloud_fraction),
            ("qa_failure_fraction", self.qa_failure_fraction),
            ("support_min_fraction", self.support_min_fraction),
            ("support_full_fraction", self.support_full_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.support_min_fraction >= self.support_full_fraction:
            raise ValueError("support_min_fraction must be below support_full_fraction")
        if self.short_gap_days < 0 or self.long_gap_days <= self.short_gap_days:
            raise ValueError("gap lengths must satisfy 0 <= short_gap_days < long_gap_days")
        if self.mmr_days < 2:
            raise ValueError("mmr_days must be at least two")
        if not self.species or len(set(self.species)) != len(self.species):
            raise ValueError("species must be nonempty and unique")

        native = _validated_domain(self.native_domain, "native_domain")
        mode = _validated_domain(self.mode_domain, "mode_domain")
        if not 1 <= self.n_modes < mode.n_lat * mode.n_lon:
            raise ValueError("n_modes must be smaller than the mode-grid cell count")
        if len(self.model_periods_days) != self.n_modes or min(self.model_periods_days) <= 0.0:
            raise ValueError("model_periods_days must contain one positive period per mode")
        if (
            len(self.correction_periods_days) != self.n_modes
            or min(self.correction_periods_days) <= 0.0
        ):
            raise ValueError("correction_periods_days must contain one positive period per mode")
        object.__setattr__(self, "native_domain", native)
        object.__setattr__(self, "mode_domain", mode)

        start = pd.Timestamp(self.time_config.start).normalize()
        end = pd.Timestamp(self.time_config.end).normalize()
        days = pd.date_range(start, end, freq="1D")
        if len(days) < max(8, self.mmr_days + 3):
            raise ValueError("time_config must contain enough days for four splits and MMR files")
        object.__setattr__(
            self, "time_config", TimeConfig(str(start.date()), str(end.date()), "1h")
        )
        windows = self.split_windows or _default_split_windows(days)
        _validate_split_windows(windows, days)
        object.__setattr__(self, "split_windows", tuple(windows))

    @classmethod
    def exact_micro(cls, master_seed: int = 20260710) -> SyntheticTuningSpec:
        """Return the compact, noiseless algebra case."""
        return cls(scenario="exact_micro", master_seed=master_seed)

    @classmethod
    def masked_chain_ci(cls, master_seed: int = 20260710) -> SyntheticTuningSpec:
        """Return the locked six-year, 12x24-to-6x12 pipeline case."""
        return cls(**_six_year_profile("masked_chain_ci", master_seed))

    @classmethod
    def multi_sensor_ci(cls, master_seed: int = 20260710) -> SyntheticTuningSpec:
        """Return a controlled complementary-footprint and precision case."""
        return cls(
            scenario="multi_sensor_ci",
            master_seed=master_seed,
            time_config=TimeConfig("2001-01-01", "2001-02-16", "1h"),
            sensor_error_sigma=(0.01, 0.04),
            common_error_sigma=0.0,
            cloud_fraction=0.0,
            qa_failure_fraction=0.0,
        )

    @classmethod
    def writer_ci(cls, master_seed: int = 20260710) -> SyntheticTuningSpec:
        """Return the locked six-year case with writer-oriented MMR inputs."""
        return cls(**_six_year_profile("writer_ci", master_seed))

    @classmethod
    def null_ci(cls, master_seed: int = 20260710) -> SyntheticTuningSpec:
        """Return a zero-correction noise and bounded-gap control case."""
        return cls(
            scenario="null_ci",
            master_seed=master_seed,
            time_config=TimeConfig("2001-01-01", "2001-04-30", "1h"),
            common_error_sigma=0.0,
        )

    @classmethod
    def calibration_null(cls, master_seed: int = 20260711) -> SyntheticTuningSpec:
        """Return the six-year null control for the exact frozen 4-180/360 policy."""
        return cls(**_six_year_profile("calibration_null", master_seed))

    @classmethod
    def low_aod_ci(cls, master_seed: int = 20260710) -> SyntheticTuningSpec:
        """Return a shifted-log boundary case with low AOD and both ratio clips."""
        return cls(scenario="low_aod_ci", master_seed=master_seed)

    @classmethod
    def synthetic_osse(cls, master_seed: int) -> SyntheticTuningSpec:
        """Return the opt-in eight-year stress case for a user-supplied seed."""
        return cls(**_synthetic_osse_profile("synthetic_osse", master_seed))

    @classmethod
    def synthetic_osse_null(cls, master_seed: int) -> SyntheticTuningSpec:
        """Return the full-size OSSE stress profile with zero physical correction."""
        return cls(**_synthetic_osse_profile("synthetic_osse_null", master_seed))

    def normalized(self) -> dict[str, Any]:
        """Return a stable JSON-safe representation for provenance hashing."""
        return {
            "scenario": self.scenario,
            "master_seed": self.master_seed,
            "native_domain": _domain_dict(self.native_domain),
            "mode_domain": _domain_dict(self.mode_domain),
            "time_config": {
                "start": str(self.time_config.start),
                "end": str(self.time_config.end),
                "freq": self.time_config.freq,
            },
            "split_windows": [list(window) for window in self.split_windows],
            "n_modes": self.n_modes,
            "model_periods_days": list(self.model_periods_days),
            "correction_periods_days": list(self.correction_periods_days),
            "local_overpass_hour": self.local_overpass_hour,
            "log_epsilon": self.log_epsilon,
            "r_bounds": list(self.r_bounds),
            "aod_floor": self.aod_floor,
            "sensor_error_sigma": list(self.sensor_error_sigma),
            "common_error_sigma": self.common_error_sigma,
            "sensor_bias_log": list(self.sensor_bias_log),
            "heteroscedastic_strength": self.heteroscedastic_strength,
            "error_temporal_correlation": self.error_temporal_correlation,
            "error_spatial_correlation": self.error_spatial_correlation,
            "cloud_fraction": self.cloud_fraction,
            "mnar_cloud_strength": self.mnar_cloud_strength,
            "qa_failure_fraction": self.qa_failure_fraction,
            "basis_drift_amplitude": self.basis_drift_amplitude,
            "off_basis_amplitude": self.off_basis_amplitude,
            "out_of_band_amplitude": self.out_of_band_amplitude,
            "out_of_band_period_days": self.out_of_band_period_days,
            "correction_trend_per_year": self.correction_trend_per_year,
            "filter_band_days": list(self.filter_band_days),
            "filter_max_bridge_days": self.filter_max_bridge_days,
            "filter_min_segment_days": self.filter_min_segment_days,
            "support_min_fraction": self.support_min_fraction,
            "support_full_fraction": self.support_full_fraction,
            "short_gap_days": self.short_gap_days,
            "long_gap_days": self.long_gap_days,
            "mmr_days": self.mmr_days,
            "species": list(self.species),
        }


def _synthetic_osse_profile(scenario: str, master_seed: int) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "master_seed": master_seed,
        "native_domain": Domain(-180.0, 180.0, -90.0, 90.0, 72, 36),
        "mode_domain": Domain(-180.0, 180.0, -90.0, 90.0, 36, 18),
        "time_config": TimeConfig("2001-01-01", "2008-12-31", "1h"),
        "split_windows": (
            ("basis_train", "2001-01-01", "2003-12-31"),
            ("bias_fit", "2004-01-01", "2005-12-31"),
            ("calibration", "2006-01-01", "2006-12-31"),
            ("development_test", "2007-01-01", "2008-12-31"),
        ),
        "model_periods_days": (12.0, 45.0, 120.0),
        "correction_periods_days": (20.0, 60.0, 150.0),
        "sensor_error_sigma": (0.035, 0.055),
        "common_error_sigma": 0.025,
        "sensor_bias_log": (0.015, -0.02),
        "heteroscedastic_strength": 0.7,
        "error_temporal_correlation": 0.55,
        "error_spatial_correlation": 0.6,
        "cloud_fraction": 0.35,
        "mnar_cloud_strength": 0.8,
        "qa_failure_fraction": 0.12,
        "basis_drift_amplitude": 0.15,
        "off_basis_amplitude": 0.03,
        "out_of_band_amplitude": 0.012,
        "out_of_band_period_days": 2.0,
        "correction_trend_per_year": 0.003,
    }


@dataclass(frozen=True)
class SyntheticTuningBundle:
    """In-memory synthetic inputs, hidden truth, and deterministic provenance."""

    spec: SyntheticTuningSpec
    model: xr.Dataset
    observations: Mapping[str, xr.Dataset]
    mmr: Mapping[str, xr.Dataset]
    truth: xr.Dataset
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ShiftedLogResult:
    """Independent shifted-log scaling result."""

    requested_ratio: np.ndarray
    applied_ratio: np.ndarray
    requested_aod: np.ndarray
    applied_aod: np.ndarray
    applied_delta: np.ndarray
    clip_mask: np.ndarray


def _six_year_profile(scenario: str, master_seed: int) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "master_seed": master_seed,
        "native_domain": Domain(-180.0, 180.0, -90.0, 90.0, 24, 12),
        "mode_domain": Domain(-180.0, 180.0, -90.0, 90.0, 12, 6),
        "time_config": TimeConfig("2001-01-01", "2006-12-31", "1h"),
        # Keep recovery signals clear of the 4- and 180-day CWT cutoffs. A
        # sinusoid centered on a hard scale boundary is intentionally split
        # across retained and rejected Morlet scales and is not an amplitude
        # recovery oracle.
        "correction_periods_days": (20.0, 60.0, 120.0),
        "split_windows": (
            ("basis_train", "2001-01-01", "2002-12-31"),
            ("bias_fit", "2003-01-01", "2004-12-31"),
            ("calibration", "2005-01-01", "2005-12-31"),
            ("development_test", "2006-01-01", "2006-12-31"),
        ),
    }


def _validated_domain(domain: Domain, name: str) -> Domain:
    if domain.n_lat < 2 or domain.n_lon < 4:
        raise ValueError(f"{name} must have at least 2 latitudes and 4 longitudes")
    if not -90.0 <= domain.lat_min < domain.lat_max <= 90.0:
        raise ValueError(f"{name} latitude bounds are invalid")
    if not (np.isclose(domain.lat_min, -90.0) and np.isclose(domain.lat_max, 90.0)):
        raise ValueError(f"{name} must span the global latitude range")
    if not np.isclose(domain.lon_max - domain.lon_min, 360.0):
        raise ValueError(f"{name} must span 360 degrees for periodic interpolation")
    return Domain(
        float(domain.lon_min),
        float(domain.lon_max),
        float(domain.lat_min),
        float(domain.lat_max),
        int(domain.n_lon),
        int(domain.n_lat),
    )


def _domain_dict(domain: Domain) -> dict[str, float | int]:
    return {
        "lon_min": domain.lon_min,
        "lon_max": domain.lon_max,
        "lat_min": domain.lat_min,
        "lat_max": domain.lat_max,
        "n_lon": domain.n_lon,
        "n_lat": domain.n_lat,
    }


def _default_split_windows(days: pd.DatetimeIndex) -> tuple[tuple[str, str, str], ...]:
    edges = np.linspace(0, len(days), len(SPLIT_NAMES) + 1, dtype=int)
    return tuple(
        (name, str(days[edges[index]].date()), str(days[edges[index + 1] - 1].date()))
        for index, name in enumerate(SPLIT_NAMES)
    )


def _validate_split_windows(
    windows: Sequence[tuple[str, str, str]], days: pd.DatetimeIndex
) -> None:
    if tuple(window[0] for window in windows) != SPLIT_NAMES:
        raise ValueError(f"split_windows must be ordered as {SPLIT_NAMES}")
    assigned: list[pd.Timestamp] = []
    for _name, raw_start, raw_end in windows:
        start = pd.Timestamp(raw_start).normalize()
        end = pd.Timestamp(raw_end).normalize()
        if start > end:
            raise ValueError("split window starts after it ends")
        assigned.extend(pd.date_range(start, end, freq="1D"))
    if not pd.DatetimeIndex(assigned).equals(days):
        raise ValueError("split_windows must cover the requested days once, contiguously")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def spec_hash(spec: SyntheticTuningSpec) -> str:
    return hashlib.sha256(canonical_json(spec.normalized()).encode("ascii")).hexdigest()


def named_stream_id(master_seed: int, stream_name: str) -> int:
    """Return the stable 128-bit identifier for a declared random stream."""
    if (
        not isinstance(master_seed, int)
        or isinstance(master_seed, bool)
        or not 0 <= master_seed <= 2**63 - 1
    ):
        raise ValueError("master_seed must be an integer in [0, 2**63 - 1]")
    if stream_name not in NAMED_STREAMS:
        raise KeyError(f"unknown synthetic random stream: {stream_name}")
    name_bytes = stream_name.encode("ascii")
    payload = b"".join(
        (
            SCHEMA_VERSION.encode("ascii"),
            master_seed.to_bytes(8, byteorder="little", signed=False),
            len(name_bytes).to_bytes(4, byteorder="little", signed=False),
            name_bytes,
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little", signed=False)


def named_rng(master_seed: int, stream_name: str) -> np.random.Generator:
    """Create an order-independent PCG64 generator for ``stream_name``."""
    return np.random.Generator(np.random.PCG64(named_stream_id(master_seed, stream_name)))


def provenance(spec: SyntheticTuningSpec, hash_value: str) -> dict[str, Any]:
    try:
        import netCDF4

        netcdf_version = netCDF4.__version__
    except ImportError:  # pragma: no cover
        netcdf_version = "unavailable"
    return {
        "schema_version": SCHEMA_VERSION,
        "serializer_version": SERIALIZER_VERSION,
        "scenario": spec.scenario,
        "root_seed": spec.master_seed,
        "spec_hash": hash_value,
        "spec": spec.normalized(),
        "stream_map": {name: named_stream_id(spec.master_seed, name) for name in NAMED_STREAMS},
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "xarray": xr.__version__,
            "netcdf4": netcdf_version,
        },
        "roles": {
            "model": "fitting_input",
            "sensor_a": "fitting_input",
            "sensor_b": "fitting_input",
            "mmr": "fitting_input",
            "truth": "evaluation_only",
        },
    }
