"""Named geographic domain extents.

This module is intentionally independent of plotting so statistics, pairing,
and renderers can share the same domain catalog without importing each other.
"""

from __future__ import annotations

Extent = tuple[float, float, float, float]


EPA_REGIONS: dict[str, Extent] = {
    "R1": (-73.5, -66.9, 40.5, 47.5),
    "R2": (-80.0, -71.8, 38.8, 45.0),
    "R3": (-83.7, -74.5, 36.5, 42.5),
    "R4": (-92.0, -75.0, 24.5, 39.5),
    "R5": (-97.5, -80.5, 36.0, 49.5),
    "R6": (-107.0, -88.5, 26.0, 37.0),
    "R7": (-104.5, -89.0, 36.0, 43.5),
    "R8": (-117.0, -96.0, 31.5, 49.0),
    "R9": (-125.0, -114.0, 32.0, 42.5),
    "R10": (-130.0, -116.0, 41.5, 49.5),
}


STANDARD_DOMAINS: dict[str, Extent] = {
    "conus": (-130.0, -60.0, 20.0, 55.0),
    "global": (-180.0, 180.0, -90.0, 90.0),
    "north_america": (-170.0, -50.0, 10.0, 75.0),
    "europe": (-15.0, 45.0, 35.0, 72.0),
    "asia": (60.0, 150.0, 0.0, 55.0),
    "asia_aq": (90.0, 140.0, 0.0, 45.0),  # ASIA-AQ campaign domain
}


def get_domain_extent(
    domain_type: str,
    domain_name: str | None = None,
) -> Extent | None:
    """Return ``(lon_min, lon_max, lat_min, lat_max)`` for a named domain."""
    if domain_type == "epa_region" and domain_name:
        return EPA_REGIONS.get(domain_name.upper())
    if domain_type in STANDARD_DOMAINS:
        return STANDARD_DOMAINS[domain_type]
    if domain_name and domain_name in STANDARD_DOMAINS:
        return STANDARD_DOMAINS[domain_name]
    return None
