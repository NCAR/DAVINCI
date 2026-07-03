import numpy as np
import pytest

from davinci_monet.pairing.grid_binning import (
    bin_points_to_grid_4d,
    bin_swath_to_grid,
    bin_swath_to_grid_uncertainty,
    edges_from_centers,
    normalize_grid,
    normalize_uncertainty_grid,
)


def test_bin_points_to_grid_4d_accumulates_by_cell():
    # 2 time x 2 lon x 2 lat x 2 alt grid, edges 0..2 each
    edges = np.array([0.0, 1.0, 2.0])
    nt = nx = ny = nz = 2
    count = np.zeros((nt, nx, ny, nz), dtype=np.int32)
    acc = np.zeros((nt, nx, ny, nz), dtype=np.float64)
    # two points in cell (0,0,0,0), one in (1,1,1,1)
    t = np.array([0.5, 0.5, 1.5])
    x = np.array([0.5, 0.5, 1.5])
    y = np.array([0.5, 0.5, 1.5])
    z = np.array([0.5, 0.5, 1.5])
    d = np.array([2.0, 4.0, 9.0])
    bin_points_to_grid_4d(edges, edges, edges, edges, t, x, y, z, d, count, acc)
    normalize_grid(count, acc)
    assert count[0, 0, 0, 0] == 2 and acc[0, 0, 0, 0] == 3.0  # mean(2,4)
    assert count[1, 1, 1, 1] == 1 and acc[1, 1, 1, 1] == 9.0
    assert np.isnan(acc[0, 1, 0, 1])  # empty cell


def test_bin_swath_to_grid_nonuniform_lat_bins_by_true_edges():
    # Non-uniform lat centers [0,1,2,4,8] -> edges [-0.5,0.5,1.5,3,6,8.5].
    # Points at 3.9 and 4.1 must land in the cell CENTERED AT 4 (lat index 3),
    # not the first-spacing index-4 cell (centered at 8) the old arithmetic hit.
    lat_edges = edges_from_centers(np.array([0.0, 1.0, 2.0, 4.0, 8.0]))
    time_edges = np.array([0.0, 1.0])
    lon_edges = np.array([-0.5, 0.5])
    time = np.array([0.5, 0.5])
    lon = np.array([0.0, 0.0])
    lat = np.array([3.9, 4.1])
    data = np.array([10.0, 20.0])
    count = np.zeros((1, 1, 5), dtype=np.int64)
    acc = np.zeros((1, 1, 5), dtype=np.float64)
    bin_swath_to_grid(time_edges, lon_edges, lat_edges, time, lon, lat, data, count, acc)
    normalize_grid(count, acc)
    assert count[0, 0, 3] == 2
    assert acc[0, 0, 3] == pytest.approx(15.0)  # mean(10, 20)
    assert count[0, 0, 4] == 0  # NOT the centered-at-8 cell the bug would use
    assert np.isnan(acc[0, 0, 4])


def test_bin_swath_to_grid_uniform_grid_matches_expected():
    # Uniform grid behavior must be identical to the pre-searchsorted arithmetic,
    # including the edge conventions: an interior edge falls in the UPPER cell,
    # the first edge in bin 0, the last edge clamps into the final cell.
    lat_edges = np.array([0.0, 1.0, 2.0, 3.0])
    time_edges = np.array([0.0, 1.0])
    lon_edges = np.array([0.0, 1.0])
    time = np.full(6, 0.5)
    lon = np.full(6, 0.5)
    #                inside inside inside interior first-edge last-edge
    lat = np.array([0.5, 1.5, 2.5, 2.0, 0.0, 3.0])
    data = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 5.0])
    count = np.zeros((1, 1, 3), dtype=np.int64)
    acc = np.zeros((1, 1, 3), dtype=np.float64)
    bin_swath_to_grid(time_edges, lon_edges, lat_edges, time, lon, lat, data, count, acc)
    normalize_grid(count, acc)
    # bin0: 0.5 and 0.0 -> mean(1, 6); bin1: 1.5 -> 2.0;
    # bin2: 2.5, 2.0 (interior edge -> upper), 3.0 (last edge -> clamp) -> mean(3, 4, 5)
    assert count[0, 0, 0] == 2 and acc[0, 0, 0] == pytest.approx(3.5)
    assert count[0, 0, 1] == 1 and acc[0, 0, 1] == pytest.approx(2.0)
    assert count[0, 0, 2] == 3 and acc[0, 0, 2] == pytest.approx(4.0)


def test_bin_points_to_grid_4d_nonuniform_alt_bins_by_true_edges():
    alt_edges = edges_from_centers(np.array([0.0, 1.0, 2.0, 4.0, 8.0]))
    unit = np.array([0.0, 1.0])
    t = np.array([0.5, 0.5])
    x = np.array([0.5, 0.5])
    y = np.array([0.5, 0.5])
    z = np.array([3.9, 4.1])  # both in the cell centered at 4 -> alt index 3
    d = np.array([10.0, 20.0])
    count = np.zeros((1, 1, 1, 5), dtype=np.int64)
    acc = np.zeros((1, 1, 1, 5), dtype=np.float64)
    bin_points_to_grid_4d(unit, unit, unit, alt_edges, t, x, y, z, d, count, acc)
    normalize_grid(count, acc)
    assert count[0, 0, 0, 3] == 2 and acc[0, 0, 0, 3] == pytest.approx(15.0)
    assert count[0, 0, 0, 4] == 0


def test_uncertainty_binning_nonuniform_lat_bins_by_true_edges():
    lat_edges = edges_from_centers(np.array([0.0, 1.0, 2.0, 4.0, 8.0]))
    time_edges = np.array([0.0, 1.0])
    lon_edges = np.array([-0.5, 0.5])
    time = np.array([0.5, 0.5])
    lon = np.array([0.0, 0.0])
    lat = np.array([3.9, 4.1])
    data = np.array([0.4, 0.8])
    sigma = np.array([0.1, 0.1])  # equal sigma -> simple mean
    count = np.zeros((1, 1, 5), dtype=np.int64)
    weight = np.zeros((1, 1, 5))
    weighted_sum = np.zeros((1, 1, 5))
    bin_swath_to_grid_uncertainty(
        time_edges, lon_edges, lat_edges, time, lon, lat, data, sigma, count, weight, weighted_sum
    )
    assert count[0, 0, 3] == 2
    assert count[0, 0, 4] == 0
    mean, _ = normalize_uncertainty_grid(count, weight, weighted_sum)
    assert np.isclose(mean[0, 0, 3], (0.4 + 0.8) / 2)


def test_uncertainty_binning_inverse_variance_mean_and_sigma():
    edges = np.array([0.0, 1.0, 2.0])
    time = np.array([0.5, 0.5])
    lon = np.array([0.5, 0.5])
    lat = np.array([0.5, 0.5])
    data = np.array([0.4, 0.8])
    sigma = np.array([0.1, 0.2])
    count = np.zeros((1, 2, 2), dtype=np.int64)
    weight = np.zeros((1, 2, 2))
    weighted_sum = np.zeros((1, 2, 2))
    bin_swath_to_grid_uncertainty(
        np.array([0.0, 1.0]),
        edges,
        edges,
        time,
        lon,
        lat,
        data,
        sigma,
        count,
        weight,
        weighted_sum,
    )
    mean, combined_sigma = normalize_uncertainty_grid(count, weight, weighted_sum)
    w1, w2 = 1 / 0.1**2, 1 / 0.2**2
    assert count[0, 0, 0] == 2
    assert np.isclose(mean[0, 0, 0], (w1 * 0.4 + w2 * 0.8) / (w1 + w2))
    assert np.isclose(combined_sigma[0, 0, 0], np.sqrt(1 / (w1 + w2)))
    assert np.isnan(mean[0, 1, 1])
    assert np.isnan(combined_sigma[0, 1, 1])


def test_uncertainty_binning_skips_bad_sigma_and_matches_base_count():
    edges = np.array([0.0, 1.0, 2.0])
    time = np.array([0.5, 0.5, 0.5])
    lon = np.array([0.5, 0.5, 1.5])
    lat = np.array([0.5, 0.5, 1.5])
    data = np.array([0.4, 0.6, 0.9])
    sigma = np.array([0.1, -1.0, 0.2])
    count = np.zeros((1, 2, 2), dtype=np.int64)
    weight = np.zeros((1, 2, 2))
    weighted_sum = np.zeros((1, 2, 2))
    bin_swath_to_grid_uncertainty(
        np.array([0.0, 1.0]),
        edges,
        edges,
        time,
        lon,
        lat,
        data,
        sigma,
        count,
        weight,
        weighted_sum,
    )
    assert count[0, 0, 0] == 1
    assert count[0, 1, 1] == 1
    mean, _ = normalize_uncertainty_grid(count, weight, weighted_sum)
    assert np.isclose(mean[0, 0, 0], 0.4)

    base_count = np.zeros((1, 2, 2), dtype=np.int64)
    base_sum = np.zeros((1, 2, 2))
    bin_swath_to_grid(
        np.array([0.0, 1.0]),
        edges,
        edges,
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
        np.array([0.4, 0.8]),
        base_count,
        base_sum,
    )
    unc_count = np.zeros((1, 2, 2), dtype=np.int64)
    unc_weight = np.zeros((1, 2, 2))
    unc_weighted_sum = np.zeros((1, 2, 2))
    bin_swath_to_grid_uncertainty(
        np.array([0.0, 1.0]),
        edges,
        edges,
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
        np.array([0.4, 0.8]),
        np.full(2, 0.1),
        unc_count,
        unc_weight,
        unc_weighted_sum,
    )
    assert np.array_equal(base_count, unc_count)
    unc_mean, _ = normalize_uncertainty_grid(unc_count, unc_weight, unc_weighted_sum)
    assert np.isclose(unc_mean[0, 0, 0], (0.4 + 0.8) / 2)
