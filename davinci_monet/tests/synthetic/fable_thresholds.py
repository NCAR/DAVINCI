"""Anchor-free scientific thresholds shared by calibration and acceptance."""

RECOVERY_THRESHOLDS = {
    "field_correlation_min": 0.90,
    "field_origin_slope_min": 0.80,
    "field_origin_slope_max": 1.20,
    "field_nrmse_max": 0.35,
    "aod_rmse_ratio_max": 0.70,
    "full_target_aod_rmse_ratio_max": 1.0,
}

__all__ = ["RECOVERY_THRESHOLDS"]
