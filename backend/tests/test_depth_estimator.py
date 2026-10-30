import pytest
from backend.app.depth_estimator import estimate_depth_cm
from backend.app.policy import classify_vehicle_risk, classify_road_status


def test_half_submerged_60cm_reference_is_30cm():
    result = estimate_depth_cm(reference_height_cm=60, y_top=100, y_base=200, y_waterline=150)
    assert result.estimated_depth_cm == 30.0
    assert result.submerged_ratio == 0.5


def test_invalid_reference_returns_error():
    with pytest.raises(ValueError):
        estimate_depth_cm(reference_height_cm=60, y_top=200, y_base=100, y_waterline=150)


def test_waterline_outside_reference_abstains_via_error():
    with pytest.raises(ValueError):
        estimate_depth_cm(reference_height_cm=60, y_top=100, y_base=200, y_waterline=220)


def test_demo_policy_maps_32cm():
    assert classify_vehicle_risk(32, "two_wheeler") == "BLOCKED"
    assert classify_vehicle_risk(32, "light_vehicle") == "HIGH RISK"
    assert classify_vehicle_risk(32, "higher_clearance") == "CONFIGURABLE / HUMAN REVIEW"
    assert classify_road_status(32) == "HIGH RISK"
