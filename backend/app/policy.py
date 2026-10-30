"""Configurable illustrative policy defaults for the hackathon demo only."""
import json
from pathlib import Path
from typing import Literal

VehicleClass = Literal["two_wheeler", "light_vehicle", "higher_clearance"]
_POLICY_PATH = Path(__file__).with_name("policy.json")


def load_policy() -> dict:
    """Load policy settings from JSON so demo thresholds are easy to change."""
    with _POLICY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def classify_vehicle_risk(depth_cm: float, vehicle_class: VehicleClass) -> str:
    if depth_cm < 0:
        raise ValueError("depth_cm cannot be negative")
    policy = load_policy()
    rule = policy["vehicles"][vehicle_class]
    if rule.get("human_review_required"):
        return "CONFIGURABLE / HUMAN REVIEW"
    if depth_cm >= rule.get("blocked_from_cm", float("inf")):
        return "BLOCKED"
    if depth_cm >= rule.get("high_risk_from_cm", float("inf")):
        return "HIGH RISK"
    if depth_cm >= rule.get("caution_from_cm", float("inf")):
        return "CAUTION"
    return "MONITOR"


def classify_road_status(depth_cm: float) -> str:
    if depth_cm < 0:
        raise ValueError("depth_cm cannot be negative")
    bands = load_policy()["depth_bands_cm"]
    if depth_cm >= bands["closure_review_from"]:
        return "CLOSURE REVIEW"
    if depth_cm >= bands["high_risk_from"]:
        return "HIGH RISK"
    if depth_cm >= bands["caution_from"]:
        return "CAUTION"
    return "MONITOR"
