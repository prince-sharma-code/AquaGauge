from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .depth_estimator import estimate_depth_cm
from .policy import classify_vehicle_risk, classify_road_status

app = FastAPI(
    title="AquaGauge Hackathon API",
    version="0.1.0",
    description=(
        "Prototype API. Estimates use operator-supplied reference geometry; "
        "risk thresholds are illustrative demo defaults, not safety guidance."
    ),
)


class EstimateRequest(BaseModel):
    reference_height_cm: float = Field(default=60.0, gt=0, description="Known physical height/diameter of the selected reference object.")
    y_top: float = Field(description="Pixel y-coordinate at top of reference object.")
    y_base: float = Field(description="Pixel y-coordinate at base of reference object.")
    y_waterline: float = Field(description="Pixel y-coordinate where waterline intersects reference object.")
    vehicle_class: Literal["two_wheeler", "light_vehicle", "higher_clearance"] = "light_vehicle"


@app.get("/api/health")
def health():
    return {"ok": True, "service": "aquagauge-hackathon-api", "mode": "prototype"}


@app.post("/api/estimate")
def estimate(payload: EstimateRequest):
    try:
        depth = estimate_depth_cm(
            reference_height_cm=payload.reference_height_cm,
            y_top=payload.y_top,
            y_base=payload.y_base,
            y_waterline=payload.y_waterline,
        )
    except ValueError as exc:
        return {
            "status": "ABSTAIN / REVIEW",
            "reason": str(exc),
            "estimated_depth_cm": None,
            "note": "Insufficient reference geometry; no numeric estimate emitted.",
        }
    return {
        "status": "ESTIMATED",
        "estimated_depth_cm": depth.estimated_depth_cm,
        "submerged_ratio": depth.submerged_ratio,
        "reference_height_cm": depth.reference_height_cm,
        "method": depth.method,
        "vehicle_class": payload.vehicle_class,
        "vehicle_risk": classify_vehicle_risk(depth.estimated_depth_cm, payload.vehicle_class),
        "road_status": classify_road_status(depth.estimated_depth_cm),
        "note": depth.note,
        "policy_notice": "Illustrative demo thresholds only; not real-world safety advice.",
    }
