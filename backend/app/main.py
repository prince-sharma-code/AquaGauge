from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from .depth_estimator import estimate_depth_cm
from .policy import classify_vehicle_risk, classify_road_status
from .vision.fixtures import get_demo_fixture
app = FastAPI(
    title="AquaGauge Hackathon API",
    version="0.1.0",
    description=(
        "Prototype API. Estimates use operator-supplied reference geometry; "
        "risk thresholds are illustrative demo defaults, not safety guidance."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/api/incidents")
def get_incidents():
    return {
        "incidents": [
            {
                "incident_id": "DEMO-001",
                "location": {
                    "name": "Sector 12 Demo Zone",
                    "latitude": 28.6139,
                    "longitude": 77.2090,
                },
                "description": "Illustrative sample water-level observation.",
                "risk_level": "High",
                "water_depth_cm": 48,
                "timestamp": "Demo record",
                "source": "Demo sensor",
                "verification_status": "Demo record",
            },
            {
                "incident_id": "DEMO-002",
                "location": {
                    "name": "Riverside Road Demo Zone",
                    "latitude": 28.6250,
                    "longitude": 77.2150,
                },
                "description": "Illustrative sample observation.",
                "risk_level": "Medium",
                "water_depth_cm": 32,
                "timestamp": "Demo record",
                "source": "Demo sensor",
                "verification_status": "Demo record",
            },
        ],
        "data_source": "PRECOMPUTED_DEMO_DATA",
    }
@app.get("/api/weather")
def get_weather():
    return {
        "temperature": 28,
        "rainfall": 12,
        "humidity": 76,
        "wind_speed": 14,
        "data_source": "PRECOMPUTED_DEMO_DATA",
        "note": "Illustrative values only; not live weather observations.",
    }
class CommunityReportRequest(BaseModel):
    report_id: str
    location: dict
    description: str
    severity: str
    source: str
    timestamp: str
    verification_status: str


@app.post("/api/community-reports")
def submit_community_report(report: CommunityReportRequest):
    return {
        "ok": True,
        "report_id": report.report_id,
        "data_source": "DEMO_ACKNOWLEDGEMENT",
        "note": "Report received by the prototype API; server-side persistence is not configured.",
    }
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
@app.post("/api/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    contents = await image.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        with Image.open(BytesIO(contents)) as img:
            img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    fixture = get_demo_fixture(image.filename or "")

    if fixture is not None:
        fixture.status = "FALLBACK"
        fixture.note = (
            "PRECOMPUTED DEMO FIXTURE: this is not live model inference."
        )
        return {
            "analysis_source": "PRECOMPUTED_DEMO_FIXTURE",
            "model_inference": False,
            "result": fixture.model_dump(mode="json"),
        }

    return {
        "status": "ABSTAIN / REVIEW",
        "analysis_source": "LIVE_MODEL_UNAVAILABLE",
        "model_inference": False,
        "detections": [],
        "estimated_depth_cm": None,
        "reason": (
            "Image received and validated, but no matching demo fixture "
            "was found and live YOLO inference is not configured."
        ),
        "note": "No AI analysis result is being claimed for this image.",
    }
