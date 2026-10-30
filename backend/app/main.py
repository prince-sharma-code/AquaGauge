"""FastAPI application for AquaGauge Hackathon Prototype.

Adheres strictly to the AquaGauge REST API Contract v1.0-FINAL:
- Backwards-compatible estimate math and risk policy defaults.
- Backend-verified processing history (provenance) tracking.
- Transactional SQLite duplicate prevention.
- Advisory corridor detour routing with GeoJSON [longitude, latitude].
"""
from typing import List, Optional, Union
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from .depth_estimator import estimate_depth_cm
from .policy import classify_vehicle_risk, classify_road_status
from .session_store import session_store
from .models import (
    EstimateRequest,
    EstimateResponse,
    EstimateAbstainResponse,
    CorridorResponse,
    RouteResponse,
    RoadStatus,
    IncidentCreateRequest,
    IncidentResponse,
    DetectJSONRequest,
    DetectResponse,
    DetectionItem,
    BoundingBoxPixels,
    SuggestedReference,
    WaterBoundary,
    FloodReportCreate,
    FloodReportResponse,
)

from .database import (
    init_db,
    create_incident_atomic,
    get_recent_incidents,
    DuplicateIncidentError,
    AbstainedIncidentError,
    create_flood_report,
    get_flood_reports,
)

from .route_service import get_corridor_metadata, get_route_detour

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database before serving requests
    init_db()
    yield

app = FastAPI(
    title="AquaGauge Hackathon API",
    version="0.1.0",
    description=(
        "AquaGauge prototype API. Estimates use operator-supplied or candidate reference geometry; "
        "risk thresholds and detour routes are illustrative demo defaults, not safety guidance."
    ),
    lifespan=lifespan,
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "aquagauge-hackathon-api",
        "mode": "prototype",
        "version": "0.1.0",
    }


@app.post(
    "/api/estimate",
    response_model=Union[EstimateResponse, EstimateAbstainResponse],
    summary="Estimate depth and classify risk with backend-verified provenance",
)
def estimate(payload: EstimateRequest):
    # 1. Evaluate provenance
    try:
        source = session_store.evaluate_provenance(
            detection_id=payload.detection_id,
            y_top=payload.y_top,
            y_base=payload.y_base,
            y_waterline=payload.y_waterline,
            reference_height_cm=payload.reference_height_cm,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    # 2. Attempt calculation
    try:
        depth = estimate_depth_cm(
            reference_height_cm=payload.reference_height_cm,
            y_top=payload.y_top,
            y_base=payload.y_base,
            y_waterline=payload.y_waterline,
        )
    except ValueError as exc:
        # Register abstained estimate session
        abstain_estimate_id = session_store.create_estimate_session(
            status="ABSTAIN / REVIEW",
            estimated_depth_cm=None,
            submerged_ratio=None,
            reference_height_cm=payload.reference_height_cm,
            vehicle_class=payload.vehicle_class,
            vehicle_risk="MONITOR",
            road_status="CLOSURE REVIEW",
            source=source,
        )
        return EstimateAbstainResponse(
            status="ABSTAIN / REVIEW",
            reason=str(exc),
            estimated_depth_cm=None,
            note="Insufficient reference geometry; no numeric estimate emitted.",
            source=source,
            estimate_id=abstain_estimate_id,
        )

    # 3. Classify risk
    vehicle_risk = classify_vehicle_risk(depth.estimated_depth_cm, payload.vehicle_class)
    road_status = classify_road_status(depth.estimated_depth_cm)

    # 4. Register valid estimate session
    estimate_id = session_store.create_estimate_session(
        status="ESTIMATED",
        estimated_depth_cm=depth.estimated_depth_cm,
        submerged_ratio=depth.submerged_ratio,
        reference_height_cm=depth.reference_height_cm,
        vehicle_class=payload.vehicle_class,
        vehicle_risk=vehicle_risk,
        road_status=road_status,
        source=source,
    )

    return EstimateResponse(
        status="ESTIMATED",
        estimated_depth_cm=depth.estimated_depth_cm,
        submerged_ratio=depth.submerged_ratio,
        reference_height_cm=depth.reference_height_cm,
        method=depth.method,
        vehicle_class=payload.vehicle_class,
        vehicle_risk=vehicle_risk,
        road_status=road_status,
        note=depth.note,
        policy_notice="Illustrative demo thresholds only; not real-world safety advice.",
        source=source,
        estimate_id=estimate_id,
    )


@app.get("/api/corridor", response_model=CorridorResponse)
def get_corridor(corridor_id: str = Query(default="CORRIDOR_MG_ROAD")):
    if corridor_id != "CORRIDOR_MG_ROAD":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corridor '{corridor_id}' not found.",
        )
    return get_corridor_metadata(corridor_id)


@app.get("/api/route", response_model=RouteResponse)
def get_route(road_status: RoadStatus = Query(default="HIGH RISK")):
    return get_route_detour(road_status)


@app.get("/api/incidents", response_model=List[IncidentResponse])
def list_incidents(limit: int = Query(default=50, ge=1, le=100)):
    return get_recent_incidents(limit=limit)


@app.post(
    "/api/incidents",
    response_model=IncidentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record verified flood incident into SQLite",
)
def create_incident(payload: IncidentCreateRequest):
    # Retrieve verified estimate from backend session store
    estimate_session = session_store.get_estimate_session(payload.estimate_id)
    if not estimate_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Estimate session '{payload.estimate_id}' not found or expired.",
        )

    try:
        return create_incident_atomic(
            request=payload,
            estimate=estimate_session,
            session_store=session_store,
        )
    except AbstainedIncidentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    except DuplicateIncidentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
@app.post(
    "/api/reports",
    response_model=FloodReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a multi-source flood report",
)
def submit_flood_report(payload: FloodReportCreate):
    return create_flood_report(request=payload)


@app.get(
    "/api/reports",
    response_model=List[FloodReportResponse],
    summary="Retrieve recent multi-source flood reports",
)
def list_flood_reports(
    limit: int = Query(default=50, ge=1, le=100)
):
    return get_flood_reports(limit=limit)

# Allowlist for starter-kit bundled demo images
_DEMO_PRESETS = {
    "flooded_vehicle_underpass.jpg": {
        "dimensions": {"width": 640, "height": 480},
        "item_id": "det_car_01",
        "class_name": "car",
        "mapped_vehicle_class": "light_vehicle",
        "confidence": 0.88,
        "box": {"x1": 210.0, "y1": 178.0, "x2": 452.0, "y2": 338.0},
        "ref": {"y_top": 280.0, "y_base": 338.0, "reference_height_cm": 60.0},
        "waterline_y": 308.0,
        "polygon": [
            [80.0, 310.0], [240.0, 308.0], [480.0, 309.0],
            [640.0, 312.0], [640.0, 480.0], [0.0, 480.0]
        ],
    },
    "underpass_overview.jpg": {
        "dimensions": {"width": 1280, "height": 720},
        "item_id": "det_car_02",
        "class_name": "car",
        "mapped_vehicle_class": "light_vehicle",
        "confidence": 0.82,
        "box": {"x1": 420.0, "y1": 340.0, "x2": 710.0, "y2": 520.0},
        "ref": {"y_top": 450.0, "y_base": 520.0, "reference_height_cm": 60.0},
        "waterline_y": 485.0,
        "polygon": [
            [200.0, 480.0], [600.0, 485.0], [1000.0, 485.0],
            [1280.0, 500.0], [1280.0, 720.0], [0.0, 720.0]
        ],
    },
    "waterlogged_road_vehicle.jpg": {
        "dimensions": {"width": 800, "height": 600},
        "item_id": "det_car_03",
        "class_name": "car",
        "mapped_vehicle_class": "light_vehicle",
        "confidence": 0.79,
        "box": {"x1": 250.0, "y1": 220.0, "x2": 540.0, "y2": 420.0},
        "ref": {"y_top": 360.0, "y_base": 420.0, "reference_height_cm": 60.0},
        "waterline_y": 395.0,
        "polygon": [
            [100.0, 390.0], [400.0, 395.0], [700.0, 395.0],
            [800.0, 410.0], [800.0, 600.0], [0.0, 600.0]
        ],
    },
}


@app.post(
    "/api/detect",
    response_model=DetectResponse,
    summary="Detect vehicles and propose candidate waterline",
)
async def detect(request: Request):
    content_type = request.headers.get("content-type", "")

    # Option A: Uploaded file via multipart/form-data
    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Form field 'file' must contain an uploaded image.",
            )
        if getattr(uploaded_file, "content_type", "") not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File type not supported. Use JPEG, PNG, or WebP.",
            )
        # Person 2's vision module is not yet integrated on this branch.
        # Do not fabricate detections for arbitrary uploads.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "Live vision model inference for arbitrary image uploads is not yet implemented on this branch. "
                "Use bundled demo assets with Option B (JSON payload with image_id) for deterministic demonstration."
            ),
        )

    # Option B: Preloaded demo image via application/json
    image_id: Optional[str] = None
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                image_id = body.get("image_id")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON payload.",
            )

    if not image_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either a file upload via multipart/form-data or a JSON body with 'image_id' must be provided.",
        )

    preset = _DEMO_PRESETS.get(image_id)
    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preloaded demo image '{image_id}' not found in starter-kit library.",
        )

    # Register detection session with PRECOMPUTED_DEMO provenance
    detection_id = session_store.create_detection_session(
        image_id=image_id,
        detection_item_id=preset["item_id"],
        source="PRECOMPUTED_DEMO",
        y_top=preset["ref"]["y_top"],
        y_base=preset["ref"]["y_base"],
        y_waterline=preset["waterline_y"],
        reference_height_cm=preset["ref"]["reference_height_cm"],
    )

    return DetectResponse(
        detection_id=detection_id,
        image_id=image_id,
        image_dimensions=preset["dimensions"],
        source="PRECOMPUTED_DEMO",
        model_info={
            "vehicle_detector": "precomputed_demo_fixture",
            "water_segmentation": "candidate_heuristic_proposal",
            "fallback_active": True,
        },
        detections=[
            DetectionItem(
                detection_id=preset["item_id"],
                class_name=preset["class_name"],
                mapped_vehicle_class=preset["mapped_vehicle_class"],
                confidence=preset["confidence"],
                source="PRECOMPUTED_DEMO",
                box_pixels=BoundingBoxPixels(**preset["box"]),
                suggested_reference=SuggestedReference(**preset["ref"]),
            )
        ],
        water_boundary=WaterBoundary(
            status="PROPOSED",
            source="PRECOMPUTED_DEMO",
            candidate_waterline_y=preset["waterline_y"],
            water_polygon_pixels=preset["polygon"],
            operator_review_required=True,
        ),
        policy_notice="AI proposals are candidate estimates. Operator confirmation required.",
    )
