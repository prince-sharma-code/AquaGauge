"""Typed Pydantic models implementing the AquaGauge REST API Contract v1.0-FINAL."""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

ProvenanceSource = Literal["MODEL", "OPERATOR_CONFIRMED", "PRECOMPUTED_DEMO"]
VehicleClass = Literal["two_wheeler", "light_vehicle", "higher_clearance"]
RoadStatus = Literal["MONITOR", "CAUTION", "HIGH RISK", "CLOSURE REVIEW"]
ClosureState = Literal[
    "MONITOR_NORMAL",
    "CAUTION_SLOW",
    "SUGGESTED_DIVERSION",
    "CLOSURE_REVIEW_ADVISORY"
]


class EstimateRequest(BaseModel):
    reference_height_cm: float = Field(
        default=60.0,
        gt=0,
        description="Known physical height/diameter of the selected reference object in cm."
    )
    y_top: float = Field(description="Pixel y-coordinate at top of reference object.")
    y_base: float = Field(description="Pixel y-coordinate at base of reference object.")
    y_waterline: float = Field(description="Pixel y-coordinate where waterline intersects reference object.")
    vehicle_class: VehicleClass = Field(
        default="light_vehicle",
        description="Target vehicle classification."
    )
    detection_id: Optional[str] = Field(
        default=None,
        description="Optional session ID from prior detection. If omitted, input is treated as manual."
    )


class EstimateResponse(BaseModel):
    status: Literal["ESTIMATED"] = "ESTIMATED"
    estimated_depth_cm: float
    submerged_ratio: float
    reference_height_cm: float
    method: str = "reference_height_ratio_demo"
    vehicle_class: VehicleClass
    vehicle_risk: str
    road_status: str
    note: str
    policy_notice: str = "Illustrative demo thresholds only; not real-world safety advice."
    source: ProvenanceSource
    estimate_id: str


class EstimateAbstainResponse(BaseModel):
    status: Literal["ABSTAIN / REVIEW"] = "ABSTAIN / REVIEW"
    reason: str
    estimated_depth_cm: Optional[float] = None
    note: str = "Insufficient reference geometry; no numeric estimate emitted."
    source: ProvenanceSource
    estimate_id: str


# Corridor Models
class ReferenceObjectOption(BaseModel):
    id: str
    name: str
    reference_height_cm: float
    is_default: bool = False


class CorridorLocation(BaseModel):
    longitude: float
    latitude: float
    city: str
    country: str


class CorridorCalibration(BaseModel):
    status: str = "DEMO_REFERENCE_CONFIGURATION"
    verification_notice: str = "Illustrative demo reference values. Not physically surveyed or site-calibrated."
    reference_objects: List[ReferenceObjectOption]


class CorridorResponse(BaseModel):
    corridor_id: str
    camera_id: str
    corridor_name: str
    description: str
    location: CorridorLocation
    calibration: CorridorCalibration
    default_image_id: str


# Route Models
class RouteSegment(BaseModel):
    segment_id: str
    name: str
    color_hex: str
    coordinate_format: str = "[longitude, latitude]"
    coordinates: List[List[float]]


class DetourRoute(BaseModel):
    route_id: str
    name: str
    color_hex: str
    demo_distance_km: float
    demo_delay_minutes: float
    coordinate_format: str = "[longitude, latitude]"
    coordinates: List[List[float]]


class RouteResponse(BaseModel):
    corridor_id: str
    corridor_name: str
    road_status: RoadStatus
    closure_state: ClosureState
    is_diversion_recommended: bool
    advisory_notice: str = "Demo detour schematic only. Does not represent official municipal road closure or live traffic routing."
    blocked_segment: RouteSegment
    detour_route: DetourRoute


# Incident Models
class IncidentCreateRequest(BaseModel):
    estimate_id: str = Field(description="Authoritative backend-verified estimate identifier.")
    camera_id: str = Field(description="Monitored camera identifier.")
    corridor_name: str = Field(description="Monitored road/corridor name.")
    closure_state: ClosureState = Field(
        default="SUGGESTED_DIVERSION",
        description="Advisory route diversion state."
    )
    operator_notes: Optional[str] = Field(default="", description="Operator observations.")
    image_id: Optional[str] = Field(default=None, description="Image reference key.")


class IncidentResponse(BaseModel):
    incident_id: str
    estimate_id: str
    timestamp: str
    camera_id: str
    corridor_name: str
    estimated_depth_cm: Optional[float]
    road_status: str
    vehicle_risk: str
    vehicle_class: str
    source: ProvenanceSource
    closure_state: str
    operator_notes: Optional[str] = ""
    image_id: Optional[str] = None


# Detection Models
class BoundingBoxPixels(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class SuggestedReference(BaseModel):
    type: str = "wheel_bounding_box"
    y_top: float
    y_base: float
    reference_height_cm: float = 60.0


class DetectionItem(BaseModel):
    detection_id: str
    class_name: str
    mapped_vehicle_class: VehicleClass
    confidence: float
    source: ProvenanceSource
    box_pixels: BoundingBoxPixels
    suggested_reference: SuggestedReference


class WaterBoundary(BaseModel):
    status: Literal["PROPOSED", "UNAVAILABLE", "PRECOMPUTED"] = "PROPOSED"
    source: ProvenanceSource
    candidate_waterline_y: Optional[float] = None
    water_polygon_pixels: Optional[List[List[float]]] = None
    operator_review_required: bool = True


class DetectResponse(BaseModel):
    detection_id: str
    image_id: str
    image_dimensions: dict
    source: ProvenanceSource
    model_info: dict
    detections: List[DetectionItem]
    water_boundary: WaterBoundary
    policy_notice: str = "AI proposals are candidate estimates. Operator confirmation required."


class DetectJSONRequest(BaseModel):
    image_id: str = Field(description="Preloaded demo asset key from assets/demo_images/.")


# ==========================================
# MULTI-SOURCE FLOOD REPORT MODELS
# ==========================================

ReportType = Literal[
    "IMAGE",
    "MANUAL",
    "COMMUNITY",
    "RAINFALL",
    "WATER_LEVEL",
    "SATELLITE",
    "SMS",
    "OFFLINE"
]

ReportVerificationStatus = Literal[
    "UNVERIFIED",
    "VERIFIED",
    "REJECTED"
]


class FloodReportCreate(BaseModel):
    report_type: ReportType

    location: str = Field(min_length=2, max_length=250)

    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)

    description: str = Field(min_length=3, max_length=2000)

    reported_severity: Optional[
        Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    ] = "UNKNOWN"

    image_id: Optional[str] = None

    water_level_cm: Optional[float] = Field(default=None, ge=0)

    rainfall_mm: Optional[float] = Field(default=None, ge=0)

    observed_at: Optional[str] = None

    reporter_notes: Optional[str] = Field(default=None, max_length=1000)


class FloodReportResponse(BaseModel):
    report_id: str
    report_type: ReportType
    location: str

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    description: str
    reported_severity: str
    verification_status: ReportVerificationStatus

    image_id: Optional[str] = None
    water_level_cm: Optional[float] = None
    rainfall_mm: Optional[float] = None

    observed_at: Optional[str] = None
    reporter_notes: Optional[str] = None

    submitted_at: str

class FloodReportVerificationRequest(BaseModel):
    verification_status: Literal["VERIFIED", "REJECTED"]

