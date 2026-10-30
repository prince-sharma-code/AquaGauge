"""Data contracts and Pydantic schemas for the AquaGauge Computer Vision module.

All vision outputs use strict schemas. In accordance with project safety guidelines:
1. Object detection detects vehicles/obstacles; it does NOT detect floodwater.
2. 2D image coordinates (pixel or normalized) do NOT represent physical water depth.
"""
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, model_validator


VehicleClass = Literal["car", "bus", "truck", "motorcycle", "bicycle"]
TARGET_VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}

WaterSegmentationMethod = Literal[
    "classical_cv_heuristic",
    "operator_polygon",
    "precomputed_demo",
    "abstain",
]

WaterlineStatus = Literal[
    "READY_FOR_ESTIMATE",
    "ABSTAIN_UNCLEAR_WATERLINE",
    "ABSTAIN_NO_INTERSECTION",
    "OPERATOR_CONFIRMED",
]

VisionPipelineStatus = Literal[
    "SUCCESS",
    "PARTIAL_ABSTAIN",
    "FALLBACK",
    "NO_DETECTION",
]


class BoxPixel(BaseModel):
    """Bounding box coordinates in image pixels."""
    x1: float = Field(description="Left pixel coordinate.")
    y1: float = Field(description="Top pixel coordinate.")
    x2: float = Field(description="Right pixel coordinate.")
    y2: float = Field(description="Bottom pixel coordinate.")

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoxPixel":
        if self.x2 <= self.x1:
            raise ValueError(f"x2 ({self.x2}) must be greater than x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 ({self.y2}) must be greater than y1 ({self.y1})")
        return self


class BoxNormalized(BaseModel):
    """Bounding box coordinates normalized between 0.0 and 1.0."""
    x1: float = Field(ge=0.0, le=1.0, description="Normalized left coordinate.")
    y1: float = Field(ge=0.0, le=1.0, description="Normalized top coordinate.")
    x2: float = Field(ge=0.0, le=1.0, description="Normalized right coordinate.")
    y2: float = Field(ge=0.0, le=1.0, description="Normalized bottom coordinate.")

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoxNormalized":
        if self.x2 <= self.x1:
            raise ValueError(f"Normalized x2 ({self.x2}) must be greater than x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"Normalized y2 ({self.y2}) must be greater than y1 ({self.y1})")
        return self


class VehicleDetection(BaseModel):
    """A single vehicle detected in the frame."""
    id: int = Field(description="Unique detection ID in the frame.")
    class_name: str = Field(description="Detected object class (e.g. car, bus, truck).")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score.")
    box_pixel: BoxPixel
    box_normalized: BoxNormalized


class WaterRegionResult(BaseModel):
    """Estimated or operator-provided water region summary."""
    detected: bool = Field(description="True if a water surface is identified.")
    confidence: float = Field(ge=0.0, le=1.0, default=0.0, description="Confidence in water region delineation.")
    area_percentage: float = Field(ge=0.0, le=100.0, default=0.0, description="Percentage of frame area covered by water.")
    polygon_points: Optional[List[List[Tuple[float, float]]]] = Field(
        default=None,
        description="List of polygons bounding the water surface in pixel coordinates [(x, y), ...].",
    )
    method: WaterSegmentationMethod = Field(
        default="abstain",
        description="Method used to establish the water surface region.",
    )
    note: Optional[str] = Field(
        default=None,
        description="Contextual note or reason for abstention.",
    )


class WaterlineCandidate(BaseModel):
    """Candidate vertical waterline contact points on a detected vehicle."""
    vehicle_id: int = Field(description="ID of the associated VehicleDetection.")
    y_waterline: Optional[float] = Field(
        default=None,
        description="Candidate pixel y-coordinate where waterline intersects the vehicle.",
    )
    y_base: float = Field(description="Pixel y-coordinate at bottom/base of vehicle.")
    y_top: float = Field(description="Pixel y-coordinate at top of vehicle reference.")
    can_estimate: bool = Field(
        description="True if geometry satisfies y_top < y_waterline < y_base for depth estimation.",
    )
    status: WaterlineStatus = Field(description="Geometric status of the candidate waterline.")
    reason: Optional[str] = Field(
        default=None,
        description="Explanation if water level cannot be automatically extracted.",
    )


class VisionPipelineResult(BaseModel):
    """Top-level structured output of the AquaGauge computer vision pipeline."""
    status: VisionPipelineStatus = Field(description="Overall execution status.")
    image_id: Optional[str] = Field(default=None, description="Identifier or filename of the processed image.")
    image_width: int = Field(gt=0, description="Image width in pixels.")
    image_height: int = Field(gt=0, description="Image height in pixels.")
    detections: List[VehicleDetection] = Field(
        default_factory=list,
        description="List of detected vehicles.",
    )
    water_region: WaterRegionResult = Field(
        description="Segmented or operator-confirmed floodwater region.",
    )
    waterline_candidates: List[WaterlineCandidate] = Field(
        default_factory=list,
        description="Candidate waterline contact points intersecting vehicles.",
    )
    annotated_image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded JPEG image with rendered bounding boxes and water overlays.",
    )
    note: str = Field(
        default="Automated visual detections; physical water depth requires reference geometry calibration.",
        description="Safety and operational disclaimer.",
    )


def pixel_to_normalized(box: BoxPixel, width: int, height: int) -> BoxNormalized:
    """Safely convert pixel coordinates to clamped normalized coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Width ({width}) and height ({height}) must be positive integers.")
    x1 = max(0.0, min(1.0, box.x1 / width))
    y1 = max(0.0, min(1.0, box.y1 / height))
    x2 = max(0.0, min(1.0, box.x2 / width))
    y2 = max(0.0, min(1.0, box.y2 / height))
    # Ensure minimum non-zero extent if clamped
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-4)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-4)
    return BoxNormalized(x1=round(x1, 4), y1=round(y1, 4), x2=round(x2, 4), y2=round(y2, 4))


def normalized_to_pixel(box: BoxNormalized, width: int, height: int) -> BoxPixel:
    """Convert normalized coordinates back to image pixel coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Width ({width}) and height ({height}) must be positive integers.")
    return BoxPixel(
        x1=round(box.x1 * width, 1),
        y1=round(box.y1 * height, 1),
        x2=round(box.x2 * width, 1),
        y2=round(box.y2 * height, 1),
    )
