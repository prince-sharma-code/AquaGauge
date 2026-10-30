"""Deterministic offline fixtures for the AquaGauge demo images.

These fixtures serve as a rock-solid offline fallback during hackathon presentations,
network outages, or CPU inference testing, guaranteeing deterministic results without
requiring live model weight loading or GPU execution.
"""
from typing import Dict, Optional
from pathlib import Path
from .schemas import (
    BoxPixel,
    BoxNormalized,
    VehicleDetection,
    WaterRegionResult,
    WaterlineCandidate,
    VisionPipelineResult,
    pixel_to_normalized,
)


def _build_detection(
    det_id: int,
    class_name: str,
    confidence: float,
    xyxy: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> VehicleDetection:
    box_px = BoxPixel(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
    box_norm = pixel_to_normalized(box_px, img_w, img_h)
    return VehicleDetection(
        id=det_id,
        class_name=class_name,
        confidence=confidence,
        box_pixel=box_px,
        box_normalized=box_norm,
    )


# 1. flooded_vehicle_underpass.jpg (1598 x 770)
_FIXTURE_UNDERPASS_VEHICLE = VisionPipelineResult(
    status="SUCCESS",
    image_id="flooded_vehicle_underpass.jpg",
    image_width=1598,
    image_height=770,
    detections=[
        _build_detection(
            det_id=1,
            class_name="car",
            confidence=0.921,
            xyxy=(528.2, 320.2, 1115.6, 560.1),
            img_w=1598,
            img_h=770,
        ),
    ],
    water_region=WaterRegionResult(
        detected=True,
        confidence=0.88,
        area_percentage=42.5,
        polygon_points=[
            [(0.0, 480.0), (1598.0, 470.0), (1598.0, 770.0), (0.0, 770.0)]
        ],
        method="precomputed_demo",
        note="Submerged underpass road surface with clearly defined standing water.",
    ),
    waterline_candidates=[
        WaterlineCandidate(
            vehicle_id=1,
            y_waterline=470.0,
            y_base=560.1,
            y_top=320.2,
            can_estimate=True,
            status="READY_FOR_ESTIMATE",
            reason="Waterline visibly intersects the lower chassis and wheels of the vehicle.",
        ),
    ],
    note="Precomputed offline demo fixture; verified against YOLO11n on CPU.",
)


# 2. waterlogged_road_vehicle.jpg (898 x 684)
_FIXTURE_WATERLOGGED_ROAD = VisionPipelineResult(
    status="SUCCESS",
    image_id="waterlogged_road_vehicle.jpg",
    image_width=898,
    image_height=684,
    detections=[
        _build_detection(
            det_id=1,
            class_name="car",
            confidence=0.919,
            xyxy=(477.8, 56.5, 854.1, 293.5),
            img_w=898,
            img_h=684,
        ),
    ],
    water_region=WaterRegionResult(
        detected=True,
        confidence=0.85,
        area_percentage=55.0,
        polygon_points=[
            [(0.0, 210.0), (898.0, 200.0), (898.0, 684.0), (0.0, 684.0)]
        ],
        method="precomputed_demo",
        note="Waterlogged road section covering primary driving lanes.",
    ),
    waterline_candidates=[
        WaterlineCandidate(
            vehicle_id=1,
            y_waterline=235.0,
            y_base=293.5,
            y_top=56.5,
            can_estimate=True,
            status="READY_FOR_ESTIMATE",
            reason="Waterline intersects the wheel hubs of the vehicle.",
        ),
    ],
    note="Precomputed offline demo fixture; verified against YOLO11n on CPU.",
)


# 3. underpass_overview.jpg (2880 x 1620)
_FIXTURE_UNDERPASS_OVERVIEW = VisionPipelineResult(
    status="SUCCESS",
    image_id="underpass_overview.jpg",
    image_width=2880,
    image_height=1620,
    detections=[
        _build_detection(
            det_id=1,
            class_name="car",
            confidence=0.813,
            xyxy=(1900.1, 1333.2, 2874.7, 1620.0),
            img_w=2880,
            img_h=1620,
        ),
        _build_detection(
            det_id=2,
            class_name="car",
            confidence=0.769,
            xyxy=(1811.3, 1090.3, 2302.1, 1467.5),
            img_w=2880,
            img_h=1620,
        ),
        _build_detection(
            det_id=3,
            class_name="motorcycle",
            confidence=0.413,
            xyxy=(1659.1, 1242.6, 1829.1, 1477.6),
            img_w=2880,
            img_h=1620,
        ),
        _build_detection(
            det_id=4,
            class_name="motorcycle",
            confidence=0.395,
            xyxy=(1663.2, 1284.9, 1821.9, 1452.2),
            img_w=2880,
            img_h=1620,
        ),
    ],
    water_region=WaterRegionResult(
        detected=True,
        confidence=0.80,
        area_percentage=38.0,
        polygon_points=[
            [(1200.0, 1200.0), (2880.0, 1300.0), (2880.0, 1620.0), (1200.0, 1620.0)]
        ],
        method="precomputed_demo",
        note="Wide-angle view of flooded underpass with multiple vehicles and responders.",
    ),
    waterline_candidates=[
        WaterlineCandidate(
            vehicle_id=1,
            y_waterline=1480.0,
            y_base=1620.0,
            y_top=1333.2,
            can_estimate=True,
            status="READY_FOR_ESTIMATE",
            reason="Foreground vehicle partially submerged at exit ramp.",
        ),
        WaterlineCandidate(
            vehicle_id=2,
            y_waterline=1320.0,
            y_base=1467.5,
            y_top=1090.3,
            can_estimate=True,
            status="READY_FOR_ESTIMATE",
            reason="Secondary vehicle in mid-water pool.",
        ),
    ],
    note="Precomputed offline demo fixture; verified against YOLO11n on CPU.",
)


DEMO_FIXTURES: Dict[str, VisionPipelineResult] = {
    "flooded_vehicle_underpass.jpg": _FIXTURE_UNDERPASS_VEHICLE,
    "waterlogged_road_vehicle.jpg": _FIXTURE_WATERLOGGED_ROAD,
    "underpass_overview.jpg": _FIXTURE_UNDERPASS_OVERVIEW,
}


def get_demo_fixture(image_name_or_path: str) -> Optional[VisionPipelineResult]:
    """Look up a precomputed demo fixture by filename or filepath.

    Returns a deep copy of VisionPipelineResult if found, otherwise None.
    """
    filename = Path(image_name_or_path).name.lower()
    for key, fixture in DEMO_FIXTURES.items():
        if key.lower() == filename or key.lower() in filename:
            # Return model copy to prevent mutation of the template
            return fixture.model_copy(deep=True)
    return None
