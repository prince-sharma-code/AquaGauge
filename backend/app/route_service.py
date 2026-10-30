"""Corridor calibration and advisory route detour data service."""
from .models import (
    CorridorResponse,
    CorridorLocation,
    CorridorCalibration,
    ReferenceObjectOption,
    RouteResponse,
    RouteSegment,
    DetourRoute,
    RoadStatus,
    ClosureState,
)


def get_corridor_metadata(corridor_id: str = "CORRIDOR_MG_ROAD") -> CorridorResponse:
    """Return site configuration and reference geometry for the prototype corridor."""
    return CorridorResponse(
        corridor_id="CORRIDOR_MG_ROAD",
        camera_id="CAM_UNDERPASS_04",
        corridor_name="MG Road Underpass - Northbound",
        description="Monitored sub-surface roadway section for hackathon demonstration.",
        location=CorridorLocation(
            longitude=77.5946,
            latitude=12.9716,
            city="Bengaluru",
            country="India",
        ),
        calibration=CorridorCalibration(
            status="DEMO_REFERENCE_CONFIGURATION",
            verification_notice="Illustrative demo reference values. Not physically surveyed or site-calibrated.",
            reference_objects=[
                ReferenceObjectOption(
                    id="ref_std_wheel",
                    name="Standard Passenger Car Wheel / Tire (Illustrative)",
                    reference_height_cm=60.0,
                    is_default=True,
                ),
                ReferenceObjectOption(
                    id="ref_curb_marker",
                    name="Roadside Drainage Curb (Illustrative)",
                    reference_height_cm=25.0,
                    is_default=False,
                ),
            ],
        ),
        default_image_id="flooded_vehicle_underpass.jpg",
    )


def get_route_detour(road_status: RoadStatus = "HIGH RISK") -> RouteResponse:
    """Return GeoJSON [longitude, latitude] polylines and advisory diversion states."""
    # Map road_status to advisory closure_state
    if road_status == "MONITOR":
        closure_state: ClosureState = "MONITOR_NORMAL"
        is_diversion_recommended = False
    elif road_status == "CAUTION":
        closure_state: ClosureState = "CAUTION_SLOW"
        is_diversion_recommended = False
    elif road_status == "HIGH RISK":
        closure_state: ClosureState = "SUGGESTED_DIVERSION"
        is_diversion_recommended = True
    else:  # "CLOSURE REVIEW"
        closure_state: ClosureState = "CLOSURE_REVIEW_ADVISORY"
        is_diversion_recommended = True

    # Coordinates strictly in [longitude, latitude] format
    blocked_coords = [
        [77.5940, 12.9712],
        [77.5946, 12.9716],
        [77.5952, 12.9720],
    ]

    detour_coords = [
        [77.5940, 12.9712],
        [77.5932, 12.9725],
        [77.5942, 12.9738],
        [77.5958, 12.9729],
        [77.5952, 12.9720],
    ]

    return RouteResponse(
        corridor_id="CORRIDOR_MG_ROAD",
        corridor_name="MG Road Underpass",
        road_status=road_status,
        closure_state=closure_state,
        is_diversion_recommended=is_diversion_recommended,
        advisory_notice="Demo detour schematic only. Does not represent official municipal road closure or live traffic routing.",
        blocked_segment=RouteSegment(
            segment_id="SEG_MG_01",
            name="MG Road Underpass Trench (Illustrative)",
            color_hex="#EF4444",
            coordinate_format="[longitude, latitude]",
            coordinates=blocked_coords,
        ),
        detour_route=DetourRoute(
            route_id="ROUTE_DETOUR_01",
            name="Surface Link Road & Flyover Bypass (Illustrative)",
            color_hex="#10B981",
            demo_distance_km=1.2,
            demo_delay_minutes=4.5,
            coordinate_format="[longitude, latitude]",
            coordinates=detour_coords,
        ),
    )
