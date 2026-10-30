"""Comprehensive test suite for AquaGauge REST API endpoints and integration logic.

Tests:
- /api/health check
- /api/estimate manual entry, candidate matching, coordinate modification, abstention, and 404 for unknown session
- /api/corridor metadata and GeoJSON coordinate ordering
- /api/route GeoJSON polylines and advisory diversion states
- /api/detect bundled demo images (PRECOMPUTED_DEMO) and 501 for arbitrary file upload
- /api/incidents creation, retrieval, abstention rejection (409), and duplicate prevention (409)
"""
import os
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.session_store import session_store
from backend.app.database import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure in-memory sessions are cleared between tests."""
    session_store.clear()
    init_db()


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["service"] == "aquagauge-hackathon-api"
    assert data["mode"] == "prototype"


def test_estimate_manual_provenance():
    """When detection_id is omitted, source is assigned as OPERATOR_CONFIRMED."""
    payload = {
        "reference_height_cm": 60.0,
        "y_top": 100.0,
        "y_base": 200.0,
        "y_waterline": 150.0,
        "vehicle_class": "light_vehicle",
    }
    res = client.post("/api/estimate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ESTIMATED"
    assert data["estimated_depth_cm"] == 30.0
    assert data["submerged_ratio"] == 0.5
    assert data["source"] == "OPERATOR_CONFIRMED"
    assert "estimate_id" in data
    assert data["estimate_id"].startswith("est_")


def test_estimate_matching_candidate_preserves_provenance():
    """When coordinates match candidate detection within tolerance, preserve original source."""
    # 1. Create a detection session
    detect_res = client.post(
        "/api/detect",
        json={"image_id": "flooded_vehicle_underpass.jpg"},
    )
    assert detect_res.status_code == 200
    det_data = detect_res.json()
    detection_id = det_data["detection_id"]
    assert det_data["source"] == "PRECOMPUTED_DEMO"

    cand_ref = det_data["detections"][0]["suggested_reference"]
    cand_water = det_data["water_boundary"]["candidate_waterline_y"]

    # 2. Submit exact coordinates with detection_id
    payload = {
        "reference_height_cm": cand_ref["reference_height_cm"],
        "y_top": cand_ref["y_top"],
        "y_base": cand_ref["y_base"],
        "y_waterline": cand_water,
        "vehicle_class": "light_vehicle",
        "detection_id": detection_id,
    }
    res = client.post("/api/estimate", json=payload)
    assert res.status_code == 200
    est_data = res.json()
    assert est_data["status"] == "ESTIMATED"
    # Provenance preserved as PRECOMPUTED_DEMO
    assert est_data["source"] == "PRECOMPUTED_DEMO"


def test_estimate_altered_coordinates_assigns_operator_confirmed():
    """When coordinates are altered beyond 0.5px tolerance, source becomes OPERATOR_CONFIRMED."""
    detect_res = client.post(
        "/api/detect",
        json={"image_id": "flooded_vehicle_underpass.jpg"},
    )
    det_data = detect_res.json()
    detection_id = det_data["detection_id"]
    cand_ref = det_data["detections"][0]["suggested_reference"]
    cand_water = det_data["water_boundary"]["candidate_waterline_y"]

    # Alter waterline by 5 pixels
    payload = {
        "reference_height_cm": cand_ref["reference_height_cm"],
        "y_top": cand_ref["y_top"],
        "y_base": cand_ref["y_base"],
        "y_waterline": cand_water - 5.0,
        "vehicle_class": "light_vehicle",
        "detection_id": detection_id,
    }
    res = client.post("/api/estimate", json=payload)
    assert res.status_code == 200
    est_data = res.json()
    assert est_data["status"] == "ESTIMATED"
    assert est_data["source"] == "OPERATOR_CONFIRMED"


def test_estimate_unknown_detection_id_returns_404():
    payload = {
        "reference_height_cm": 60.0,
        "y_top": 100.0,
        "y_base": 200.0,
        "y_waterline": 150.0,
        "vehicle_class": "light_vehicle",
        "detection_id": "det_sess_nonexistent",
    }
    res = client.post("/api/estimate", json=payload)
    assert res.status_code == 404
    assert "not found or expired" in res.json()["detail"]


def test_estimate_geometry_abstains():
    """Invalid waterline outside bounds returns ABSTAIN / REVIEW with null depth and unique estimate_id."""
    payload = {
        "reference_height_cm": 60.0,
        "y_top": 100.0,
        "y_base": 200.0,
        "y_waterline": 250.0,  # below base
        "vehicle_class": "light_vehicle",
    }
    res = client.post("/api/estimate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ABSTAIN / REVIEW"
    assert data["estimated_depth_cm"] is None
    assert "Insufficient reference geometry" in data["note"]
    assert "estimate_id" in data


def test_corridor_endpoint():
    res = client.get("/api/corridor")
    assert res.status_code == 200
    data = res.json()
    assert data["corridor_id"] == "CORRIDOR_MG_ROAD"
    assert data["calibration"]["status"] == "DEMO_REFERENCE_CONFIGURATION"
    # Verify longitude comes first, latitude second
    loc = data["location"]
    assert loc["longitude"] == 77.5946
    assert loc["latitude"] == 12.9716


def test_route_endpoint_advisory_states():
    # Test Monitor
    res_mon = client.get("/api/route?road_status=MONITOR")
    assert res_mon.status_code == 200
    assert res_mon.json()["closure_state"] == "MONITOR_NORMAL"
    assert res_mon.json()["is_diversion_recommended"] is False

    # Test High Risk
    res_hr = client.get("/api/route?road_status=HIGH RISK")
    assert res_hr.status_code == 200
    hr_data = res_hr.json()
    assert hr_data["closure_state"] == "SUGGESTED_DIVERSION"
    assert hr_data["is_diversion_recommended"] is True
    # Verify coordinates are [longitude, latitude]
    coords = hr_data["blocked_segment"]["coordinates"]
    assert coords[0][0] == 77.5940  # Longitude
    assert coords[0][1] == 12.9712  # Latitude


def test_detect_endpoint_bundled_preset():
    res = client.post("/api/detect", json={"image_id": "flooded_vehicle_underpass.jpg"})
    assert res.status_code == 200
    data = res.json()
    assert data["image_id"] == "flooded_vehicle_underpass.jpg"
    assert data["source"] == "PRECOMPUTED_DEMO"
    assert len(data["detections"]) > 0
    assert data["water_boundary"]["status"] == "PROPOSED"


def test_detect_endpoint_unsupported_image_returns_404():
    res = client.post("/api/detect", json={"image_id": "arbitrary_unseen.jpg"})
    assert res.status_code == 404


def test_detect_endpoint_upload_returns_501():
    files = {"file": ("test.png", b"fakebytes", "image/png")}
    res = client.post("/api/detect", files=files)
    assert res.status_code == 501
    assert "Live vision model inference" in res.json()["detail"]


def test_incident_creation_and_retrieval_flow():
    # 1. Produce valid estimate
    est_res = client.post(
        "/api/estimate",
        json={
            "reference_height_cm": 60.0,
            "y_top": 100.0,
            "y_base": 200.0,
            "y_waterline": 147.0,
            "vehicle_class": "light_vehicle",
        },
    )
    est_data = est_res.json()
    estimate_id = est_data["estimate_id"]

    # 2. Record incident
    inc_payload = {
        "estimate_id": estimate_id,
        "camera_id": "CAM_UNDERPASS_04",
        "corridor_name": "MG Road Underpass - Northbound",
        "closure_state": "SUGGESTED_DIVERSION",
        "operator_notes": "Operator verified depth using wheel hub.",
        "image_id": "flooded_vehicle_underpass.jpg",
    }
    inc_res = client.post("/api/incidents", json=inc_payload)
    assert inc_res.status_code == 201
    inc_data = inc_res.json()
    assert inc_data["incident_id"].startswith("INC-")
    assert inc_data["estimate_id"] == estimate_id
    assert inc_data["estimated_depth_cm"] == est_data["estimated_depth_cm"]
    assert inc_data["road_status"] == est_data["road_status"]
    assert inc_data["source"] == est_data["source"]

    # 3. Retrieve list of incidents
    list_res = client.get("/api/incidents")
    assert list_res.status_code == 200
    incidents = list_res.json()
    assert len(incidents) >= 1
    assert incidents[0]["incident_id"] == inc_data["incident_id"]


def test_incident_creation_rejects_abstained_estimate():
    """An abstained estimate cannot create a depth incident and returns HTTP 409."""
    est_res = client.post(
        "/api/estimate",
        json={
            "reference_height_cm": 60.0,
            "y_top": 100.0,
            "y_base": 200.0,
            "y_waterline": 250.0,  # Invalid -> abstains
            "vehicle_class": "light_vehicle",
        },
    )
    est_data = est_res.json()
    estimate_id = est_data["estimate_id"]
    assert est_data["status"] == "ABSTAIN / REVIEW"

    inc_payload = {
        "estimate_id": estimate_id,
        "camera_id": "CAM_UNDERPASS_04",
        "corridor_name": "MG Road Underpass",
        "closure_state": "SUGGESTED_DIVERSION",
    }
    res = client.post("/api/incidents", json=inc_payload)
    assert res.status_code == 409
    assert "Cannot create a depth-based incident from an abstained estimate" in res.json()["detail"]


def test_incident_creation_duplicate_prevention_http_409():
    """Submitting the same estimate_id twice is rejected with HTTP 409."""
    est_res = client.post(
        "/api/estimate",
        json={
            "reference_height_cm": 60.0,
            "y_top": 100.0,
            "y_base": 200.0,
            "y_waterline": 150.0,
            "vehicle_class": "light_vehicle",
        },
    )
    estimate_id = est_res.json()["estimate_id"]

    inc_payload = {
        "estimate_id": estimate_id,
        "camera_id": "CAM_UNDERPASS_04",
        "corridor_name": "MG Road Underpass",
        "closure_state": "SUGGESTED_DIVERSION",
    }

    # First attempt succeeds
    res1 = client.post("/api/incidents", json=inc_payload)
    assert res1.status_code == 201

    # Second attempt fails with 409 Conflict
    res2 = client.post("/api/incidents", json=inc_payload)
    assert res2.status_code == 409
    assert "already been recorded" in res2.json()["detail"]


def test_detection_session_expiry_30_minutes():
    """Verify that a detection session older than 30 minutes expires and returns HTTP 404."""
    # 1. Create a detection session
    detect_res = client.post("/api/detect", json={"image_id": "flooded_vehicle_underpass.jpg"})
    assert detect_res.status_code == 200
    detection_id = detect_res.json()["detection_id"]

    # 2. Simulate expiration by aging the session created_at by 1801 seconds (30m + 1s)
    session = session_store.get_detection_session(detection_id)
    assert session is not None
    session.created_at -= 1801

    # 3. Submit estimate using expired detection_id -> must return 404
    payload = {
        "reference_height_cm": 60.0,
        "y_top": 100.0,
        "y_base": 200.0,
        "y_waterline": 150.0,
        "vehicle_class": "light_vehicle",
        "detection_id": detection_id,
    }
    res = client.post("/api/estimate", json=payload)
    assert res.status_code == 404
    assert "not found or expired" in res.json()["detail"]


def test_estimate_session_expiry_60_minutes():
    """Verify that an estimate session older than 60 minutes expires and cannot create an incident."""
    # 1. Produce valid estimate
    est_res = client.post(
        "/api/estimate",
        json={
            "reference_height_cm": 60.0,
            "y_top": 100.0,
            "y_base": 200.0,
            "y_waterline": 150.0,
            "vehicle_class": "light_vehicle",
        },
    )
    assert est_res.status_code == 200
    estimate_id = est_res.json()["estimate_id"]

    # 2. Simulate expiration by aging created_at by 3601 seconds (60m + 1s)
    est_session = session_store.get_estimate_session(estimate_id)
    assert est_session is not None
    est_session.created_at -= 3601

    # 3. Attempt to create incident from expired estimate -> must return 404
    inc_payload = {
        "estimate_id": estimate_id,
        "camera_id": "CAM_UNDERPASS_04",
        "corridor_name": "MG Road Underpass",
        "closure_state": "SUGGESTED_DIVERSION",
    }
    res = client.post("/api/incidents", json=inc_payload)
    assert res.status_code == 404
    assert "not found or expired" in res.json()["detail"]
def test_create_manual_flood_report():
    """Verify that a manual flood report can be submitted successfully."""

    payload = {
        "report_type": "MANUAL",
        "location": "Test Village",
        "description": "Flooding reported near the village road",
        "reported_severity": "HIGH",
        "latitude": 28.6139,
        "longitude": 77.2090,
    }

    response = client.post("/api/reports", json=payload)

    assert response.status_code == 201

    data = response.json()
    assert data["report_type"] == "MANUAL"
    assert data["location"] == "Test Village"
    assert data["verification_status"] == "UNVERIFIED"
    assert data["reported_severity"] == "HIGH"


def test_get_flood_reports():
    """Verify that submitted flood reports can be retrieved."""

    response = client.get("/api/reports")

    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_verify_flood_report():
    """Verify that an operator can mark a report as verified."""

    create_response = client.post(
        "/api/reports",
        json={
            "report_type": "MANUAL",
            "location": "Verification Test Village",
            "description": "Testing report verification",
        },
    )

    assert create_response.status_code == 201
    report_id = create_response.json()["report_id"]

    response = client.patch(
        f"/api/reports/{report_id}/verification",
        json={"verification_status": "VERIFIED"},
        headers={"x-operator-api-key": os.environ["AQUAGAUGE_OPERATOR_API_KEY"]},
    )

    assert response.status_code == 200
    assert response.json()["verification_status"] == "VERIFIED"


def test_reject_flood_report():
    """Verify that a report can be marked as rejected."""

    create_response = client.post(
        "/api/reports",
        json={
            "report_type": "COMMUNITY",
            "location": "Rejection Test Village",
            "description": "Testing report rejection",
        },
    )

    assert create_response.status_code == 201
    report_id = create_response.json()["report_id"]

    response = client.patch(
        f"/api/reports/{report_id}/verification",
        json={"verification_status": "REJECTED"},
        headers={"x-operator-api-key": os.environ["AQUAGAUGE_OPERATOR_API_KEY"]},
    )

    assert response.status_code == 200
    assert response.json()["verification_status"] == "REJECTED"


def test_verify_nonexistent_flood_report():
    """Verify that an unknown report ID returns HTTP 404."""

    response = client.patch(
        "/api/reports/UNKNOWN-REPORT/verification",
        json={"verification_status": "VERIFIED"},
        headers={"x-operator-api-key": os.environ["AQUAGAUGE_OPERATOR_API_KEY"]},
    )

    assert response.status_code == 404
