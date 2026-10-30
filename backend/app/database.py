"""SQLite storage manager for incident logging with transactional duplicate prevention."""

import sqlite3
import datetime
import uuid

from pathlib import Path
from typing import List, Optional

from .models import (
    IncidentResponse,
    IncidentCreateRequest,
    FloodReportCreate,
    FloodReportResponse,
)
from .session_store import EstimateSession, SessionStore


# Ensure local data directory exists inside backend/data/ (ignored by git)
_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_PATH = _DB_DIR / "aquagauge.db"


class DuplicateIncidentError(Exception):
    """Raised when an incident has already been recorded for this estimate_id."""
    pass


class AbstainedIncidentError(Exception):
    """Raised when an attempt is made to log a depth incident from an abstained estimate."""
    pass


def get_db_connection() -> sqlite3.Connection:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize incidents table with UNIQUE constraint on estimate_id."""
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                estimate_id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                corridor_name TEXT NOT NULL,
                estimated_depth_cm REAL,
                road_status TEXT NOT NULL,
                vehicle_risk TEXT NOT NULL,
                vehicle_class TEXT NOT NULL,
                source TEXT NOT NULL,
                closure_state TEXT NOT NULL,
                operator_notes TEXT,
                image_id TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_incidents_timestamp 
            ON incidents(timestamp DESC);
            """
        )
        
        # Multi-source flood reports table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                report_type TEXT NOT NULL,
                location TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                description TEXT NOT NULL,
                reported_severity TEXT NOT NULL DEFAULT 'UNKNOWN',
                verification_status TEXT NOT NULL DEFAULT 'UNVERIFIED',
                image_id TEXT,
                water_level_cm REAL,
                rainfall_mm REAL,
                observed_at TEXT,
                reporter_notes TEXT,
                submitted_at TEXT NOT NULL,
                CHECK (report_type IN (
                    'IMAGE', 'MANUAL', 'COMMUNITY', 'RAINFALL',
                    'WATER_LEVEL', 'SATELLITE', 'SMS', 'OFFLINE'
                )),
                CHECK (verification_status IN (
                    'UNVERIFIED', 'VERIFIED', 'REJECTED'
                ))
            );
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reports_submitted_at
            ON reports(submitted_at DESC);
            """
        )

        conn.commit()


def create_incident_atomic(
    request: IncidentCreateRequest,
    estimate: EstimateSession,
    session_store: SessionStore,
) -> IncidentResponse:
    """Atomically commit an incident into SQLite and update in-memory session.

    Guarantees:
      1. Rejects estimates with status "ABSTAIN / REVIEW".
      2. The SQLite UNIQUE(estimate_id) constraint is the authoritative mechanism.
      3. Commit must succeed BEFORE marking session.consumed_for_incident = True.
      4. Database failures or duplicates do not mark the session consumed.
    """
    if estimate.status == "ABSTAIN / REVIEW" or estimate.estimated_depth_cm is None:
        raise AbstainedIncidentError(
            "Cannot create a depth-based incident from an abstained estimate ('ABSTAIN / REVIEW')."
        )

    # Thread-safe preliminary in-memory check
    if session_store.is_estimate_consumed(estimate.estimate_id):
        raise DuplicateIncidentError(
            f"An incident has already been recorded for estimate '{estimate.estimate_id}'."
        )

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Generate clean human-readable incident id with date prefix
    date_prefix = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    short_hash = estimate.estimate_id.replace("est_", "")[:6]
    incident_id = f"INC-{date_prefix}-{short_hash.upper()}"

    conn = get_db_connection()
    try:
        with conn:
            # Begin explicit transaction
            conn.execute(
                """
                INSERT INTO incidents (
                    incident_id,
                    estimate_id,
                    timestamp,
                    camera_id,
                    corridor_name,
                    estimated_depth_cm,
                    road_status,
                    vehicle_risk,
                    vehicle_class,
                    source,
                    closure_state,
                    operator_notes,
                    image_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    incident_id,
                    estimate.estimate_id,
                    timestamp,
                    request.camera_id,
                    request.corridor_name,
                    estimate.estimated_depth_cm,
                    estimate.road_status,
                    estimate.vehicle_risk,
                    estimate.vehicle_class,
                    estimate.source,
                    request.closure_state,
                    request.operator_notes or "",
                    request.image_id,
                ),
            )
        # Successfully committed transaction: now update in-memory flag
        session_store.mark_estimate_consumed(estimate.estimate_id)
    except sqlite3.IntegrityError as exc:
        raise DuplicateIncidentError(
            f"An incident has already been recorded for estimate '{estimate.estimate_id}'. Duplicate incident creation is prohibited."
        ) from exc
    finally:
        conn.close()

    return IncidentResponse(
        incident_id=incident_id,
        estimate_id=estimate.estimate_id,
        timestamp=timestamp,
        camera_id=request.camera_id,
        corridor_name=request.corridor_name,
        estimated_depth_cm=estimate.estimated_depth_cm,
        road_status=estimate.road_status,
        vehicle_risk=estimate.vehicle_risk,
        vehicle_class=estimate.vehicle_class,
        source=estimate.source,
        closure_state=request.closure_state,
        operator_notes=request.operator_notes or "",
        image_id=request.image_id,
    )


def get_recent_incidents(limit: int = 50) -> List[IncidentResponse]:
    """Retrieve historical incidents ordered by timestamp descending."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            SELECT incident_id, estimate_id, timestamp, camera_id, corridor_name,
                   estimated_depth_cm, road_status, vehicle_risk, vehicle_class,
                   source, closure_state, operator_notes, image_id
            FROM incidents
            ORDER BY timestamp DESC
            LIMIT ?;
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            IncidentResponse(
                incident_id=row["incident_id"],
                estimate_id=row["estimate_id"],
                timestamp=row["timestamp"],
                camera_id=row["camera_id"],
                corridor_name=row["corridor_name"],
                estimated_depth_cm=row["estimated_depth_cm"],
                road_status=row["road_status"],
                vehicle_risk=row["vehicle_risk"],
                vehicle_class=row["vehicle_class"],
                source=row["source"],
                closure_state=row["closure_state"],
                operator_notes=row["operator_notes"],
                image_id=row["image_id"],
            )
            for row in rows
        ]
    finally:
        conn.close()


def create_flood_report(request: FloodReportCreate) -> FloodReportResponse:
    """Save a multi-source flood report in SQLite."""

    init_db()

    report_id = f"RPT-{uuid.uuid4().hex[:12].upper()}"
    submitted_at = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO reports (
                report_id,
                report_type,
                location,
                latitude,
                longitude,
                description,
                reported_severity,
                verification_status,
                image_id,
                water_level_cm,
                rainfall_mm,
                observed_at,
                reporter_notes,
                submitted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                report_id,
                request.report_type,
                request.location,
                request.latitude,
                request.longitude,
                request.description,
                request.reported_severity or "UNKNOWN",
                "UNVERIFIED",
                request.image_id,
                request.water_level_cm,
                request.rainfall_mm,
                request.observed_at,
                request.reporter_notes,
                submitted_at,
            ),
        )

    return FloodReportResponse(
        report_id=report_id,
        report_type=request.report_type,
        location=request.location,
        latitude=request.latitude,
        longitude=request.longitude,
        description=request.description,
        reported_severity=request.reported_severity or "UNKNOWN",
        verification_status="UNVERIFIED",
        image_id=request.image_id,
        water_level_cm=request.water_level_cm,
        rainfall_mm=request.rainfall_mm,
        observed_at=request.observed_at,
        reporter_notes=request.reporter_notes,
        submitted_at=submitted_at,
    )

def get_flood_reports(limit: int = 100) -> List[FloodReportResponse]:
    """Retrieve recent multi-source flood reports from SQLite."""

    init_db()

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM reports
            ORDER BY submitted_at DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()

    return [
        FloodReportResponse(**dict(row))
        for row in rows
    ]

def verify_flood_report(
    report_id: str,
    verification_status: str,
) -> Optional[FloodReportResponse]:
    """Update the verification status of an existing flood report."""

    if verification_status not in ("VERIFIED", "REJECTED"):
        raise ValueError("Invalid verification status.")

    init_db()

    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE reports
            SET verification_status = ?
            WHERE report_id = ?;
            """,
            (verification_status, report_id),
        )

        row = conn.execute(
            """
            SELECT *
            FROM reports
            WHERE report_id = ?;
            """,
            (report_id,),
        ).fetchone()

    if row is None:
        return None

    return FloodReportResponse(**dict(row))