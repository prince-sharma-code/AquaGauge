"""In-memory session storage for transient detection and estimate lifecycles.

Provenance tracking represents backend-verified processing history across API stages.
It does not constitute cryptographic non-repudiation and sessions are lost on server restart.
"""
import time
import uuid
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal


ProvenanceSource = Literal["MODEL", "OPERATOR_CONFIRMED", "PRECOMPUTED_DEMO"]


@dataclass
class CandidateGeometry:
    y_top: float
    y_base: float
    y_waterline: float
    reference_height_cm: float


@dataclass
class DetectionSession:
    detection_id: str
    image_id: str
    detection_item_id: str
    created_at: float
    source: ProvenanceSource
    candidate_geometry: CandidateGeometry


@dataclass
class EstimateSession:
    estimate_id: str
    created_at: float
    status: Literal["ESTIMATED", "ABSTAIN / REVIEW"]
    estimated_depth_cm: Optional[float]
    submerged_ratio: Optional[float]
    reference_height_cm: float
    vehicle_class: str
    vehicle_risk: str
    road_status: str
    source: ProvenanceSource
    consumed_for_incident: bool = False


class SessionStore:
    """Thread-safe storage managing detection and estimation session lifecycles."""

    def __init__(self, detection_ttl_seconds: int = 1800, estimate_ttl_seconds: int = 3600):
        self._lock = threading.Lock()
        self._detection_ttl = detection_ttl_seconds  # 30 minutes
        self._estimate_ttl = estimate_ttl_seconds    # 60 minutes
        self._detections: Dict[str, DetectionSession] = {}
        self._estimates: Dict[str, EstimateSession] = {}

    def create_detection_session(
        self,
        image_id: str,
        detection_item_id: str,
        source: ProvenanceSource,
        y_top: float,
        y_base: float,
        y_waterline: float,
        reference_height_cm: float,
    ) -> str:
        with self._lock:
            detection_id = f"det_sess_{uuid.uuid4().hex[:12]}"
            session = DetectionSession(
                detection_id=detection_id,
                image_id=image_id,
                detection_item_id=detection_item_id,
                created_at=time.time(),
                source=source,
                candidate_geometry=CandidateGeometry(
                    y_top=round(float(y_top), 2),
                    y_base=round(float(y_base), 2),
                    y_waterline=round(float(y_waterline), 2),
                    reference_height_cm=round(float(reference_height_cm), 2),
                ),
            )
            self._detections[detection_id] = session
            return detection_id

    def get_detection_session(self, detection_id: str) -> Optional[DetectionSession]:
        with self._lock:
            session = self._detections.get(detection_id)
            if not session:
                return None
            if time.time() - session.created_at > self._detection_ttl:
                del self._detections[detection_id]
                return None
            return session

    def evaluate_provenance(
        self,
        detection_id: Optional[str],
        y_top: float,
        y_base: float,
        y_waterline: float,
        reference_height_cm: float,
    ) -> ProvenanceSource:
        """Evaluate coordinates against candidate proposal.

        Returns:
            - "OPERATOR_CONFIRMED" if detection_id is None.
            - Original session source if all coordinates match within 0.5px and height within 0.1cm.
            - "OPERATOR_CONFIRMED" if any coordinate was modified.
        Raises:
            KeyError: If detection_id was provided but not found or expired.
        """
        if detection_id is None:
            return "OPERATOR_CONFIRMED"

        session = self.get_detection_session(detection_id)
        if not session:
            raise KeyError(f"Detection session '{detection_id}' not found or expired.")

        cand = session.candidate_geometry
        diff_top = abs(y_top - cand.y_top)
        diff_base = abs(y_base - cand.y_base)
        diff_water = abs(y_waterline - cand.y_waterline)
        diff_height = abs(reference_height_cm - cand.reference_height_cm)

        # Coordinate tolerance: 0.5px. Reference height tolerance: 0.1cm.
        if diff_top <= 0.5 and diff_base <= 0.5 and diff_water <= 0.5 and diff_height <= 0.1:
            return session.source
        return "OPERATOR_CONFIRMED"

    def create_estimate_session(
        self,
        status: Literal["ESTIMATED", "ABSTAIN / REVIEW"],
        estimated_depth_cm: Optional[float],
        submerged_ratio: Optional[float],
        reference_height_cm: float,
        vehicle_class: str,
        vehicle_risk: str,
        road_status: str,
        source: ProvenanceSource,
    ) -> str:
        with self._lock:
            estimate_id = f"est_{uuid.uuid4().hex[:12]}"
            session = EstimateSession(
                estimate_id=estimate_id,
                created_at=time.time(),
                status=status,
                estimated_depth_cm=estimated_depth_cm,
                submerged_ratio=submerged_ratio,
                reference_height_cm=reference_height_cm,
                vehicle_class=vehicle_class,
                vehicle_risk=vehicle_risk,
                road_status=road_status,
                source=source,
                consumed_for_incident=False,
            )
            self._estimates[estimate_id] = session
            return estimate_id

    def get_estimate_session(self, estimate_id: str) -> Optional[EstimateSession]:
        with self._lock:
            session = self._estimates.get(estimate_id)
            if not session:
                return None
            if time.time() - session.created_at > self._estimate_ttl:
                del self._estimates[estimate_id]
                return None
            return session

    def is_estimate_consumed(self, estimate_id: str) -> bool:
        """Thread-safe check to determine if an estimate was already consumed."""
        with self._lock:
            session = self._estimates.get(estimate_id)
            if not session:
                return False
            return session.consumed_for_incident

    def mark_estimate_consumed(self, estimate_id: str) -> None:
        """Mark an estimate session consumed after a successful database commit."""
        with self._lock:
            session = self._estimates.get(estimate_id)
            if session:
                session.consumed_for_incident = True

    def clear(self) -> None:
        """Helper for unit test isolation."""
        with self._lock:
            self._detections.clear()
            self._estimates.clear()


# Global singleton instance
session_store = SessionStore()
