"""Candidate Waterline Extraction and Geometric Validation for AquaGauge (Person 2 - Milestone 4).

Extracts candidate 2D waterline pixel coordinates by analyzing the spatial intersection
between segmented floodwater masks and detected vehicle bounding boxes.

Strictly validates geometric constraints:
1. Coordinates must reside strictly within image bounds.
2. Candidate lines must have sufficient horizontal contact length across the object.
3. Candidate coordinates must satisfy y_top < y_waterline < y_base for physical depth estimation.
4. When water contact is missing, ambiguous, or exceeds object boundaries, returns an
   explicit ABSTAIN state.

IMPORTANT RESTRICTIONS:
- Does NOT convert pixel coordinates into centimetres.
- Does NOT calculate physical water depth.
- Does NOT assume vehicle top or base is a calibrated physical datum.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

from .schemas import (
    VehicleDetection,
    WaterlineCandidate,
    WaterlineStatus,
)


class WaterlineExtractor:
    """Extracts and validates candidate waterlines from flood masks and vehicle boxes."""

    def __init__(
        self,
        min_width_ratio: float = 0.15,
        min_contact_pixels: int = 5,
        max_std_ratio: float = 0.35,
        boundary_margin_px: float = 3.0,
    ):
        """Initialize extractor parameters.

        Args:
            min_width_ratio: Minimum fraction of vehicle width that must contact the water mask.
            min_contact_pixels: Minimum absolute pixel count for contact support.
            max_std_ratio: Maximum allowable vertical standard deviation relative to vehicle height.
            boundary_margin_px: Minimum pixel margin required between candidate waterline and top/base.
        """
        self.min_width_ratio = min_width_ratio
        self.min_contact_pixels = min_contact_pixels
        self.max_std_ratio = max_std_ratio
        self.boundary_margin_px = boundary_margin_px

    def _validate_mask(self, mask: np.ndarray) -> Tuple[int, int]:
        """Validate that the mask is a non-empty 2D numpy array."""
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"Mask must be a numpy ndarray, got {type(mask)}.")
        if mask.size == 0 or len(mask.shape) != 2:
            raise ValueError(f"Mask must be a non-empty 2D array, got shape {mask.shape}.")
        return mask.shape[1], mask.shape[0]

    def extract_scene_waterline_segments(
        self,
        mask: np.ndarray,
        min_length: int = 30,
    ) -> List[Tuple[float, float, float, float]]:
        """Extract prominent horizontal waterline boundary segments across the scene mask.

        Returns list of segments [(x1, y1, x2, y2), ...] along the upper edge of the water pool.
        """
        w, h = self._validate_mask(mask)
        if np.count_nonzero(mask) == 0:
            return []

        # Find upper edge of water mask using vertical morphological gradient
        kernel = cv2_kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        eroded = cv2_eroded = np.pad(mask[1:, :], ((0, 1), (0, 0)), mode="constant")
        upper_edge = ((mask > 0) & (eroded == 0)).astype(np.uint8) * 255

        # Horizontal line extraction using morphological horizontal kernel
        h_kernel = np.ones((1, min_length), dtype=np.uint8)
        h_lines = np.zeros_like(upper_edge)
        # Scan along columns to find contiguous horizontal segments
        segments: List[Tuple[float, float, float, float]] = []

        # Find connected components along upper edge
        contours, _ = np.zeros(1), []
        try:
            import cv2
            contours, _ = cv2.findContours(upper_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                if cw >= min_length:
                    segments.append((float(x), float(y), float(x + cw), float(y)))
        except Exception:
            pass

        return segments

    def extract_candidates(
        self,
        mask: np.ndarray,
        detections: Optional[List[VehicleDetection]] = None,
        operator_waterlines: Optional[Dict[int, float]] = None,
    ) -> List[WaterlineCandidate]:
        """Extract candidate waterlines for detected vehicles.

        Args:
            mask: 2D binary water mask (H, W) where water > 0.
            detections: List of VehicleDetection instances.
            operator_waterlines: Optional manual override mapping vehicle_id -> y_waterline_pixel.

        Returns:
            List of validated WaterlineCandidate objects.
        """
        img_w, img_h = self._validate_mask(mask)
        candidates: List[WaterlineCandidate] = []

        if not detections:
            return candidates

        for det in detections:
            vehicle_id = det.id
            box = det.box_pixel
            x1, y1 = float(box.x1), float(box.y1)
            x2, y2 = float(box.x2), float(box.y2)
            v_w = x2 - x1
            v_h = y2 - y1

            # 1. Handle operator-confirmed waterline override if provided
            if operator_waterlines and vehicle_id in operator_waterlines:
                op_y = float(operator_waterlines[vehicle_id])
                if y1 < op_y < y2:
                    candidates.append(
                        WaterlineCandidate(
                            vehicle_id=vehicle_id,
                            y_waterline=round(op_y, 1),
                            y_base=round(y2, 1),
                            y_top=round(y1, 1),
                            can_estimate=True,
                            status="OPERATOR_CONFIRMED",
                            reason="Waterline manually set and verified by operator.",
                        )
                    )
                else:
                    candidates.append(
                        WaterlineCandidate(
                            vehicle_id=vehicle_id,
                            y_waterline=round(op_y, 1),
                            y_base=round(y2, 1),
                            y_top=round(y1, 1),
                            can_estimate=False,
                            status="ABSTAIN_UNCLEAR_WATERLINE",
                            reason=f"Operator waterline y={op_y} violates vehicle vertical bounds [{y1}, {y2}].",
                        )
                    )
                continue

            # 2. Geometric validation of vehicle box bounds against image canvas
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h or v_w <= 0 or v_h <= 0:
                candidates.append(
                    WaterlineCandidate(
                        vehicle_id=vehicle_id,
                        y_waterline=None,
                        y_base=max(0.0, min(float(img_h), y2)),
                        y_top=max(0.0, min(float(img_h), y1)),
                        can_estimate=False,
                        status="ABSTAIN_UNCLEAR_WATERLINE",
                        reason="Vehicle bounding box exceeds image dimensions or has degenerate size.",
                    )
                )
                continue

            # 3. Check for water intersection inside vehicle bounding box
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            sub_mask = mask[iy1:iy2, ix1:ix2]
            water_pixel_count = int(np.count_nonzero(sub_mask))

            if water_pixel_count == 0:
                candidates.append(
                    WaterlineCandidate(
                        vehicle_id=vehicle_id,
                        y_waterline=None,
                        y_base=round(y2, 1),
                        y_top=round(y1, 1),
                        can_estimate=False,
                        status="ABSTAIN_NO_INTERSECTION",
                        reason="No intersection between segmented water mask and vehicle bounding box.",
                    )
                )
                continue

            # 4. Column-wise water profile analysis across inner 70% of vehicle width
            # (Insets avoid outer background noise and edge clipping)
            x_start = int(v_w * 0.15)
            x_end = max(x_start + 1, int(v_w * 0.85))
            col_water_ys: List[float] = []

            for c in range(x_start, x_end):
                water_y_indices = np.where(sub_mask[:, c] > 0)[0]
                if len(water_y_indices) > 0:
                    # Topmost water pixel in this vehicle column
                    col_water_ys.append(float(iy1 + water_y_indices[0]))

            required_cols = max(self.min_contact_pixels, int(v_w * self.min_width_ratio))
            if len(col_water_ys) < required_cols:
                candidates.append(
                    WaterlineCandidate(
                        vehicle_id=vehicle_id,
                        y_waterline=None,
                        y_base=round(y2, 1),
                        y_top=round(y1, 1),
                        can_estimate=False,
                        status="ABSTAIN_UNCLEAR_WATERLINE",
                        reason=(
                            f"Waterline contact length too short ({len(col_water_ys)} columns < "
                            f"{required_cols} required); contact is fragmented or ambiguous."
                        ),
                    )
                )
                continue

            # 5. Candidate waterline estimation and geometric bounds checks
            cand_y = float(np.median(col_water_ys))
            std_y = float(np.std(col_water_ys))

            # Case A: Waterline at or near upper roof/hood
            if cand_y <= (y1 + self.boundary_margin_px):
                candidates.append(
                    WaterlineCandidate(
                        vehicle_id=vehicle_id,
                        y_waterline=round(cand_y, 1),
                        y_base=round(y2, 1),
                        y_top=round(y1, 1),
                        can_estimate=False,
                        status="ABSTAIN_UNCLEAR_WATERLINE",
                        reason=(
                            f"Candidate waterline y={cand_y:.1f} touches or exceeds upper vehicle bound "
                            f"(y_top={y1:.1f}); vehicle appears fully submerged or occluded."
                        ),
                    )
                )
            # Case B: Waterline at or below base/contact plane
            elif cand_y >= (y2 - self.boundary_margin_px):
                candidates.append(
                    WaterlineCandidate(
                        vehicle_id=vehicle_id,
                        y_waterline=round(cand_y, 1),
                        y_base=round(y2, 1),
                        y_top=round(y1, 1),
                        can_estimate=False,
                        status="ABSTAIN_UNCLEAR_WATERLINE",
                        reason=(
                            f"Candidate waterline y={cand_y:.1f} is at or below vehicle contact base "
                            f"(y_base={y2:.1f}); vehicle appears on dry ground or only edge-adjacent."
                        ),
                    )
                )
            # Case C: Excessive vertical variance (non-horizontal/turbulent boundary)
            elif std_y > (self.max_std_ratio * v_h):
                candidates.append(
                    WaterlineCandidate(
                        vehicle_id=vehicle_id,
                        y_waterline=round(cand_y, 1),
                        y_base=round(y2, 1),
                        y_top=round(y1, 1),
                        can_estimate=False,
                        status="ABSTAIN_UNCLEAR_WATERLINE",
                        reason=(
                            f"High vertical variance across contact line (std={std_y:.1f} > "
                            f"{self.max_std_ratio * v_h:.1f}); candidate waterline is non-planar."
                        ),
                    )
                )
            # Case D: Valid intermediate candidate waterline
            else:
                candidates.append(
                    WaterlineCandidate(
                        vehicle_id=vehicle_id,
                        y_waterline=round(cand_y, 1),
                        y_base=round(y2, 1),
                        y_top=round(y1, 1),
                        can_estimate=True,
                        status="READY_FOR_ESTIMATE",
                        reason="Candidate waterline geometrically valid for physical depth estimation.",
                    )
                )

        return candidates
