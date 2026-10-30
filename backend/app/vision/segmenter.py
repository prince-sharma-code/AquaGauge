"""Classical OpenCV Floodwater Segmenter for AquaGauge (Person 2 - Milestone 3).

Uses multi-cue classical computer vision (HSV, Lab, texture smoothness, edge suppression,
and spatial priors) to extract candidate floodwater surfaces on road scenes without requiring
heavy neural network downloads or GPU execution.

Supports operator-in-the-loop polygon overrides and enforces explicit abstention when
segmentation cues are ambiguous or unverified.

IMPORTANT: Image segmentation identifies 2D visual surface boundaries; it does NOT
by itself determine calibrated physical water depth.
"""
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from .schemas import WaterRegionResult, WaterSegmentationMethod


class FloodwaterSegmenter:
    """Multi-cue classical OpenCV segmenter for urban road floodwater."""

    def __init__(
        self,
        horizon_ratio: float = 0.25,
        min_area_pct: float = 2.0,
        max_area_pct: float = 95.0,
        base_confidence: float = 0.75,
    ):
        """Initialize segmenter parameters.

        Args:
            horizon_ratio: Fraction of top frame excluded by spatial prior (sky/overpasses).
            min_area_pct: Minimum contiguous water area percentage to avoid false triggers.
            max_area_pct: Maximum allowed water area to prevent whole-frame false positives.
            base_confidence: Baseline confidence assigned to clear heuristic segmentation.
        """
        self.horizon_ratio = max(0.0, min(0.6, horizon_ratio))
        self.min_area_pct = min_area_pct
        self.max_area_pct = max_area_pct
        self.base_confidence = base_confidence

    def _load_bgr(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[np.ndarray, int, int]:
        """Convert various input types to a valid BGR numpy array."""
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {img_path}")
            bgr = cv2.imread(str(img_path))
            if bgr is None or bgr.size == 0:
                raise ValueError(f"Failed to read or decode image file '{img_path}'.")
            h, w = bgr.shape[:2]
            return bgr, w, h

        if isinstance(image_input, Image.Image):
            w, h = image_input.width, image_input.height
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid PIL Image dimensions: {w}x{h}")
            rgb = np.array(image_input.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr, w, h

        if isinstance(image_input, np.ndarray):
            if image_input.size == 0 or len(image_input.shape) < 2:
                raise ValueError("Input numpy array is empty or has invalid shape.")
            h, w = image_input.shape[:2]
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid image dimensions: {w}x{h}")
            if len(image_input.shape) == 2:
                bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                bgr = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            else:
                bgr = image_input.copy()
            return bgr, w, h

        raise TypeError(
            f"Unsupported image input type '{type(image_input)}'. "
            "Expected str, Path, PIL.Image, or np.ndarray."
        )

    def _segment_operator_polygon(
        self,
        polygon: Union[List[Tuple[float, float]], List[List[Tuple[float, float]]]],
        width: int,
        height: int,
    ) -> Tuple[WaterRegionResult, np.ndarray]:
        """Generate water mask from operator-confirmed polygon coordinates."""
        mask = np.zeros((height, width), dtype=np.uint8)

        # Normalize polygon representation to list of polygons
        if not polygon or len(polygon) == 0:
            return (
                WaterRegionResult(
                    detected=False,
                    confidence=0.0,
                    area_percentage=0.0,
                    method="abstain",
                    note="Empty operator polygon provided; abstaining.",
                ),
                mask,
            )

        # Check if polygon is a single list of (x, y) points or a list of point lists
        if isinstance(polygon[0], (tuple, list)) and len(polygon[0]) == 2 and isinstance(polygon[0][0], (int, float)):
            poly_list = [polygon]
        else:
            poly_list = polygon

        formatted_polys = []
        for p in poly_list:
            pts = np.array(p, dtype=np.int32).reshape((-1, 1, 2))
            formatted_polys.append(pts)

        cv2.fillPoly(mask, formatted_polys, color=255)
        pixel_count = np.count_nonzero(mask)
        total_pixels = width * height
        area_pct = round((pixel_count / total_pixels) * 100.0, 2)

        # Serialize polygon points as list of tuples for schema
        poly_points = [
            [(float(pt[0]), float(pt[1])) for pt in p]
            for p in poly_list
        ]

        result = WaterRegionResult(
            detected=pixel_count > 0,
            confidence=1.0,
            area_percentage=area_pct,
            polygon_points=poly_points,
            method="operator_polygon",
            note="Water surface defined and confirmed by human operator.",
        )
        return result, mask

    def segment(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        operator_polygon: Optional[Union[List[Tuple[float, float]], List[List[Tuple[float, float]]]]] = None,
        return_mask: bool = False,
    ) -> Union[WaterRegionResult, Tuple[WaterRegionResult, np.ndarray]]:
        """Perform floodwater region segmentation.

        Args:
            image_input: Image file path, PIL Image, or numpy array.
            operator_polygon: Optional human-in-the-loop polygon coordinates [(x, y), ...].
            return_mask: If True, returns (WaterRegionResult, binary_mask_uint8).

        Returns:
            WaterRegionResult or tuple of (WaterRegionResult, mask_ndarray).
        """
        bgr, width, height = self._load_bgr(image_input)
        total_pixels = width * height

        # 1. Handle operator override if provided
        if operator_polygon is not None:
            res, mask = self._segment_operator_polygon(operator_polygon, width, height)
            return (res, mask) if return_mask else res

        # 2. Check for blank/flat or degenerate images
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray))
        if std_dev < 5.0:
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            res = WaterRegionResult(
                detected=False,
                confidence=0.0,
                area_percentage=0.0,
                method="abstain",
                note="Image is uniform or low-contrast (std_dev < 5.0); no water features discernible.",
            )
            return (res, empty_mask) if return_mask else res

        # 3. Spatial context: exclude upper sky/overhead horizon
        horizon_y = int(height * self.horizon_ratio)

        # 4. Multi-cue color and texture extraction
        # Smooth high-frequency noise while preserving boundary edges
        blurred = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        h_chan, s_chan, v_chan = cv2.split(hsv)
        l_chan, a_chan, b_chan = cv2.split(lab)

        # Cue A: Saturation filter (standing water/murky pools have low to moderate saturation)
        sat_mask = (s_chan < 115).astype(np.uint8) * 255

        # Cue B: Chromaticity in Lab (neutral earthy tones, rejecting saturated objects)
        chroma_mask = (
            (a_chan >= 115) & (a_chan <= 142) &
            (b_chan >= 115) & (b_chan <= 145)
        ).astype(np.uint8) * 255

        # Cue C: Texture smoothness & Edge suppression (water surfaces lack high-frequency gravel edges)
        edges = cv2.Canny(gray, threshold1=30, threshold2=100)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_edges = cv2.dilate(edges, kernel_edge, iterations=1)
        smooth_mask = (dilated_edges == 0).astype(np.uint8) * 255

        # Combine cues
        combined = cv2.bitwise_and(sat_mask, chroma_mask)
        combined = cv2.bitwise_and(combined, smooth_mask)

        # Suppress above horizon
        combined[:horizon_y, :] = 0

        # 5. Morphological filtering to bridge water surface pools
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_morph)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_morph)

        # 6. Contour extraction & area thresholding
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = total_pixels * (self.min_area_pct / 100.0)

        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        total_water_area = sum(cv2.contourArea(c) for c in valid_contours)
        area_pct = round((total_water_area / total_pixels) * 100.0, 2)

        final_mask = np.zeros((height, width), dtype=np.uint8)
        if valid_contours:
            cv2.drawContours(final_mask, valid_contours, -1, color=255, thickness=-1)

        # 7. Abstain / Validity Checks
        if area_pct < self.min_area_pct:
            res = WaterRegionResult(
                detected=False,
                confidence=0.0,
                area_percentage=0.0,
                method="abstain",
                note=f"No significant standing water detected (<{self.min_area_pct}% coverage); operator review recommended.",
            )
            return (res, final_mask) if return_mask else res

        if area_pct > self.max_area_pct:
            res = WaterRegionResult(
                detected=False,
                confidence=0.0,
                area_percentage=area_pct,
                method="abstain",
                note="Water mask exceeds plausible road bounds (>95%); possible whole-frame reflection or camera flare.",
            )
            return (res, final_mask) if return_mask else res

        # Extract simplified polygon points for UI transmission
        simplified_polygons: List[List[Tuple[float, float]]] = []
        for c in valid_contours:
            epsilon = 0.005 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            pts = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            if len(pts) >= 3:
                simplified_polygons.append(pts)

        # Confidence heuristic based on area coherence and single dominant pool vs fragmented pools
        coherence_factor = min(1.0, cv2.contourArea(valid_contours[0]) / (total_water_area + 1e-5))
        confidence = round(self.base_confidence * (0.8 + 0.2 * coherence_factor), 2)

        res = WaterRegionResult(
            detected=True,
            confidence=confidence,
            area_percentage=area_pct,
            polygon_points=simplified_polygons if simplified_polygons else None,
            method="classical_cv_heuristic",
            note="Candidate floodwater surface detected using multi-cue color and texture analysis.",
        )
        return (res, final_mask) if return_mask else res
