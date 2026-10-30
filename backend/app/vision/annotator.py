"""Visual Annotator for AquaGauge Computer Vision (Person 2 - Milestone 5).

Renders vehicle bounding boxes, class labels, confidence tags, semi-transparent
floodwater segmentation masks, candidate waterline levels, and operational legends.

IMPORTANT SAFETY NOTICE:
All visual overlays represent 2D image coordinates only. They do not represent
calibrated physical water depth or validated municipal safety barriers.
"""
import base64
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from .schemas import (
    VehicleDetection,
    WaterlineCandidate,
)

# Color palettes (BGR format for OpenCV)
COLOR_VEHICLE_BOX = (0, 230, 115)       # Bright Emerald Green
COLOR_VEHICLE_TEXT = (255, 255, 255)    # White
COLOR_VEHICLE_BG = (20, 20, 20)         # Dark Gray pill background

COLOR_WATER_MASK = (255, 160, 30)       # Cyan/Blue tint for standing water
COLOR_WATERLINE_READY = (0, 255, 255)   # Bright Yellow/Cyan for ready candidate
COLOR_WATERLINE_ABSTAIN = (0, 100, 255) # Orange/Red warning for abstained candidate
COLOR_WATERLINE_OPERATOR = (255, 0, 255)# Magenta for operator-confirmed waterline


class ImageAnnotator:
    """Renders visual overlays onto flood images without mutating the original input."""

    def __init__(
        self,
        mask_alpha: float = 0.35,
        box_thickness: int = 2,
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        """Initialize annotation style parameters.

        Args:
            mask_alpha: Transparency ratio of the water mask overlay [0.0 - 1.0].
            box_thickness: Pixel thickness of vehicle bounding boxes.
            line_thickness: Pixel thickness of waterline indicator lines.
            font_scale: OpenCV Hershey font scaling factor.
        """
        self.mask_alpha = max(0.0, min(1.0, mask_alpha))
        self.box_thickness = max(1, box_thickness)
        self.line_thickness = max(1, line_thickness)
        self.font_scale = max(0.3, font_scale)

    def _to_bgr(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> np.ndarray:
        """Convert input image to a fresh, independent BGR numpy array copy."""
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {img_path}")
            bgr = cv2.imread(str(img_path))
            if bgr is None or bgr.size == 0:
                raise ValueError(f"Failed to read image file: {img_path}")
            return bgr.copy()

        if isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image_input, np.ndarray):
            if image_input.size == 0 or len(image_input.shape) < 2:
                raise ValueError("Image array is empty or has invalid shape.")
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input.copy(), cv2.COLOR_GRAY2BGR)
            if image_input.shape[2] == 4:
                return cv2.cvtColor(image_input.copy(), cv2.COLOR_BGRA2BGR)
            return image_input.copy()

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def annotate(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        detections: Optional[List[VehicleDetection]] = None,
        water_mask: Optional[np.ndarray] = None,
        waterline_candidates: Optional[List[WaterlineCandidate]] = None,
        show_legend: bool = True,
    ) -> np.ndarray:
        """Render detection boxes, water mask, and waterlines onto a copy of the image.

        Args:
            image_input: Input image (never modified in place).
            detections: List of vehicle detections.
            water_mask: 2D binary water mask array (H, W).
            waterline_candidates: List of candidate waterline geometry.
            show_legend: If True, draws top operational status banner.

        Returns:
            Annotated BGR numpy array of identical dimensions.
        """
        canvas = self._to_bgr(image_input)
        h, w = canvas.shape[:2]

        # 1. Overlay floodwater segmentation mask with transparency
        if water_mask is not None and np.count_nonzero(water_mask) > 0:
            if water_mask.shape[:2] == (h, w):
                colored_mask = np.zeros_like(canvas, dtype=np.uint8)
                colored_mask[water_mask > 0] = COLOR_WATER_MASK
                # Alpha blend only in regions where water is present
                mask_bool = water_mask > 0
                canvas[mask_bool] = cv2.addWeighted(
                    canvas[mask_bool], 1.0 - self.mask_alpha,
                    colored_mask[mask_bool], self.mask_alpha,
                    0,
                )

        # 2. Draw vehicle bounding boxes and labels
        if detections:
            for det in detections:
                box = det.box_pixel
                x1, y1 = int(box.x1), int(box.y1)
                x2, y2 = int(box.x2), int(box.y2)

                # Draw box
                cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_VEHICLE_BOX, self.box_thickness)

                # Label text
                label = f"[ID {det.id}] {det.class_name} {det.confidence:.2f}"
                (txt_w, txt_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                )

                # Background pill for label
                pill_y1 = max(0, y1 - txt_h - 6)
                pill_y2 = y1
                cv2.rectangle(
                    canvas,
                    (x1, pill_y1),
                    (x1 + txt_w + 6, pill_y2),
                    COLOR_VEHICLE_BG,
                    thickness=-1,
                )
                cv2.rectangle(
                    canvas,
                    (x1, pill_y1),
                    (x1 + txt_w + 6, pill_y2),
                    COLOR_VEHICLE_BOX,
                    thickness=1,
                )
                cv2.putText(
                    canvas,
                    label,
                    (x1 + 3, pill_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    COLOR_VEHICLE_TEXT,
                    1,
                    cv2.LINE_AA,
                )

        # 3. Draw candidate waterline markers
        if waterline_candidates and detections:
            det_map = {d.id: d for d in detections}
            for cand in waterline_candidates:
                if cand.y_waterline is None:
                    continue

                det = det_map.get(cand.vehicle_id)
                if not det:
                    continue

                box = det.box_pixel
                x1, x2 = int(box.x1), int(box.x2)
                y_w = int(cand.y_waterline)

                # Determine line color and label by candidate status
                if cand.status == "READY_FOR_ESTIMATE":
                    color = COLOR_WATERLINE_READY
                    tag = "WATERLINE: READY"
                elif cand.status == "OPERATOR_CONFIRMED":
                    color = COLOR_WATERLINE_OPERATOR
                    tag = "WATERLINE: OPERATOR"
                else:
                    color = COLOR_WATERLINE_ABSTAIN
                    tag = "WATERLINE: ABSTAIN"

                # Draw waterline segment across vehicle with extended marker handles
                handle_ext = max(10, int((x2 - x1) * 0.1))
                lx1 = max(0, x1 - handle_ext)
                lx2 = min(w, x2 + handle_ext)
                cv2.line(canvas, (lx1, y_w), (lx2, y_w), color, self.line_thickness, cv2.LINE_AA)

                # Small circle handle at ends
                cv2.circle(canvas, (lx1, y_w), 3, color, -1)
                cv2.circle(canvas, (lx2, y_w), 3, color, -1)

                # Waterline tag label
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                tag_x = max(0, min(w - tw - 4, lx1))
                tag_y = max(th + 2, y_w - 4)
                cv2.rectangle(canvas, (tag_x, tag_y - th - 2), (tag_x + tw + 4, tag_y + 2), (10, 10, 10), -1)
                cv2.putText(canvas, tag, (tag_x + 2, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 4. Draw operational status legend banner
        if show_legend:
            banner_h = 24
            cv2.rectangle(canvas, (0, 0), (w, banner_h), (25, 25, 25), -1)
            cv2.line(canvas, (0, banner_h), (w, banner_h), (80, 80, 80), 1)
            legend_txt = "AquaGauge Vision | Visual detections only | Physical depth requires reference calibration"
            cv2.putText(
                canvas,
                legend_txt,
                (10, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return canvas

    @staticmethod
    def to_base64_jpeg(image_bgr: np.ndarray, quality: int = 85) -> str:
        """Encode BGR numpy array to JPEG Base64 string for API transport."""
        success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            raise RuntimeError("Failed to encode annotated image to JPEG.")
        return base64.b64encode(buffer).decode("utf-8")
