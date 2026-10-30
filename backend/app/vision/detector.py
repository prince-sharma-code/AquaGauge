"""Reusable YOLO11n Vehicle Detector for AquaGauge (Person 2 - Milestone 2).

Forces CPU inference and filters detections strictly to road vehicles:
- bicycle (COCO 1)
- car (COCO 2)
- motorcycle (COCO 3)
- bus (COCO 5)
- truck (COCO 7)

All non-vehicle detections (persons, boats, animals, etc.) are strictly filtered out.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from .schemas import (
    BoxNormalized,
    BoxPixel,
    VehicleDetection,
    pixel_to_normalized,
)

# Standard COCO vehicle class mappings
TARGET_VEHICLE_CLASSES: Dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def find_default_weights() -> Path:
    """Locate local yolo11n.pt file without triggering automatic online downloads."""
    candidates = [
        # In current working directory
        Path.cwd() / "yolo11n.pt",
        # In repository root (3 levels up from backend/app/vision/detector.py)
        Path(__file__).resolve().parents[3] / "yolo11n.pt",
        # In assets directory
        Path(__file__).resolve().parents[3] / "assets" / "yolo11n.pt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


class VehicleDetector:
    """Reusable vehicle detector wrapper around Ultralytics YOLO."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        model_instance: Optional[object] = None,
    ):
        """Initialize the vehicle detector.

        Args:
            weights_path: Path to yolo11n.pt. If None, searches standard local paths.
            conf_threshold: Minimum confidence score [0.0, 1.0].
            imgsz: Input resolution for inference (default 640).
            model_instance: Optional pre-loaded or mocked YOLO model instance.
        """
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.target_classes = TARGET_VEHICLE_CLASSES

        if model_instance is not None:
            self.model = model_instance
            self.weights_path = Path("custom_instance")
        else:
            resolved_path = Path(weights_path) if weights_path else find_default_weights()
            if not resolved_path.is_file():
                raise FileNotFoundError(
                    f"YOLO11n weights not found at '{resolved_path}'. "
                    "Ensure yolo11n.pt is in the project root."
                )
            self.weights_path = resolved_path
            from ultralytics import YOLO

            # Load model once on initialization
            self.model = YOLO(str(self.weights_path))

    def _resolve_image_and_dimensions(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[Union[str, Path, Image.Image, np.ndarray], int, int]:
        """Validate input image and determine width and height."""
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {img_path}")
            try:
                with Image.open(img_path) as img:
                    width, height = img.width, img.height
            except Exception as exc:
                raise ValueError(f"Corrupted or invalid image file '{img_path}': {exc}") from exc
            return str(img_path), width, height

        if isinstance(image_input, Image.Image):
            width, height = image_input.width, image_input.height
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid PIL Image dimensions: {width}x{height}")
            return image_input, width, height

        if isinstance(image_input, np.ndarray):
            if image_input.size == 0 or len(image_input.shape) < 2:
                raise ValueError("Input numpy array is empty or has invalid shape.")
            height, width = image_input.shape[0], image_input.shape[1]
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid numpy image dimensions: {width}x{height}")
            return image_input, width, height

        raise TypeError(
            f"Unsupported image input type '{type(image_input)}'. "
            "Expected str, Path, PIL.Image, or np.ndarray."
        )

    def detect(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> List[VehicleDetection]:
        """Run vehicle detection on an image.

        Args:
            image_input: Filepath, PIL Image, or numpy array.
            conf_threshold: Optional threshold override.
            imgsz: Optional image size override.

        Returns:
            List of filtered VehicleDetection instances.
        """
        resolved_img, width, height = self._resolve_image_and_dimensions(image_input)
        effective_conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        effective_imgsz = imgsz if imgsz is not None else self.imgsz

        try:
            results = self.model.predict(
                source=resolved_img,
                conf=effective_conf,
                device="cpu",
                imgsz=effective_imgsz,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed: {exc}") from exc

        detections: List[VehicleDetection] = []
        det_id = 1

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Extract class id, confidence, and coordinates
                raw_cls = box.cls[0].item() if hasattr(box.cls[0], "item") else float(box.cls[0])
                cls_id = int(raw_cls)

                # Strictly filter for vehicle classes (exclude person, boat, etc.)
                if cls_id not in self.target_classes:
                    continue

                class_name = self.target_classes[cls_id]
                raw_conf = box.conf[0].item() if hasattr(box.conf[0], "item") else float(box.conf[0])
                confidence = round(float(raw_conf), 4)

                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                x1, y1, x2, y2 = [float(v) for v in xyxy]

                # Clamp to image boundaries
                x1 = max(0.0, min(float(width), x1))
                y1 = max(0.0, min(float(height), y1))
                x2 = max(0.0, min(float(width), x2))
                y2 = max(0.0, min(float(height), y2))

                # Skip invalid zero-area boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                box_px = BoxPixel(
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    x2=round(x2, 1),
                    y2=round(y2, 1),
                )
                box_norm = pixel_to_normalized(box_px, width, height)

                detections.append(
                    VehicleDetection(
                        id=det_id,
                        class_name=class_name,
                        confidence=confidence,
                        box_pixel=box_px,
                        box_normalized=box_norm,
                    )
                )
                det_id += 1

        return detections
