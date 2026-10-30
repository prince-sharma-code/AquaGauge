"""Unified Vision Pipeline for AquaGauge (Person 2 - Milestone 5).

Orchestrates vehicle detection, floodwater segmentation, waterline extraction,
and visual annotation into a cohesive, production-grade computer vision pipeline.

Maintains strict separation between 2D visual perception and physical depth calibration:
- Object detection locates vehicles/obstacles (does NOT detect water).
- Floodwater segmentation marks 2D surface coverage.
- Waterline extraction validates geometric contact coordinates.
- Depth estimation is deferred to depth_estimator.py via reference calibration.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from .annotator import ImageAnnotator
from .detector import VehicleDetector
from .fixtures import get_demo_fixture
from .schemas import (
    VisionPipelineResult,
    VisionPipelineStatus,
    WaterRegionResult,
)
from .segmenter import FloodwaterSegmenter
from .waterline import WaterlineExtractor


class VisionPipeline:
    """Coordinated multi-stage computer vision pipeline for urban flood monitoring."""

    def __init__(
        self,
        detector: Optional[VehicleDetector] = None,
        segmenter: Optional[FloodwaterSegmenter] = None,
        waterline_extractor: Optional[WaterlineExtractor] = None,
        annotator: Optional[ImageAnnotator] = None,
        use_fixture_fallback: bool = False,
    ):
        """Initialize pipeline with components (supports dependency injection for testing).

        Args:
            detector: VehicleDetector instance or mock.
            segmenter: FloodwaterSegmenter instance or mock.
            waterline_extractor: WaterlineExtractor instance or mock.
            annotator: ImageAnnotator instance or mock.
            use_fixture_fallback: If True, falls back to precomputed demo fixtures on failure.
        """
        self.detector = detector if detector is not None else VehicleDetector()
        self.segmenter = segmenter if segmenter is not None else FloodwaterSegmenter()
        self.waterline_extractor = (
            waterline_extractor if waterline_extractor is not None else WaterlineExtractor()
        )
        self.annotator = annotator if annotator is not None else ImageAnnotator()
        self.use_fixture_fallback = use_fixture_fallback

    def _resolve_image_meta(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        image_id: Optional[str] = None,
    ) -> Tuple[str, int, int]:
        """Extract image dimensions and identifier."""
        derived_id = image_id or "unnamed_input"
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            derived_id = image_id or p.name
            with Image.open(p) as img:
                return derived_id, img.width, img.height

        if isinstance(image_input, Image.Image):
            return derived_id, image_input.width, image_input.height

        if isinstance(image_input, np.ndarray):
            h, w = image_input.shape[:2]
            return derived_id, w, h

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def process(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        operator_polygon: Optional[Union[List[Tuple[float, float]], List[List[Tuple[float, float]]]]] = None,
        operator_waterlines: Optional[Dict[int, float]] = None,
        image_id: Optional[str] = None,
        generate_annotated_image: bool = True,
    ) -> VisionPipelineResult:
        """Execute end-to-end vision pipeline.

        Args:
            image_input: File path, PIL Image, or numpy array.
            operator_polygon: Optional human-confirmed water polygon.
            operator_waterlines: Optional manual waterline mappings (vehicle_id -> y_waterline).
            image_id: Optional identifier string.
            generate_annotated_image: If True, renders base64-encoded annotated JPEG.

        Returns:
            Structured VisionPipelineResult matching the API contract.
        """
        try:
            img_id, width, height = self._resolve_image_meta(image_input, image_id)
        except Exception as exc:
            if self.use_fixture_fallback and image_id:
                fixture = get_demo_fixture(image_id)
                if fixture:
                    fixture.status = "FALLBACK"
                    fixture.note = f"Primary image decode failed; served fallback fixture ({exc})."
                    return fixture
            raise

        try:
            # Stage 1: Vehicle Detection
            detections = self.detector.detect(image_input)

            # Stage 2: Floodwater Segmentation
            water_region, mask = self.segmenter.segment(
                image_input,
                operator_polygon=operator_polygon,
                return_mask=True,
            )

            # Stage 3: Waterline Candidate Extraction & Geometric Validation
            waterline_candidates = self.waterline_extractor.extract_candidates(
                mask=mask,
                detections=detections,
                operator_waterlines=operator_waterlines,
            )

            # Stage 4: Pipeline Status Determination
            if len(detections) == 0:
                pipeline_status: VisionPipelineStatus = "NO_DETECTION"
            elif any(c.can_estimate for c in waterline_candidates):
                pipeline_status = "SUCCESS"
            else:
                pipeline_status = "PARTIAL_ABSTAIN"

            # Stage 5: Visual Annotation Overlay
            annotated_b64: Optional[str] = None
            if generate_annotated_image:
                annotated_img = self.annotator.annotate(
                    image_input=image_input,
                    detections=detections,
                    water_mask=mask,
                    waterline_candidates=waterline_candidates,
                    show_legend=True,
                )
                annotated_b64 = self.annotator.to_base64_jpeg(annotated_img)

            return VisionPipelineResult(
                status=pipeline_status,
                image_id=img_id,
                image_width=width,
                image_height=height,
                detections=detections,
                water_region=water_region,
                waterline_candidates=waterline_candidates,
                annotated_image_base64=annotated_b64,
                note="Automated visual detections; physical water depth requires reference geometry calibration.",
            )

        except Exception as exc:
            if self.use_fixture_fallback and image_id:
                fixture = get_demo_fixture(image_id)
                if fixture:
                    fixture.status = "FALLBACK"
                    fixture.note = f"Pipeline execution failed ({exc}); served precomputed demo fixture."
                    return fixture

            # Return graceful partial failure result rather than crashing
            empty_water = WaterRegionResult(
                detected=False,
                confidence=0.0,
                area_percentage=0.0,
                method="abstain",
                note=f"Pipeline error occurred: {exc}",
            )
            return VisionPipelineResult(
                status="FALLBACK",
                image_id=img_id,
                image_width=width,
                image_height=height,
                detections=[],
                water_region=empty_water,
                waterline_candidates=[],
                annotated_image_base64=None,
                note=f"Pipeline error: {exc}. Abstaining from depth candidate emission.",
            )
