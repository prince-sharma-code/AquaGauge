"""AquaGauge Computer Vision Module.

Owned by Person 2.
Provides vehicle detection, water region segmentation, waterline extraction,
visual annotation overlays, and unified pipeline processing.
"""
from .schemas import (
    BoxPixel,
    BoxNormalized,
    VehicleClass,
    VehicleDetection,
    WaterRegionResult,
    WaterlineCandidate,
    VisionPipelineResult,
    pixel_to_normalized,
    normalized_to_pixel,
)
from .fixtures import (
    DEMO_FIXTURES,
    get_demo_fixture,
)
from .detector import (
    VehicleDetector,
    TARGET_VEHICLE_CLASSES,
    find_default_weights,
)
from .segmenter import (
    FloodwaterSegmenter,
)
from .waterline import (
    WaterlineExtractor,
)
from .annotator import (
    ImageAnnotator,
)
from .pipeline import (
    VisionPipeline,
)

__all__ = [
    "BoxPixel",
    "BoxNormalized",
    "VehicleClass",
    "VehicleDetection",
    "WaterRegionResult",
    "WaterlineCandidate",
    "VisionPipelineResult",
    "pixel_to_normalized",
    "normalized_to_pixel",
    "TARGET_VEHICLE_CLASSES",
    "DEMO_FIXTURES",
    "get_demo_fixture",
    "VehicleDetector",
    "find_default_weights",
    "FloodwaterSegmenter",
    "WaterlineExtractor",
    "ImageAnnotator",
    "VisionPipeline",
]
