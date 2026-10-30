"""Tests for AquaGauge Computer Vision module schemas, fixtures, and vehicle detector (Person 2 - Milestones 1 & 2)."""
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from pydantic import ValidationError
from PIL import Image

from backend.app.depth_estimator import estimate_depth_cm
from backend.app.vision import (
    BoxPixel,
    BoxNormalized,
    VehicleDetection,
    WaterRegionResult,
    WaterlineCandidate,
    VisionPipelineResult,
    pixel_to_normalized,
    normalized_to_pixel,
    DEMO_FIXTURES,
    get_demo_fixture,
    VehicleDetector,
    TARGET_VEHICLE_CLASSES,
    find_default_weights,
    FloodwaterSegmenter,
    WaterlineExtractor,
    ImageAnnotator,
    VisionPipeline,
)


# ==========================================
# Schema Tests
# ==========================================

def test_box_pixel_valid():
    box = BoxPixel(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
    assert box.x1 == 10.0
    assert box.y1 == 20.0
    assert box.x2 == 100.0
    assert box.y2 == 200.0


def test_box_pixel_invalid_raises():
    with pytest.raises(ValueError):
        BoxPixel(x1=100.0, y1=20.0, x2=50.0, y2=200.0)  # x2 < x1
    with pytest.raises(ValueError):
        BoxPixel(x1=10.0, y1=200.0, x2=100.0, y2=50.0)  # y2 < y1
    with pytest.raises(ValueError):
        BoxPixel(x1=10.0, y1=20.0, x2=10.0, y2=200.0)   # x2 == x1


def test_box_normalized_valid():
    box = BoxNormalized(x1=0.1, y1=0.2, x2=0.8, y2=0.9)
    assert box.x1 == 0.1
    assert box.y2 == 0.9


def test_box_normalized_out_of_range_raises():
    with pytest.raises(ValidationError):
        BoxNormalized(x1=-0.1, y1=0.2, x2=0.8, y2=0.9)
    with pytest.raises(ValidationError):
        BoxNormalized(x1=0.1, y1=0.2, x2=1.5, y2=0.9)


def test_pixel_to_normalized_and_back():
    width, height = 1000, 500
    px = BoxPixel(x1=100.0, y1=50.0, x2=500.0, y2=300.0)
    norm = pixel_to_normalized(px, width, height)
    assert norm.x1 == 0.1
    assert norm.y1 == 0.1
    assert norm.x2 == 0.5
    assert norm.y2 == 0.6

    px_recovered = normalized_to_pixel(norm, width, height)
    assert abs(px_recovered.x1 - px.x1) < 1.0
    assert abs(px_recovered.y2 - px.y2) < 1.0


def test_pixel_to_normalized_invalid_dimensions():
    px = BoxPixel(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
    with pytest.raises(ValueError):
        pixel_to_normalized(px, 0, 500)
    with pytest.raises(ValueError):
        pixel_to_normalized(px, 500, -10)


def test_vehicle_detection_schema():
    px = BoxPixel(x1=50.0, y1=50.0, x2=150.0, y2=150.0)
    norm = BoxNormalized(x1=0.1, y1=0.1, x2=0.3, y2=0.3)
    det = VehicleDetection(
        id=1,
        class_name="car",
        confidence=0.95,
        box_pixel=px,
        box_normalized=norm,
    )
    assert det.class_name in TARGET_VEHICLE_CLASSES.values()
    assert det.confidence == 0.95


def test_water_region_result_defaults():
    water = WaterRegionResult(detected=False)
    assert water.detected is False
    assert water.method == "abstain"
    assert water.confidence == 0.0
    assert water.area_percentage == 0.0


def test_waterline_candidate_schema():
    cand = WaterlineCandidate(
        vehicle_id=1,
        y_waterline=145.0,
        y_base=200.0,
        y_top=100.0,
        can_estimate=True,
        status="READY_FOR_ESTIMATE",
    )
    assert cand.can_estimate is True
    assert cand.y_top < cand.y_waterline < cand.y_base


# ==========================================
# Fixture Tests
# ==========================================

def test_demo_fixtures_present_and_valid():
    expected_keys = {
        "flooded_vehicle_underpass.jpg",
        "waterlogged_road_vehicle.jpg",
        "underpass_overview.jpg",
    }
    assert set(DEMO_FIXTURES.keys()) == expected_keys

    for filename, fixture in DEMO_FIXTURES.items():
        assert isinstance(fixture, VisionPipelineResult)
        assert fixture.status == "SUCCESS"
        assert fixture.image_width > 0
        assert fixture.image_height > 0
        assert len(fixture.detections) > 0
        assert fixture.water_region.detected is True

        for det in fixture.detections:
            assert det.confidence > 0.0
            assert det.box_pixel.x2 <= fixture.image_width
            assert det.box_pixel.y2 <= fixture.image_height

        for cand in fixture.waterline_candidates:
            assert cand.y_base <= fixture.image_height
            assert cand.y_top >= 0.0
            if cand.can_estimate:
                assert cand.y_top < cand.y_waterline < cand.y_base


def test_get_demo_fixture_lookup():
    res1 = get_demo_fixture("flooded_vehicle_underpass.jpg")
    assert res1 is not None
    assert res1.image_width == 1598

    res2 = get_demo_fixture("assets/demo_images/waterlogged_road_vehicle.jpg")
    assert res2 is not None
    assert res2.image_width == 898

    res3 = get_demo_fixture("UNDERPASS_OVERVIEW.JPG")
    assert res3 is not None
    assert res3.image_width == 2880

    res_none = get_demo_fixture("unknown_image.jpg")
    assert res_none is None

    res1.image_width = 9999
    fresh = get_demo_fixture("flooded_vehicle_underpass.jpg")
    assert fresh.image_width == 1598


def test_fixture_depth_estimator_compatibility():
    for filename, fixture in DEMO_FIXTURES.items():
        for cand in fixture.waterline_candidates:
            if cand.can_estimate:
                depth = estimate_depth_cm(
                    reference_height_cm=60.0,
                    y_top=cand.y_top,
                    y_base=cand.y_base,
                    y_waterline=cand.y_waterline,
                )
                assert depth.estimated_depth_cm > 0.0
                assert 0.0 < depth.submerged_ratio < 1.0


# ==========================================
# VehicleDetector Unit Tests (Mocked Inference)
# ==========================================

class _MockBox:
    def __init__(self, cls_id: int, conf: float, xyxy: list[float]):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _MockResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _MockYOLOModel:
    def __init__(self, results=None):
        self.results = results if results is not None else []
        self.last_kwargs = {}

    def predict(self, **kwargs):
        self.last_kwargs = kwargs
        return self.results


def test_detector_missing_weights_raises():
    with pytest.raises(FileNotFoundError):
        VehicleDetector(weights_path="nonexistent_model_weights.pt")


def test_detector_invalid_image_inputs():
    mock_model = _MockYOLOModel()
    detector = VehicleDetector(model_instance=mock_model)

    # Missing file path
    with pytest.raises(FileNotFoundError):
        detector.detect("nonexistent_path_to_image.jpg")

    # Unsupported type
    with pytest.raises(TypeError):
        detector.detect(12345)

    # Empty numpy array
    with pytest.raises(ValueError):
        detector.detect(np.array([]))

    # Zero-dimension numpy array
    with pytest.raises(ValueError):
        detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))


def test_detector_filters_non_vehicle_classes():
    """Verify detector extracts cars, buses, trucks, bikes, motorcycles and rejects persons, boats, dogs."""
    mock_boxes = [
        _MockBox(cls_id=2, conf=0.92, xyxy=[100.0, 100.0, 300.0, 250.0]),   # car (keep)
        _MockBox(cls_id=0, conf=0.88, xyxy=[50.0, 50.0, 80.0, 150.0]),       # person (ignore)
        _MockBox(cls_id=8, conf=0.75, xyxy=[10.0, 10.0, 200.0, 100.0]),     # boat (ignore)
        _MockBox(cls_id=3, conf=0.85, xyxy=[400.0, 200.0, 500.0, 350.0]),   # motorcycle (keep)
        _MockBox(cls_id=7, conf=0.90, xyxy=[600.0, 100.0, 900.0, 400.0]),   # truck (keep)
        _MockBox(cls_id=5, conf=0.80, xyxy=[50.0, 300.0, 250.0, 480.0]),    # bus (keep)
        _MockBox(cls_id=1, conf=0.70, xyxy=[300.0, 300.0, 350.0, 400.0]),   # bicycle (keep)
        _MockBox(cls_id=16, conf=0.65, xyxy=[10.0, 400.0, 50.0, 450.0]),    # dog (ignore)
    ]
    mock_model = _MockYOLOModel(results=[_MockResult(boxes=mock_boxes)])
    detector = VehicleDetector(model_instance=mock_model)

    test_image = Image.new("RGB", (1000, 500), color="white")
    detections = detector.detect(test_image)

    # Expected: 5 vehicles (car, motorcycle, truck, bus, bicycle)
    assert len(detections) == 5
    classes = [d.class_name for d in detections]
    assert classes == ["car", "motorcycle", "truck", "bus", "bicycle"]
    assert "person" not in classes
    assert "boat" not in classes
    assert "dog" not in classes


def test_detector_bounding_boxes_and_coordinates():
    mock_boxes = [
        _MockBox(cls_id=2, conf=0.954, xyxy=[100.0, 50.0, 500.0, 250.0]),
    ]
    mock_model = _MockYOLOModel(results=[_MockResult(boxes=mock_boxes)])
    detector = VehicleDetector(model_instance=mock_model)

    test_image = Image.new("RGB", (1000, 500), color="white")
    detections = detector.detect(test_image)

    assert len(detections) == 1
    det = detections[0]
    assert det.id == 1
    assert det.class_name == "car"
    assert det.confidence == 0.954

    # Check pixel box
    assert det.box_pixel.x1 == 100.0
    assert det.box_pixel.y1 == 50.0
    assert det.box_pixel.x2 == 500.0
    assert det.box_pixel.y2 == 250.0

    # Check normalized box
    assert det.box_normalized.x1 == 0.1
    assert det.box_normalized.y1 == 0.1
    assert det.box_normalized.x2 == 0.5
    assert det.box_normalized.y2 == 0.5


def test_detector_empty_detections():
    mock_model = _MockYOLOModel(results=[_MockResult(boxes=[])])
    detector = VehicleDetector(model_instance=mock_model)

    test_image = Image.new("RGB", (640, 480), color="gray")
    detections = detector.detect(test_image)
    assert detections == []


def test_detector_device_cpu_enforced():
    mock_model = _MockYOLOModel(results=[_MockResult(boxes=[])])
    detector = VehicleDetector(model_instance=mock_model)

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.detect(test_image, conf_threshold=0.3, imgsz=320)

    # Verify predict kwargs passed to model
    assert mock_model.last_kwargs["device"] == "cpu"
    assert mock_model.last_kwargs["conf"] == 0.3
    assert mock_model.last_kwargs["imgsz"] == 320
    assert mock_model.last_kwargs["verbose"] is False


def test_detector_inference_error_handling():
    faulty_model = MagicMock()
    faulty_model.predict.side_effect = RuntimeError("Torch internal memory allocation failure")
    detector = VehicleDetector(model_instance=faulty_model)

    test_image = Image.new("RGB", (100, 100), color="black")
    with pytest.raises(RuntimeError) as exc_info:
        detector.detect(test_image)
    assert "YOLO inference failed" in str(exc_info.value)


# ==========================================
# FloodwaterSegmenter Unit Tests (Classical CV)
# ==========================================

def test_segmenter_invalid_image_inputs():
    seg = FloodwaterSegmenter()

    with pytest.raises(FileNotFoundError):
        seg.segment("nonexistent_flood_image.jpg")

    with pytest.raises(TypeError):
        seg.segment(98765)

    with pytest.raises(ValueError):
        seg.segment(np.array([]))

    with pytest.raises(ValueError):
        seg.segment(np.zeros((0, 0, 3), dtype=np.uint8))


def test_segmenter_blank_image_abstains():
    seg = FloodwaterSegmenter()
    blank = np.full((400, 600, 3), 128, dtype=np.uint8)

    result, mask = seg.segment(blank, return_mask=True)
    assert result.detected is False
    assert result.method == "abstain"
    assert result.confidence == 0.0
    assert result.area_percentage == 0.0
    assert mask.shape == (400, 600)
    assert np.count_nonzero(mask) == 0


def test_segmenter_operator_polygon():
    seg = FloodwaterSegmenter()
    # 500x500 image, 200x200 box in center -> area = 40,000 / 250,000 = 16.0%
    poly = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
    blank = np.zeros((500, 500, 3), dtype=np.uint8)

    result, mask = seg.segment(blank, operator_polygon=poly, return_mask=True)

    assert result.detected is True
    assert result.method == "operator_polygon"
    assert result.confidence == 1.0
    assert abs(result.area_percentage - 16.0) < 0.5
    assert result.polygon_points is not None
    assert mask.shape == (500, 500)
    assert mask[200, 200] == 255
    assert mask[50, 50] == 0


def test_segmenter_empty_operator_polygon_abstains():
    seg = FloodwaterSegmenter()
    blank = np.zeros((200, 200, 3), dtype=np.uint8)

    result, mask = seg.segment(blank, operator_polygon=[], return_mask=True)
    assert result.detected is False
    assert result.method == "abstain"
    assert result.confidence == 0.0
    assert result.area_percentage == 0.0


def test_segmenter_demo_images_detected():
    seg = FloodwaterSegmenter()
    demo_images = [
        "assets/demo_images/flooded_vehicle_underpass.jpg",
        "assets/demo_images/waterlogged_road_vehicle.jpg",
        "assets/demo_images/underpass_overview.jpg",
    ]

    for img_path in demo_images:
        full_path = Path("C:/Users/win11/Downloads/AquaGauge") / img_path
        if not full_path.is_file():
            continue

        result, mask = seg.segment(full_path, return_mask=True)
        assert result.detected is True
        assert result.method == "classical_cv_heuristic"
        assert result.confidence >= 0.5
        assert result.area_percentage >= 10.0
        assert len(mask.shape) == 2
        assert np.count_nonzero(mask) > 0


def test_segmenter_mask_dimensions_match_pil():
    seg = FloodwaterSegmenter()
    pil_img = Image.new("RGB", (320, 240), color=(100, 100, 100))
    result, mask = seg.segment(pil_img, return_mask=True)
    assert mask.shape == (240, 320)


# ==========================================
# WaterlineExtractor Unit Tests
# ==========================================

def test_waterline_invalid_masks():
    ext = WaterlineExtractor()
    with pytest.raises(TypeError):
        ext.extract_candidates("invalid_mask")
    with pytest.raises(ValueError):
        ext.extract_candidates(np.zeros((0, 0), dtype=np.uint8))
    with pytest.raises(ValueError):
        ext.extract_candidates(np.zeros((10, 10, 3), dtype=np.uint8))


def test_waterline_no_detections():
    ext = WaterlineExtractor()
    mask = np.zeros((100, 100), dtype=np.uint8)
    assert ext.extract_candidates(mask, None) == []
    assert ext.extract_candidates(mask, []) == []


def test_waterline_no_intersection():
    ext = WaterlineExtractor()
    # Mask has water on the left, vehicle is on the right
    mask = np.zeros((500, 500), dtype=np.uint8)
    mask[:, :200] = 255

    det = VehicleDetection(
        id=1,
        class_name="car",
        confidence=0.9,
        box_pixel=BoxPixel(x1=300.0, y1=100.0, x2=450.0, y2=250.0),
        box_normalized=BoxNormalized(x1=0.6, y1=0.2, x2=0.9, y2=0.5),
    )

    cands = ext.extract_candidates(mask, [det])
    assert len(cands) == 1
    assert cands[0].can_estimate is False
    assert cands[0].status == "ABSTAIN_NO_INTERSECTION"
    assert cands[0].y_waterline is None


def test_waterline_ready_for_estimate():
    ext = WaterlineExtractor()
    # 500x500 canvas. Vehicle is [x: 100-300, y: 100-300].
    # Water enters from bottom and fills y: 200-500.
    mask = np.zeros((500, 500), dtype=np.uint8)
    mask[200:500, :] = 255

    det = VehicleDetection(
        id=1,
        class_name="car",
        confidence=0.95,
        box_pixel=BoxPixel(x1=100.0, y1=100.0, x2=300.0, y2=300.0),
        box_normalized=BoxNormalized(x1=0.2, y1=0.2, x2=0.6, y2=0.6),
    )

    cands = ext.extract_candidates(mask, [det])
    assert len(cands) == 1
    cand = cands[0]
    assert cand.can_estimate is True
    assert cand.status == "READY_FOR_ESTIMATE"
    assert cand.y_waterline == 200.0
    assert cand.y_top == 100.0
    assert cand.y_base == 300.0

    # Verify direct compatibility with depth_estimator.py
    depth = estimate_depth_cm(
        reference_height_cm=60.0,
        y_top=cand.y_top,
        y_base=cand.y_base,
        y_waterline=cand.y_waterline,
    )
    assert depth.estimated_depth_cm == 30.0
    assert depth.submerged_ratio == 0.5


def test_waterline_submerged_roof_abstains():
    ext = WaterlineExtractor()
    # Water covers entire vehicle box up to the roof
    mask = np.zeros((500, 500), dtype=np.uint8)
    mask[100:500, :] = 255

    det = VehicleDetection(
        id=2,
        class_name="truck",
        confidence=0.88,
        box_pixel=BoxPixel(x1=100.0, y1=100.0, x2=300.0, y2=300.0),
        box_normalized=BoxNormalized(x1=0.2, y1=0.2, x2=0.6, y2=0.6),
    )

    cands = ext.extract_candidates(mask, [det])
    assert len(cands) == 1
    assert cands[0].can_estimate is False
    assert cands[0].status == "ABSTAIN_UNCLEAR_WATERLINE"
    assert "touches or exceeds upper vehicle bound" in cands[0].reason


def test_waterline_short_contact_length_abstains():
    ext = WaterlineExtractor()
    mask = np.zeros((500, 500), dtype=np.uint8)
    # Only 2 columns have water inside the 200px wide vehicle
    mask[200:300, 150:152] = 255

    det = VehicleDetection(
        id=3,
        class_name="bus",
        confidence=0.85,
        box_pixel=BoxPixel(x1=100.0, y1=100.0, x2=300.0, y2=300.0),
        box_normalized=BoxNormalized(x1=0.2, y1=0.2, x2=0.6, y2=0.6),
    )

    cands = ext.extract_candidates(mask, [det])
    assert len(cands) == 1
    assert cands[0].can_estimate is False
    assert cands[0].status == "ABSTAIN_UNCLEAR_WATERLINE"
    assert "contact length too short" in cands[0].reason


def test_waterline_operator_confirmed():
    ext = WaterlineExtractor()
    mask = np.zeros((500, 500), dtype=np.uint8)

    det = VehicleDetection(
        id=5,
        class_name="car",
        confidence=0.9,
        box_pixel=BoxPixel(x1=100.0, y1=100.0, x2=300.0, y2=300.0),
        box_normalized=BoxNormalized(x1=0.2, y1=0.2, x2=0.6, y2=0.6),
    )

    # Valid operator confirmation within vehicle bounds
    cands = ext.extract_candidates(mask, [det], operator_waterlines={5: 225.0})
    assert len(cands) == 1
    assert cands[0].can_estimate is True
    assert cands[0].status == "OPERATOR_CONFIRMED"
    assert cands[0].y_waterline == 225.0

    # Invalid operator confirmation out of bounds
    cands_bad = ext.extract_candidates(mask, [det], operator_waterlines={5: 50.0})
    assert len(cands_bad) == 1
    assert cands_bad[0].can_estimate is False
    assert cands_bad[0].status == "ABSTAIN_UNCLEAR_WATERLINE"


# ==========================================
# ImageAnnotator Unit Tests
# ==========================================

def test_annotator_preserves_original_image():
    annotator = ImageAnnotator()
    original = np.zeros((300, 400, 3), dtype=np.uint8)
    original_copy = original.copy()

    det = VehicleDetection(
        id=1,
        class_name="car",
        confidence=0.92,
        box_pixel=BoxPixel(x1=50.0, y1=50.0, x2=200.0, y2=200.0),
        box_normalized=BoxNormalized(x1=0.125, y1=0.1667, x2=0.5, y2=0.6667),
    )
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[150:300, :] = 255
    cand = WaterlineCandidate(
        vehicle_id=1,
        y_waterline=150.0,
        y_top=50.0,
        y_base=200.0,
        can_estimate=True,
        status="READY_FOR_ESTIMATE",
    )

    annotated = annotator.annotate(
        image_input=original,
        detections=[det],
        water_mask=mask,
        waterline_candidates=[cand],
        show_legend=True,
    )

    # Output dimensions match input dimensions
    assert annotated.shape == (300, 400, 3)
    # Original array was NOT mutated in place
    assert np.array_equal(original, original_copy)
    # Annotated array has non-zero pixels from bounding boxes, mask, legend
    assert not np.array_equal(annotated, original)


def test_annotator_base64_encoding():
    annotator = ImageAnnotator()
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    b64_str = annotator.to_base64_jpeg(blank)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 50


def test_annotator_visual_styles():
    annotator = ImageAnnotator()
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    det = VehicleDetection(
        id=1,
        class_name="truck",
        confidence=0.88,
        box_pixel=BoxPixel(x1=50.0, y1=50.0, x2=350.0, y2=350.0),
        box_normalized=BoxNormalized(x1=0.125, y1=0.125, x2=0.875, y2=0.875),
    )

    cands = [
        WaterlineCandidate(
            vehicle_id=1,
            y_waterline=150.0,
            y_top=50.0,
            y_base=350.0,
            can_estimate=True,
            status="READY_FOR_ESTIMATE",
        ),
        WaterlineCandidate(
            vehicle_id=1,
            y_waterline=200.0,
            y_top=50.0,
            y_base=350.0,
            can_estimate=True,
            status="OPERATOR_CONFIRMED",
        ),
        WaterlineCandidate(
            vehicle_id=1,
            y_waterline=52.0,
            y_top=50.0,
            y_base=350.0,
            can_estimate=False,
            status="ABSTAIN_UNCLEAR_WATERLINE",
        ),
    ]

    # Renders all three styles without exceptions
    annotated = annotator.annotate(
        image_input=img,
        detections=[det],
        waterline_candidates=cands,
        show_legend=False,
    )
    assert annotated.shape == (400, 400, 3)


# ==========================================
# VisionPipeline Unit Tests (Mocked Components)
# ==========================================

class _MockDetector:
    def __init__(self, detections=None):
        self.detections = detections if detections is not None else []

    def detect(self, image_input):
        return self.detections


class _MockSegmenter:
    def __init__(self, water_result=None, mask=None):
        self.water_result = water_result
        self.mask = mask

    def segment(self, image_input, operator_polygon=None, return_mask=False):
        res = self.water_result or WaterRegionResult(detected=False, method="abstain")
        mask = self.mask if self.mask is not None else np.zeros((300, 400), dtype=np.uint8)
        return (res, mask) if return_mask else res


class _MockWaterlineExtractor:
    def __init__(self, candidates=None):
        self.candidates = candidates if candidates is not None else []

    def extract_candidates(self, mask, detections=None, operator_waterlines=None):
        return self.candidates


def test_pipeline_no_detection():
    pipeline = VisionPipeline(
        detector=_MockDetector([]),
        segmenter=_MockSegmenter(),
        waterline_extractor=_MockWaterlineExtractor([]),
    )
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    res = pipeline.process(img, image_id="test_no_det.jpg")

    assert res.status == "NO_DETECTION"
    assert res.detections == []
    assert res.waterline_candidates == []
    assert res.image_width == 400
    assert res.image_height == 300
    assert res.annotated_image_base64 is not None


def test_pipeline_partial_abstain():
    det = VehicleDetection(
        id=1,
        class_name="car",
        confidence=0.9,
        box_pixel=BoxPixel(x1=50.0, y1=50.0, x2=200.0, y2=200.0),
        box_normalized=BoxNormalized(x1=0.125, y1=0.1667, x2=0.5, y2=0.6667),
    )
    cand_abstained = WaterlineCandidate(
        vehicle_id=1,
        y_waterline=50.0,
        y_top=50.0,
        y_base=200.0,
        can_estimate=False,
        status="ABSTAIN_UNCLEAR_WATERLINE",
        reason="Touches roof",
    )
    pipeline = VisionPipeline(
        detector=_MockDetector([det]),
        segmenter=_MockSegmenter(),
        waterline_extractor=_MockWaterlineExtractor([cand_abstained]),
    )
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    res = pipeline.process(img)

    assert res.status == "PARTIAL_ABSTAIN"
    assert len(res.detections) == 1
    assert res.waterline_candidates[0].can_estimate is False


def test_pipeline_success():
    det = VehicleDetection(
        id=1,
        class_name="car",
        confidence=0.95,
        box_pixel=BoxPixel(x1=50.0, y1=50.0, x2=200.0, y2=200.0),
        box_normalized=BoxNormalized(x1=0.125, y1=0.1667, x2=0.5, y2=0.6667),
    )
    cand_valid = WaterlineCandidate(
        vehicle_id=1,
        y_waterline=120.0,
        y_top=50.0,
        y_base=200.0,
        can_estimate=True,
        status="READY_FOR_ESTIMATE",
    )
    pipeline = VisionPipeline(
        detector=_MockDetector([det]),
        segmenter=_MockSegmenter(water_result=WaterRegionResult(detected=True, area_percentage=30.0, method="classical_cv_heuristic")),
        waterline_extractor=_MockWaterlineExtractor([cand_valid]),
    )
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    res = pipeline.process(img, image_id="test_success.jpg")

    assert res.status == "SUCCESS"
    assert len(res.detections) == 1
    assert res.waterline_candidates[0].can_estimate is True
    assert res.waterline_candidates[0].y_waterline == 120.0
    assert res.annotated_image_base64 is not None


def test_pipeline_error_handling_and_fallback():
    faulty_detector = MagicMock()
    faulty_detector.detect.side_effect = RuntimeError("GPU bus failure / model crashed")

    pipeline = VisionPipeline(
        detector=faulty_detector,
        segmenter=_MockSegmenter(),
        waterline_extractor=_MockWaterlineExtractor(),
        use_fixture_fallback=True,
    )

    # When image_id matches demo fixture, serves fallback fixture
    res_fixture = pipeline.process(
        image_input="assets/demo_images/flooded_vehicle_underpass.jpg",
        image_id="flooded_vehicle_underpass.jpg",
    )
    assert res_fixture.status == "FALLBACK"
    assert len(res_fixture.detections) > 0
    assert "served precomputed demo fixture" in res_fixture.note

    # When fallback is disabled or image is unknown, returns safe empty FALLBACK result
    pipeline_no_fb = VisionPipeline(
        detector=faulty_detector,
        use_fixture_fallback=False,
    )
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res_err = pipeline_no_fb.process(img, image_id="unknown.jpg")
    assert res_err.status == "FALLBACK"
    assert res_err.detections == []
    assert res_err.water_region.detected is False


# ==========================================
# Milestone 6 Integration Tests
# ==========================================


def test_pipeline_operator_override_flow():
    """Verify operator overrides when YOLO weights are available."""

    try:
        detector = VehicleDetector()
    except FileNotFoundError:
        pytest.skip("YOLO11n weights are not available")

    pipeline = VisionPipeline(
        detector=detector,
        segmenter=FloodwaterSegmenter(),
        waterline_extractor=WaterlineExtractor(),
    )

    full_path = Path("assets/demo_images/waterlogged_road_vehicle.jpg")

    if not full_path.is_file():
        pytest.skip("Demo image not found")

    op_poly = [
        (0.0, 150.0),
        (898.0, 150.0),
        (898.0, 684.0),
        (0.0, 684.0),
    ]
    op_waterlines = {1: 180.0}

    res = pipeline.process(
        image_input=full_path,
        operator_polygon=op_poly,
        operator_waterlines=op_waterlines,
        image_id="waterlogged_road_vehicle.jpg",
    )

    assert res.status == "SUCCESS"
    assert res.water_region.method == "operator_polygon"
    assert res.water_region.confidence == 1.0

    cand = res.waterline_candidates[0]
    assert cand.status == "OPERATOR_CONFIRMED"
    assert cand.can_estimate is True
    assert cand.y_waterline == 180.0
    assert res.annotated_image_base64 is not None


def test_pipeline_real_images_end_to_end():
    """Verify end-to-end execution when YOLO weights are available."""

    try:
        pipeline = VisionPipeline()
    except FileNotFoundError:
        pytest.skip("YOLO11n weights are not available")

    images = [
        Path("assets/demo_images/flooded_vehicle_underpass.jpg"),
        Path("assets/demo_images/waterlogged_road_vehicle.jpg"),
        Path("assets/demo_images/underpass_overview.jpg"),
    ]

    if not all(img.is_file() for img in images):
        pytest.skip("One or more demo images are missing")

    for img_path in images:
        res = pipeline.process(
            img_path,
            generate_annotated_image=True,
        )

        assert isinstance(res, VisionPipelineResult)
        assert res.image_width > 0
        assert res.image_height > 0
        assert res.status in (
            "SUCCESS",
            "PARTIAL_ABSTAIN",
            "NO_DETECTION",
        )
        assert len(res.detections) > 0
        assert res.water_region.detected is True
        assert res.annotated_image_base64 is not None
        assert len(res.annotated_image_base64) > 1000





