# AquaGauge Computer Vision Module — Integration Guide

> **Target Audience:** Person 1 (Tech Lead / Integrator) and Person 4 (Backend API)  
> **Author:** Person 2 (Computer Vision)  
> **Module Location:** `backend/app/vision/`  
> **Module Status:** Verified and Tested on CPU (45/45 tests passing)

---

## 1. Quick Start & Basic Usage

```python
from backend.app.vision import VisionPipeline

# 1. Initialize once at application startup (caches YOLO model in memory)
pipeline = VisionPipeline()

# 2. Process an image (accepts filepath, PIL Image, or numpy BGR array)
result = pipeline.process(
    image_input="assets/demo_images/waterlogged_road_vehicle.jpg",
    image_id="waterlogged_road_vehicle.jpg",
    generate_annotated_image=True,
)

# 3. Access structured outputs
print(f"Status: {result.status}")              # 'SUCCESS', 'PARTIAL_ABSTAIN', 'NO_DETECTION', or 'FALLBACK'
print(f"Vehicles: {len(result.detections)}")   # List of VehicleDetection objects
print(f"Water Area: {result.water_region.area_percentage}%")
print(f"Waterlines: {len(result.waterline_candidates)}")
```

---

## 2. Public API Interface

All primary interfaces are exported directly from `backend.app.vision`:

```python
from backend.app.vision import (
    VisionPipeline,
    VehicleDetector,
    FloodwaterSegmenter,
    WaterlineExtractor,
    ImageAnnotator,
    VisionPipelineResult,
    VehicleDetection,
    WaterRegionResult,
    WaterlineCandidate,
    BoxPixel,
    BoxNormalized,
    get_demo_fixture,
    DEMO_FIXTURES,
)
```

### Constructor Signature: `VisionPipeline`

```python
class VisionPipeline:
    def __init__(
        self,
        detector: Optional[VehicleDetector] = None,
        segmenter: Optional[FloodwaterSegmenter] = None,
        waterline_extractor: Optional[WaterlineExtractor] = None,
        annotator: Optional[ImageAnnotator] = None,
        use_fixture_fallback: bool = False,
    ): ...
```
- By default (`pipeline = VisionPipeline()`), it instantiates all four components on CPU with zero extra setup.
- If `use_fixture_fallback=True`, any unrecoverable model or image decode exception will automatically fall back to the precomputed demo fixture matching `image_id` with `status="FALLBACK"`.

### Process Signature: `pipeline.process()`

```python
def process(
    self,
    image_input: Union[str, Path, Image.Image, np.ndarray],
    operator_polygon: Optional[Union[List[Tuple[float, float]], List[List[Tuple[float, float]]]]] = None,
    operator_waterlines: Optional[Dict[int, float]] = None,
    image_id: Optional[str] = None,
    generate_annotated_image: bool = True,
) -> VisionPipelineResult: ...
```

---

## 3. Data Structures (`VisionPipelineResult`)

The pipeline returns a Pydantic model (`VisionPipelineResult`) conforming to the project schema:

```json
{
  "status": "SUCCESS",
  "image_id": "waterlogged_road_vehicle.jpg",
  "image_width": 898,
  "image_height": 684,
  "detections": [
    {
      "id": 1,
      "class_name": "car",
      "confidence": 0.919,
      "box_pixel": { "x1": 477.8, "y1": 56.5, "x2": 854.1, "y2": 293.5 },
      "box_normalized": { "x1": 0.5321, "y1": 0.0826, "x2": 0.9511, "y2": 0.4291 }
    }
  ],
  "water_region": {
    "detected": true,
    "confidence": 0.67,
    "area_percentage": 26.32,
    "polygon_points": [ [ [x1, y1], [x2, y2], ... ] ],
    "method": "classical_cv_heuristic",
    "note": "Candidate floodwater surface detected using multi-cue color and texture analysis."
  },
  "waterline_candidates": [
    {
      "vehicle_id": 1,
      "y_waterline": 171.0,
      "y_base": 293.5,
      "y_top": 56.5,
      "can_estimate": true,
      "status": "READY_FOR_ESTIMATE",
      "reason": "Candidate waterline geometrically valid for physical depth estimation."
    }
  ],
  "annotated_image_base64": "<base64_encoded_jpeg_string>",
  "note": "Automated visual detections; physical water depth requires reference geometry calibration."
}
```

---

## 4. How to Bridge Vision Outputs to the Depth Estimator

To calculate estimated physical water depth, pass the candidate's pixel coordinates and an operator-supplied physical reference scale to the existing `backend.app.depth_estimator:estimate_depth_cm`:

```python
from backend.app.depth_estimator import estimate_depth_cm
from backend.app.policy import classify_vehicle_risk, classify_road_status

# Iterate over candidate waterlines
for cand in result.waterline_candidates:
    if cand.can_estimate:
        # Standard light vehicle wheel diameter reference: 60.0 cm
        reference_height_cm = 60.0
        
        # Pure projective depth calculation (core contract)
        depth = estimate_depth_cm(
            reference_height_cm=reference_height_cm,
            y_top=cand.y_top,
            y_base=cand.y_base,
            y_waterline=cand.y_waterline,
        )
        
        # Policy risk classification
        vehicle_risk = classify_vehicle_risk(depth.estimated_depth_cm, "light_vehicle")
        road_status = classify_road_status(depth.estimated_depth_cm)
        
        print(f"Submerged Ratio: {depth.submerged_ratio:.1%}")
        print(f"Estimated Depth: {depth.estimated_depth_cm} cm")
        print(f"Vehicle Risk: {vehicle_risk} | Road Status: {road_status}")
    else:
        # ABSTAIN / REVIEW handling
        print(f"Abstaining on vehicle {cand.vehicle_id}: {cand.reason}")
```

---

## 5. Pipeline Statuses & Abstention Logic

| Status | Meaning | Recommended Action |
| :--- | :--- | :--- |
| **`SUCCESS`** | At least one vehicle has a geometrically valid waterline candidate (`y_top < y_waterline < y_base`). | Proceed to depth estimation using `cand.y_waterline` and operator reference. |
| **`PARTIAL_ABSTAIN`** | Vehicles or water were detected, but geometric validation failed (e.g., candidate waterline touches roof bound $y_{\text{top}}$, vehicle is dry, or contact span is $<15\%$). | Surface warning in UI: prompt operator to adjust waterline or supply reference manually. |
| **`NO_DETECTION`** | No vehicles detected in the frame. | Prompt operator to select a custom reference object or manual polygon. |
| **`FALLBACK`** | Model failure, image decode error, or corrupted input. | Serves deterministic precomputed fixture if enabled, or returns graceful empty response. |

---

## 6. Human-in-the-Loop Operator Overrides

The pipeline supports direct operator intervention for both water masks and waterlines:

```python
# 1. Custom operator polygon (drawn on UI canvas)
res = pipeline.process(
    image_input="assets/demo_images/waterlogged_road_vehicle.jpg",
    operator_polygon=[(0, 150), (898, 150), (898, 684), (0, 684)],
)
# Result: water_region.method="operator_polygon", confidence=1.0

# 2. Custom operator waterline coordinate (dragged on UI canvas)
res = pipeline.process(
    image_input="assets/demo_images/waterlogged_road_vehicle.jpg",
    operator_waterlines={1: 180.0},  # maps vehicle_id -> y_pixel
)
# Result: candidate.status="OPERATOR_CONFIRMED", can_estimate=True
```

---

## 7. Performance & Resource Profiles (CPU)

- **Execution Environment:** Pure CPU (`torch-cpu`, OpenCV). No CUDA/GPU required.
- **Model Storage:** Local weights `yolo11n.pt` (~5.4 MB) stored in project root.
- **Cold start (first inference):** ~4.5 seconds (one-time PyTorch CPU graph initialization).
- **Warm steady-state throughput:**
  - $898 \times 684$ image: **~370 ms**
  - $1598 \times 770$ image: **~480 ms**
  - $2880 \times 1620$ image (4K/wide): **~1.8 s**
- **Annotated Image Output:** Base64-encoded JPEG (quality=85), typically 150 KB – 600 KB, ready for direct `<img src="data:image/jpeg;base64,...">` rendering in the frontend dashboard.
