"""Reference-height depth calculation for a human-reviewed demo image.

This is a simple projective approximation, not a universal monocular-depth model.
The reference and waterline should be measured at the same image location and
per-camera/site calibration is required before operational use.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DepthEstimate:
    estimated_depth_cm: float
    submerged_ratio: float
    reference_height_cm: float
    method: str = "reference_height_ratio_demo"
    note: str = (
        "Illustrative estimate from operator-supplied image coordinates; "
        "not validated safety guidance."
    )


def estimate_depth_cm(
    *,
    reference_height_cm: float,
    y_top: float,
    y_base: float,
    y_waterline: float,
) -> DepthEstimate:
    """Estimate submerged vertical extent from image y-coordinates.

    Image y coordinates increase downward. The top and base must bound the
    reference object, and the waterline must intersect between them.
    """
    values = (reference_height_cm, y_top, y_base, y_waterline)
    if any(not isinstance(v, (int, float)) for v in values):
        raise ValueError("All values must be numeric.")
    if reference_height_cm <= 0:
        raise ValueError("reference_height_cm must be greater than zero.")
    if y_base <= y_top:
        raise ValueError("y_base must be greater than y_top in image coordinates.")
    if not (y_top < y_waterline < y_base):
        raise ValueError(
            "Waterline does not intersect the reference object; abstain/review instead."
        )
    ratio = (y_base - y_waterline) / (y_base - y_top)
    depth = reference_height_cm * ratio
    return DepthEstimate(
        estimated_depth_cm=round(depth, 1),
        submerged_ratio=round(ratio, 4),
        reference_height_cm=round(float(reference_height_cm), 1),
    )
