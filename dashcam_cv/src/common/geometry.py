"""Geometry and spatial computation utilities for autonomous vision."""
import numpy as np
from typing import Tuple, Optional, List, Dict, Any


def get_roi_polygon(w: int, h: int, roi_cfg: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Computes optimized trapezoidal ROI corners to block OSD and hood noise.

    Default ratios based on Test_Video_2 analysis:
    - Top: 0.65h (blocks front vehicle bumper and pedestrian bodies)
    - Bottom: 0.88h (blocks dashboard reflection and blinking OSD speed text)
    - Left/Right: 0.18w ~ 0.82w (blocks lens distortion and sidewalk noise)
    """
    if roi_cfg is None:
        roi_cfg = {}

    top_ratio = float(roi_cfg.get("top_ratio", 0.65))
    bot_ratio = float(roi_cfg.get("bottom_ratio", 0.88))
    tl_ratio = float(roi_cfg.get("top_left_ratio", 0.44))
    tr_ratio = float(roi_cfg.get("top_right_ratio", 0.56))
    bl_ratio = float(roi_cfg.get("bottom_left_ratio", 0.18))
    br_ratio = float(roi_cfg.get("bottom_right_ratio", 0.82))

    top_y = int(h * top_ratio)
    bot_y = int(h * bot_ratio)

    corners = np.array([[
        (int(w * tl_ratio), top_y),
        (int(w * tr_ratio), top_y),
        (int(w * br_ratio), bot_y),
        (int(w * bl_ratio), bot_y)
    ]], dtype=np.int32)

    return corners


def evaluate_poly_x(fit: Optional[List[float]], y: float) -> Optional[int]:
    """Calculates x coordinate given y for a 1st or 2nd order polynomial fit."""
    if fit is None or len(fit) == 0:
        return None
    try:
        val = np.polyval(fit, y)
        if np.isnan(val) or np.isinf(val):
            return None
        return int(round(val))
    except Exception:
        return None


def compute_slope(x1: float, y1: float, x2: float, y2: float) -> Optional[float]:
    """Computes slope dy / dx with division by zero protection."""
    dx = x2 - x1
    if abs(dx) < 1e-4:
        return None
    return (y2 - y1) / dx


def calculate_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area
