"""Lane detection module based on ROI filtering, Hough transform, and robust fitting."""
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from ..common.geometry import get_roi_polygon, compute_slope, evaluate_poly_x


class LaneDetector:
    """Detects left and right lane candidates in road video frames."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        lane_cfg = self.cfg.get("lane", {})

        self.hough_threshold = int(lane_cfg.get("hough_threshold", 35))
        self.min_line_length = int(lane_cfg.get("min_line_length", 25))
        self.max_line_gap = int(lane_cfg.get("max_line_gap", 35))
        self.min_slope = float(lane_cfg.get("min_slope", 0.45))
        self.max_slope = float(lane_cfg.get("max_slope", 3.0))

    def detect_lines(self, edges: np.ndarray, roi_corners: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Applies ROI mask and executes HoughLinesP."""
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_corners, 255)
        roi_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        return lines, roi_edges

    def filter_and_fit(
        self,
        lines: Optional[np.ndarray],
        w: int,
        h: int
    ) -> Tuple[Optional[List[float]], Optional[List[float]], float]:
        """Filters line segments by slope and lateral position, then fits 1st order polynomials.

        Returns:
            left_fit: [m, c] for x = m*y + c, or None
            right_fit: [m, c] for x = m*y + c, or None
            confidence: detection confidence (0.0 ~ 1.0)
        """
        if lines is None or len(lines) == 0:
            return None, None, 0.0

        left_x, left_y = [], []
        right_x, right_y = [], []
        center_x = w * 0.5
        margin_x = w * 0.08

        total_left_len = 0.0
        total_right_len = 0.0

        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            slope = compute_slope(x1, y1, x2, y2)
            if slope is None:
                continue

            abs_slope = abs(slope)
            # Filter out near-horizontal (crosswalks/stoplines) and near-vertical noise
            if abs_slope < self.min_slope or abs_slope > self.max_slope:
                continue

            length = np.hypot(x2 - x1, y2 - y1)

            # Left lane: negative slope in image coordinates, on the left side
            if slope < 0 and max(x1, x2) < center_x + margin_x:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
                total_left_len += length

            # Right lane: positive slope, on the right side
            elif slope > 0 and min(x1, x2) > center_x - margin_x:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])
                total_right_len += length

        left_fit = None
        right_fit = None

        if len(left_x) >= 4:
            try:
                left_fit = list(np.polyfit(left_y, left_x, 1))
            except Exception:
                left_fit = None

        if len(right_x) >= 4:
            try:
                right_fit = list(np.polyfit(right_y, right_x, 1))
            except Exception:
                right_fit = None

        # Compute confidence metric
        conf = 0.0
        if left_fit is not None and right_fit is not None:
            # Check parallel consistency (slopes should point towards vanishing point)
            conf = min(1.0, 0.5 + (total_left_len + total_right_len) / 400.0)
        elif left_fit is not None or right_fit is not None:
            one_len = total_left_len if left_fit is not None else total_right_len
            conf = min(0.65, 0.25 + one_len / 400.0)

        return left_fit, right_fit, float(conf)
