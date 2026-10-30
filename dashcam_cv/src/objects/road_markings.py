"""Road markings detection module for stop lines."""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
from ..common.data_models import RoadMarkingResult
from ..common.geometry import get_roi_polygon, compute_slope


class RoadMarkingDetector:
    """Detects horizontal stop lines using computer vision."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        rm_cfg = self.cfg.get("road_markings", {})

        self.enabled = rm_cfg.get("enable", True)
        self.stopline_min_width_ratio = float(rm_cfg.get("stopline_min_width_ratio", 0.25))
        self.stopline_max_slope = float(rm_cfg.get("stopline_max_slope", 0.20))
        self.stopline_max_jump_ratio = float(rm_cfg.get("stopline_max_jump_ratio", 0.035))
        self.stopline_smooth_frames = max(1, int(rm_cfg.get("stopline_smooth_frames", 5)))
        self.stopline_hold_frames = max(0, int(rm_cfg.get("stopline_hold_frames", 4)))
        self.stopline_max_x_jump_ratio = float(rm_cfg.get("stopline_max_x_jump_ratio", 0.08))
        self.stopline_box_history: deque = deque(maxlen=self.stopline_smooth_frames)

        self.stopline_history: deque = deque(maxlen=self.stopline_smooth_frames)
        self.last_stopline_y: Optional[float] = None
        self.last_stopline_box: Optional[Tuple[int, int, int, int]] = None
        self.stopline_missing_frames = 0

    def detect(self, frame: np.ndarray, gray: np.ndarray, edges: np.ndarray) -> RoadMarkingResult:
        """Detects stop lines in the road area."""
        if not self.enabled:
            return RoadMarkingResult()

        h, w = frame.shape[:2]
        roi_corners = get_roi_polygon(w, h, self.cfg.get("roi"))

        # Restrict markings analysis to the road area
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, roi_corners, 255)
        stopline_detected, stopline_boxes, stopline_y = self._detect_stopline(edges, mask, w, h)

        return RoadMarkingResult(
            stopline_detected=stopline_detected,
            stopline_boxes=stopline_boxes,
            stopline_y=stopline_y
        )


    def _detect_stopline(
        self,
        edges: np.ndarray,
        roi_mask: np.ndarray,
        w: int,
        h: int
    ) -> Tuple[bool, List[Tuple[int, int, int, int]], Optional[float]]:
        """Detects wide horizontal line segments characteristic of road stop lines."""
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        # Focus on lower half of road ROI (near ego vehicle)
        roi_bottom_y = int(h * 0.88)
        roi_mid_y = int(h * 0.70)

        lines = cv2.HoughLinesP(
            masked_edges[roi_mid_y:roi_bottom_y, :],
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=int(w * self.stopline_min_width_ratio * 0.5),
            maxLineGap=20
        )

        if lines is None:
            return self._hold_last_stopline()

        candidates = []
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            actual_y1 = roi_mid_y + y1
            actual_y2 = roi_mid_y + y2
            slope = compute_slope(x1, actual_y1, x2, actual_y2)

            if slope is None or abs(slope) > self.stopline_max_slope:
                continue

            line_w = abs(x2 - x1)
            if line_w < (w * self.stopline_min_width_ratio):
                continue

            found_y = float((actual_y1 + actual_y2) / 2.0)
            bx1 = min(x1, x2)
            bx2 = max(x1, x2)
            box = (bx1, int(found_y - 4), bx2, int(found_y + 4))
            candidates.append((found_y, box, float(line_w)))

        if not candidates:
            return self._hold_last_stopline()

        max_y_jump = max(12, int(h * self.stopline_max_jump_ratio))
        max_x_jump = max(16, int(w * self.stopline_max_x_jump_ratio))
        if self.last_stopline_y is not None and self.last_stopline_box is not None:
            last_center_x = (self.last_stopline_box[0] + self.last_stopline_box[2]) / 2.0
            usable = []
            for c in candidates:
                found_y_candidate, candidate_box, _ = c
                candidate_center_x = (candidate_box[0] + candidate_box[2]) / 2.0
                if (abs(found_y_candidate - self.last_stopline_y) <= max_y_jump and
                        abs(candidate_center_x - last_center_x) <= max_x_jump):
                    usable.append(c)
            if not usable:
                return self._hold_last_stopline()
        else:
            usable = candidates

        found_y, box, _ = min(
            usable,
            key=lambda c: (
                abs(c[0] - self.last_stopline_y) if self.last_stopline_y is not None else 0.0,
                abs((c[1][0] + c[1][2]) / 2.0 - last_center_x)
                if self.last_stopline_y is not None and self.last_stopline_box is not None else 0.0,
                -c[2],
            ),
        )

        self.stopline_history.append(found_y)
        self.stopline_box_history.append((float(box[0]), float(box[2])))

        smooth_y = float(np.median(np.asarray(self.stopline_history, dtype=np.float32)))
        box_x_values = np.asarray(self.stopline_box_history, dtype=np.float32)
        smooth_x1 = int(round(float(np.median(box_x_values[:, 0]))))
        smooth_x2 = int(round(float(np.median(box_x_values[:, 1]))))
        if smooth_x2 <= smooth_x1:
            smooth_x1, smooth_x2 = box[0], box[2]

        smooth_box = (smooth_x1, int(round(smooth_y - 4)), smooth_x2, int(round(smooth_y + 4)))

        self.last_stopline_y = smooth_y
        self.last_stopline_box = smooth_box
        self.stopline_missing_frames = 0
        return True, [smooth_box], smooth_y

    def _hold_last_stopline(self) -> Tuple[bool, List[Tuple[int, int, int, int]], Optional[float]]:
        if self.last_stopline_y is None or self.last_stopline_box is None:
            return False, [], None

        if self.stopline_missing_frames >= self.stopline_hold_frames:
            self.stopline_history.clear()
            self.stopline_box_history.clear()
            self.last_stopline_y = None
            self.last_stopline_box = None
            self.stopline_missing_frames = 0
            return False, [], None

        self.stopline_missing_frames += 1
        return True, [self.last_stopline_box], self.last_stopline_y
