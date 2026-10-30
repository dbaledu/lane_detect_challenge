"""Road markings detection module for crosswalks and stop lines."""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ..common.data_models import RoadMarkingResult
from ..common.geometry import get_roi_polygon, compute_slope


class RoadMarkingDetector:
    """Detects crosswalk zebra stripes and horizontal stop lines using computer vision."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        rm_cfg = self.cfg.get("road_markings", {})

        self.enabled = rm_cfg.get("enable", True)
        self.min_stripes = int(rm_cfg.get("crosswalk_min_stripes", 3))
        self.stopline_min_width_ratio = float(rm_cfg.get("stopline_min_width_ratio", 0.25))
        self.stopline_max_slope = float(rm_cfg.get("stopline_max_slope", 0.20))

    def detect(self, frame: np.ndarray, gray: np.ndarray, edges: np.ndarray,
               detect_crosswalk: bool = True) -> RoadMarkingResult:
        """Detects crosswalks and stop lines in the road area."""
        if not self.enabled:
            return RoadMarkingResult()

        h, w = frame.shape[:2]
        roi_corners = get_roi_polygon(w, h, self.cfg.get("roi"))

        # Restrict markings analysis to the road area
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, roi_corners, 255)
        road_gray = cv2.bitwise_and(gray, mask)

        if detect_crosswalk:
            crosswalk_detected, crosswalk_boxes = self._detect_crosswalk(road_gray, w, h)
        else:
            crosswalk_detected, crosswalk_boxes = False, []
        stopline_detected, stopline_boxes, stopline_y = self._detect_stopline(edges, mask, w, h)

        return RoadMarkingResult(
            crosswalk_detected=crosswalk_detected,
            crosswalk_boxes=crosswalk_boxes,
            stopline_detected=stopline_detected,
            stopline_boxes=stopline_boxes,
            stopline_y=stopline_y
        )

    def _detect_crosswalk(self, road_gray: np.ndarray, w: int, h: int) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
        """Detects zebra crosswalk pattern via thresholding and spatial clustering."""
        # Focus on bright markings in road
        _, thresh = cv2.threshold(road_gray, 180, 255, cv2.THRESH_BINARY)
        # Use a horizontal structuring element to emphasize zebra stripes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
        filtered = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stripe_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150 or area > 15000:
                continue
            x, y, rw, rh = cv2.boundingRect(cnt)
            aspect = rw / float(rh)
            # Zebra stripe candidate: wider than tall
            if 1.5 <= aspect <= 12.0 and rw > 20:
                stripe_boxes.append((x, y, x + rw, y + rh))

        if len(stripe_boxes) >= self.min_stripes:
            # Cluster stripe boxes that are vertically close
            stripe_boxes.sort(key=lambda b: b[1])
            min_x = min(b[0] for b in stripe_boxes)
            min_y = min(b[1] for b in stripe_boxes)
            max_x = max(b[2] for b in stripe_boxes)
            max_y = max(b[3] for b in stripe_boxes)

            # Combined bounding box enclosing the crosswalk
            cluster_box = (min_x, min_y, max_x, max_y)
            return True, [cluster_box]

        return False, []

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
            return False, [], None

        candidate_boxes = []
        found_y = None

        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            actual_y1 = roi_mid_y + y1
            actual_y2 = roi_mid_y + y2
            slope = compute_slope(x1, actual_y1, x2, actual_y2)

            if slope is not None and abs(slope) <= self.stopline_max_slope:
                line_w = abs(x2 - x1)
                if line_w >= (w * self.stopline_min_width_ratio):
                    bx1 = min(x1, x2)
                    bx2 = max(x1, x2)
                    by1 = min(actual_y1, actual_y2) - 4
                    by2 = max(actual_y1, actual_y2) + 4
                    candidate_boxes.append((bx1, by1, bx2, by2))
                    found_y = (actual_y1 + actual_y2) / 2.0
                    break

        if candidate_boxes:
            return True, candidate_boxes, found_y

        return False, [], None
