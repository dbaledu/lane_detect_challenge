"""Motion estimation module using sparse optical flow and robust median statistics."""
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from ..common.data_models import MotionResult, ObjectBox


class MotionEstimator:
    """Estimates ego-vehicle apparent camera motion via background optical flow."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        st_cfg = self.cfg.get("state", {})
        self.max_features = int(st_cfg.get("flow_max_features", 150))

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.frame_count: int = 0
        self.reseed_interval: int = int(st_cfg.get("flow_reseed_interval", 20))

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # Feature detector parameters (blockSize 3 for ~2x faster corner response)
        self.feature_params = dict(
            maxCorners=self.max_features,
            qualityLevel=0.02,
            minDistance=10,
            blockSize=3
        )

    def _create_feature_mask(self, h: int, w: int, detected_objects: Optional[List[ObjectBox]]) -> np.ndarray:
        """Lazily generates feature mask only when new Shi-Tomasi corners need extraction."""
        mask = np.zeros((h, w), dtype=np.uint8)
        # Exclude upper sky (0 ~ 0.35h) and bottom hood (0.90 ~ 1.0h)
        mask[int(h * 0.35):int(h * 0.90), :] = 255

        if detected_objects:
            pad = 8
            for obj in detected_objects:
                x1, y1, x2, y2 = obj.bbox
                bx1 = max(0, x1 - pad)
                by1 = max(0, y1 - pad)
                bx2 = min(w, x2 + pad)
                by2 = min(h, y2 + pad)
                mask[by1:by2, bx1:bx2] = 0
        return mask

    def estimate(
        self,
        gray: np.ndarray,
        detected_objects: Optional[List[ObjectBox]] = None
    ) -> MotionResult:
        """Computes background optical flow displacement excluding foreground objects."""
        h, w = gray.shape[:2]
        self.frame_count += 1

        # 1. Initialize or replenish features when points are critically depleted
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 25:
            mask = self._create_feature_mask(h, w, detected_objects)
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            self.prev_gray = gray
            return MotionResult(magnitude=0.0, median_dx=0.0, median_dy=0.0, tracked_points=0, is_reliable=True)

        # 2. Pure PyrLK optical flow tracking without full-mask allocation overhead
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        if curr_pts is None or status is None:
            self.prev_gray = gray
            self.prev_pts = None
            return MotionResult(magnitude=0.0, median_dx=0.0, median_dy=0.0, tracked_points=0, is_reliable=False)

        valid_idx = (status.ravel() == 1)
        good_prev = self.prev_pts[valid_idx]
        good_curr = curr_pts[valid_idx]

        # 3. [Foreground Filtering] Exclude points currently tracked on moving vehicles/pedestrians
        if detected_objects and len(good_curr) > 0:
            pts_xy = good_curr[:, 0, :]
            bg_mask = np.ones(len(good_curr), dtype=bool)
            for obj in detected_objects:
                ox1, oy1, ox2, oy2 = obj.bbox
                in_box = (
                    (pts_xy[:, 0] >= ox1) & (pts_xy[:, 0] <= ox2) &
                    (pts_xy[:, 1] >= oy1) & (pts_xy[:, 1] <= oy2)
                )
                bg_mask[in_box] = False
            good_prev = good_prev[bg_mask]
            good_curr = good_curr[bg_mask]

        if len(good_curr) < 10:
            self.prev_gray = gray
            self.prev_pts = None
            return MotionResult(magnitude=0.0, median_dx=0.0, median_dy=0.0, tracked_points=len(good_curr), is_reliable=False)

        # 4. Vectorized displacement calculation without temporary slicing copies
        diff = good_curr[:, 0, :] - good_prev[:, 0, :]
        dx = diff[:, 0]
        dy = diff[:, 1]
        mags = np.hypot(dx, dy)

        # Median statistics against rogue outliers
        med_dx = float(np.median(dx))
        med_dy = float(np.median(dy))
        med_mag = float(np.median(mags))

        # 5. Periodic replenishment gated by frame interval to avoid infinite re-detection loop
        if len(good_curr) < int(self.max_features * 0.45) and (self.frame_count % self.reseed_interval == 0):
            mask = self._create_feature_mask(h, w, detected_objects)
            new_features = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            self.prev_pts = new_features if new_features is not None else good_curr.reshape(-1, 1, 2)
        else:
            self.prev_pts = good_curr.reshape(-1, 1, 2)

        self.prev_gray = gray

        return MotionResult(
            magnitude=med_mag,
            median_dx=med_dx,
            median_dy=med_dy,
            tracked_points=len(good_curr),
            is_reliable=True
        )
