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

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # Feature detector parameters
        self.feature_params = dict(
            maxCorners=self.max_features,
            qualityLevel=0.02,
            minDistance=10,
            blockSize=7
        )

    def estimate(
        self,
        gray: np.ndarray,
        detected_objects: Optional[List[ObjectBox]] = None
    ) -> MotionResult:
        """Computes background optical flow displacement excluding foreground objects."""
        h, w = gray.shape[:2]

        # Mask out foreground objects and top sky / bottom hood
        mask = np.ones((h, w), dtype=np.uint8) * 255
        # Mask out upper sky (0 ~ 0.35h) and bottom hood (0.90 ~ 1.0h)
        mask[:int(h * 0.35), :] = 0
        mask[int(h * 0.90):, :] = 0

        if detected_objects:
            for obj in detected_objects:
                x1, y1, x2, y2 = obj.bbox
                # Expand box slightly to exclude object edges
                pad = 8
                bx1 = max(0, x1 - pad)
                by1 = max(0, y1 - pad)
                bx2 = min(w, x2 + pad)
                by2 = min(h, y2 + pad)
                mask[by1:by2, bx1:bx2] = 0

        # If first frame or need new features
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 25:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            self.prev_gray = gray.copy()
            return MotionResult(magnitude=0.0, median_dx=0.0, median_dy=0.0, tracked_points=0, is_reliable=True)

        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        if curr_pts is None or status is None:
            self.prev_gray = gray.copy()
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            return MotionResult(magnitude=0.0, median_dx=0.0, median_dy=0.0, tracked_points=0, is_reliable=False)

        valid_idx = (status.flatten() == 1)
        good_prev = self.prev_pts[valid_idx]
        good_curr = curr_pts[valid_idx]

        if len(good_curr) < 10:
            self.prev_gray = gray.copy()
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            return MotionResult(magnitude=0.0, median_dx=0.0, median_dy=0.0, tracked_points=len(good_curr), is_reliable=False)

        # Compute displacements
        dx = good_curr[:, 0, 0] - good_prev[:, 0, 0]
        dy = good_curr[:, 0, 1] - good_prev[:, 0, 1]
        mags = np.hypot(dx, dy)

        # Median is robust against rogue points and specular reflection
        med_dx = float(np.median(dx))
        med_dy = float(np.median(dy))
        med_mag = float(np.median(mags))

        # Re-detect new features periodically to avoid tracking degradation
        if len(good_curr) < (self.max_features * 0.6):
            new_features = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            self.prev_pts = new_features
        else:
            self.prev_pts = good_curr.reshape(-1, 1, 2)

        self.prev_gray = gray.copy()

        return MotionResult(
            magnitude=med_mag,
            median_dx=med_dx,
            median_dy=med_dy,
            tracked_points=len(good_curr),
            is_reliable=True
        )
