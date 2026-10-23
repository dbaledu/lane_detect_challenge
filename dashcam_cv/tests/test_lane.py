"""Unit tests for lane detection and tracking modules."""
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.geometry import get_roi_polygon, evaluate_poly_x
from src.lane.lane_detector import LaneDetector
from src.lane.lane_tracker import LaneTracker
from src.lane.lane_change import LaneChangeDetector


class TestLaneModule(unittest.TestCase):
    """Test suite for lane perception components."""

    def setUp(self):
        self.w = 960
        self.h = 540
        self.cfg = {
            "roi": {
                "top_ratio": 0.65,
                "bottom_ratio": 0.88,
                "top_left_ratio": 0.44,
                "top_right_ratio": 0.56,
                "bottom_left_ratio": 0.18,
                "bottom_right_ratio": 0.82
            },
            "lane": {
                "hough_threshold": 20,
                "min_line_length": 15,
                "max_line_gap": 30,
                "min_slope": 0.35,
                "max_slope": 3.0,
                "smoothing_alpha": 0.20,
                "max_lost_frames": 5
            },
            "lane_change": {
                "drift_threshold_px": 25.0,
                "candidate_frames": 2,
                "progress_frames": 4
            }
        }

    def test_roi_polygon_dimensions(self):
        """Verifies trapezoidal ROI generation."""
        polygon = get_roi_polygon(self.w, self.h, self.cfg["roi"])
        self.assertEqual(polygon.shape, (1, 4, 2))
        top_y = int(self.h * 0.65)
        bot_y = int(self.h * 0.88)
        self.assertEqual(polygon[0][0][1], top_y)
        self.assertEqual(polygon[0][2][1], bot_y)

    def test_lane_detection_on_synthetic_frame(self):
        """Tests Hough detection and polynomial fitting on synthetic lane lines."""
        # Create black edge image
        edges = np.zeros((self.h, self.w), dtype=np.uint8)
        top_y = int(self.h * 0.65)
        bot_y = int(self.h * 0.88)

        # Draw left lane (slope < 0)
        cv2.line(edges, (int(self.w * 0.25), bot_y), (int(self.w * 0.46), top_y), 255, 3)
        # Draw right lane (slope > 0)
        cv2.line(edges, (int(self.w * 0.75), bot_y), (int(self.w * 0.54), top_y), 255, 3)

        detector = LaneDetector(self.cfg)
        polygon = get_roi_polygon(self.w, self.h, self.cfg["roi"])
        lines, roi_edges = detector.detect_lines(edges, polygon)
        self.assertIsNotNone(lines)

        left_fit, right_fit, conf = detector.filter_and_fit(lines, self.w, self.h)
        self.assertIsNotNone(left_fit)
        self.assertIsNotNone(right_fit)
        self.assertGreater(conf, 0.4)

    def test_lane_tracker_occlusion_and_loss(self):
        """Tests temporal smoothing, opposite lane estimation, and eventual lane loss."""
        tracker = LaneTracker(self.cfg)

        # 1. Provide both lanes
        res1 = tracker.update([-0.5, 500], [0.5, 450], 0.9, self.w, self.h)
        self.assertEqual(res1.status, "BOTH_DETECTED")
        self.assertIsNotNone(res1.lane_center_x)

        # 2. Left lane only (right occluded)
        res2 = tracker.update([-0.5, 500], None, 0.5, self.w, self.h)
        self.assertEqual(res2.status, "LEFT_ONLY")
        self.assertIsNotNone(res2.right_line)  # Synthesized from learned width

        # 3. Both missing for 10 consecutive frames -> should transition to LOST
        for _ in range(7):
            res_lost = tracker.update(None, None, 0.0, self.w, self.h)

        self.assertEqual(res_lost.status, "LOST")
        self.assertIsNone(res_lost.left_line)
        self.assertIsNone(res_lost.right_line)

    def test_cv1021_dual_roi_geometry(self):
        """Tests cv1021.py dual ROI geometry and bounding box shrinking."""
        detector = LaneDetector(self.cfg)
        (l_corners, l_bbox), (r_corners, r_bbox) = detector.get_dual_roi_geometry(self.h, self.w)
        self.assertEqual(l_corners.shape, (1, 4, 2))
        self.assertEqual(r_corners.shape, (1, 4, 2))
        self.assertEqual(len(l_bbox), 4)
        self.assertEqual(len(r_bbox), 4)
        # Left bbox should be to the left of right bbox
        self.assertLess(l_bbox[2], r_bbox[0])

    def test_cv1021_kalman_filter_innovation_gating(self):
        """Tests SingleLaneKalmanFilter innovation gating against sudden jumps."""
        from src.lane.lane_tracker import SingleLaneKalmanFilter
        kf = SingleLaneKalmanFilter(dt=1.0, max_lost_frames=10)
        # Initial measurement
        fit, lost = kf.update([-0.5, 400.0])
        self.assertEqual(lost, 0)
        self.assertAlmostEqual(fit[0], -0.5, places=2)

        # Normal small change
        fit, lost = kf.update([-0.51, 402.0])
        self.assertEqual(lost, 0)

        # Extreme jump (diff_c > 160 or diff_m > 0.45) -> should be gated out
        fit, lost = kf.update([0.8, 700.0])
        self.assertEqual(lost, 1)  # Gated: lost_frames incremented
        self.assertAlmostEqual(fit[0], -0.51, delta=0.1)  # Maintained previous prediction

    def test_cv1021_unpack_compatibility(self):
        """Tests unpacking LaneResult as (left_fit, right_fit, lost_frames) like cv1021.py."""
        tracker = LaneTracker(self.cfg)
        smooth_l, smooth_r, lost_cnt = tracker.update([-0.5, 400], [0.5, 500], 1.0, self.w, self.h)
        self.assertIsNotNone(smooth_l)
        self.assertIsNotNone(smooth_r)
        self.assertEqual(lost_cnt, 0)


if __name__ == "__main__":
    unittest.main()

