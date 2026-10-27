"""Unit tests for motion estimation, state classification, and decision fusion."""
import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.data_models import (
    MotionResult,
    QualityMetric,
    LaneResult,
    ObjectBox,
    RoadMarkingResult
)
from src.state.decision_fusion import DecisionFusion
from src.state.motion_estimator import MotionEstimator


class TestStateModule(unittest.TestCase):
    """Test suite for driving state classification and decision fusion."""

    def setUp(self):
        self.cfg = {
            "state": {
                "flow_stopped_threshold": 0.85,
                "flow_decel_slope": -0.04,
                "flow_start_threshold": 1.25,
                "window_size_frames": 20,
                "stopped_persistence_frames": 10
            }
        }
        self.quality_good = QualityMetric(brightness=90.0, is_night=False, blur_score=150.0, is_blurred=False)

    def test_decision_fusion_motion_transitions(self):
        """Tests DecisionFusion response to motion changes (driving vs stopped)."""
        fusion = DecisionFusion(self.cfg)
        lane = LaneResult(lane_center_x=480.0, lane_width_px=300.0, status="BOTH_DETECTED")
        markings = RoadMarkingResult()

        # 1. Normal driving with flow 3.5
        res = fusion.evaluate(
            MotionResult(magnitude=3.5, is_reliable=True),
            self.quality_good, lane, [], markings, "KEEPING", (540, 960, 3)
        )
        self.assertEqual(res.ego_state, "NORMAL_DRIVING")
        self.assertEqual(res.fusion_state, "NORMAL")

        # 2. Stopped with flow 0.3 (after debounce frames)
        for _ in range(5):
            res = fusion.evaluate(
                MotionResult(magnitude=0.3, is_reliable=True),
                self.quality_good, lane, [], markings, "KEEPING", (540, 960, 3)
            )
        self.assertEqual(res.ego_state, "STOPPED_SUSPECTED")
        self.assertEqual(res.fusion_state, "STOPPED")

    def test_decision_fusion_pedestrian_override(self):
        """Tests that a pedestrian in the driving path forces STOP_REQUIRED."""
        fusion = DecisionFusion(self.cfg)
        motion = MotionResult(magnitude=3.0, is_reliable=True)
        lane = LaneResult(lane_center_x=480.0, lane_width_px=300.0, status="BOTH_DETECTED")
        markings = RoadMarkingResult()

        # Pedestrian directly in center lane, close (bottom y = 460 in 540h)
        ped = ObjectBox(
            track_id=1,
            bbox=(460, 390, 500, 460),
            class_id=0,
            label="person",
            confidence=0.92,
            center=(480.0, 425.0)
        )

        res = fusion.evaluate(
            motion=motion,
            quality=self.quality_good,
            lane=lane,
            objects=[ped],
            markings=markings,
            lane_change_status="KEEPING",
            frame_shape=(540, 960, 3)
        )
        self.assertEqual(res.fusion_state, "STOP_REQUIRED")
        self.assertIn("보행자", res.reason)

    def test_decision_fusion_traffic_light_red(self):
        """Tests that a red circular traffic light in upper ROI forces STOP_REQUIRED."""
        import cv2
        import numpy as np

        fusion = DecisionFusion(self.cfg)
        motion = MotionResult(magnitude=3.0, is_reliable=True)
        lane = LaneResult(lane_center_x=480.0, lane_width_px=300.0, status="BOTH_DETECTED")
        markings = RoadMarkingResult()

        # Create a synthetic 540x960 frame with a bright saturated red circular light at y=120 (0.22h)
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        # BGR: Bright pure saturated red (B=10, G=10, R=240) -> In HSV: H~0, S~240, V~240
        cv2.circle(frame, (480, 120), 12, (10, 10, 240), -1)

        # Evaluate over debounce frames (default 2)
        res = None
        for _ in range(3):
            res = fusion.evaluate(
                motion=motion,
                quality=self.quality_good,
                lane=lane,
                objects=[],
                markings=markings,
                lane_change_status="KEEPING",
                frame_shape=frame.shape,
                frame=frame
            )

        self.assertIsNotNone(res)
        self.assertTrue(res.traffic_light.detected)
        self.assertEqual(res.traffic_light.color, "RED")
        self.assertEqual(res.fusion_state, "STOP_REQUIRED")
        self.assertIn("적색 신호등", res.reason)

    def test_decision_fusion_traffic_light_roi_noise_filtered(self):
        """Tests that bright red lights in the lower half (car taillights) are excluded by the ROI."""
        import cv2
        import numpy as np

        fusion = DecisionFusion(self.cfg)
        motion = MotionResult(magnitude=3.0, is_reliable=True)
        lane = LaneResult(lane_center_x=480.0, lane_width_px=300.0, status="BOTH_DETECTED")
        markings = RoadMarkingResult()

        # Create frame with red light at y=380 (0.70h, bottom half)
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.circle(frame, (480, 380), 15, (10, 10, 240), -1)

        for _ in range(3):
            res = fusion.evaluate(
                motion=motion,
                quality=self.quality_good,
                lane=lane,
                objects=[],
                markings=markings,
                lane_change_status="KEEPING",
                frame_shape=frame.shape,
                frame=frame
            )

        self.assertFalse(res.traffic_light.detected)
        self.assertNotEqual(res.fusion_state, "STOP_REQUIRED")

    def test_decision_fusion_traffic_light_yolo_topdown(self):
        """Tests that YOLO traffic light bounding boxes take top priority and classify colors accurately."""
        import cv2
        import numpy as np

        fusion = DecisionFusion(self.cfg)
        motion = MotionResult(magnitude=3.0, is_reliable=True)
        lane = LaneResult(lane_center_x=480.0, lane_width_px=300.0, status="BOTH_DETECTED")
        markings = RoadMarkingResult()

        # Frame with a green lamp inside a 50x30 traffic light housing
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        # BGR: Pure bright green (B=20, G=240, R=20)
        cv2.circle(frame, (480, 115), 10, (20, 240, 20), -1)

        tl_box = ObjectBox(
            track_id=10,
            bbox=(455, 100, 505, 130),
            class_id=9,
            label="traffic light",
            confidence=0.91,
            center=(480.0, 115.0)
        )

        res = None
        for _ in range(3):
            res = fusion.evaluate(
                motion=motion,
                quality=self.quality_good,
                lane=lane,
                objects=[tl_box],
                markings=markings,
                lane_change_status="KEEPING",
                frame_shape=frame.shape,
                frame=frame
            )

        self.assertIsNotNone(res)
        self.assertTrue(res.traffic_light.detected)
        self.assertEqual(res.traffic_light.color, "GREEN")
        self.assertEqual(res.traffic_light.bbox, (455, 100, 505, 130))

    def test_decision_fusion_traffic_light_billboard_filtered(self):
        """Tests that bright red lights embedded in bright billboards (no dark housing) are filtered out."""
        import cv2
        import numpy as np

        fusion = DecisionFusion(self.cfg)
        motion = MotionResult(magnitude=3.0, is_reliable=True)
        lane = LaneResult(lane_center_x=480.0, lane_width_px=300.0, status="BOTH_DETECTED")
        markings = RoadMarkingResult()

        # Create a bright white electronic billboard with a red dot inside it
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        # White billboard background at y=100~160, x=440~520
        cv2.rectangle(frame, (440, 100), (520, 160), (240, 240, 240), -1)
        # Red graphic dot inside billboard
        cv2.circle(frame, (480, 130), 10, (10, 10, 240), -1)

        res = None
        for _ in range(3):
            res = fusion.evaluate(
                motion=motion,
                quality=self.quality_good,
                lane=lane,
                objects=[],
                markings=markings,
                lane_change_status="KEEPING",
                frame_shape=frame.shape,
                frame=frame
            )

        self.assertIsNotNone(res)
        self.assertFalse(res.traffic_light.detected)
        self.assertNotEqual(res.fusion_state, "STOP_REQUIRED")

    def test_decision_fusion_traffic_light_lateral_noise_filtered(self):
        """Tests that roadside commercial signs outside lateral ROI (x < 0.15w) are filtered out."""
        import cv2
        import numpy as np

        fusion = DecisionFusion(self.cfg)
        motion = MotionResult(magnitude=3.0, is_reliable=True)
        lane = LaneResult(lane_center_x=480.0, lane_width_px=300.0, status="BOTH_DETECTED")
        markings = RoadMarkingResult()

        # Red circular sign at far-left sidewalk (x=50, 0.05w)
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.circle(frame, (50, 120), 12, (10, 10, 240), -1)

        res = None
        for _ in range(3):
            res = fusion.evaluate(
                motion=motion,
                quality=self.quality_good,
                lane=lane,
                objects=[],
                markings=markings,
                lane_change_status="KEEPING",
                frame_shape=frame.shape,
                frame=frame
            )

        self.assertIsNotNone(res)
        self.assertFalse(res.traffic_light.detected)
        self.assertNotEqual(res.fusion_state, "STOP_REQUIRED")

    def test_motion_estimator_pure_tracking(self):
        """Tests that MotionEstimator accurately tracks camera displacement between shifted frames."""
        import cv2
        import numpy as np

        estimator = MotionEstimator(self.cfg)

        # Create a textured frame with grid dots in the active road region (0.35h ~ 0.90h)
        h, w = 540, 960
        f1 = np.zeros((h, w), dtype=np.uint8)
        for y in range(220, 460, 25):
            for x in range(150, 810, 25):
                cv2.circle(f1, (x, y), 3, 255, -1)

        # Shift frame by dx = 3.0, dy = 1.0 using affine warp
        M = np.float32([[1, 0, 3], [0, 1, 1]])
        f2 = cv2.warpAffine(f1, M, (w, h))

        # Frame 1: initialization
        res1 = estimator.estimate(f1, detected_objects=[])
        self.assertEqual(res1.tracked_points, 0)
        self.assertEqual(res1.magnitude, 0.0)

        # Frame 2: motion measurement
        res2 = estimator.estimate(f2, detected_objects=[])
        self.assertTrue(res2.is_reliable)
        self.assertGreater(res2.tracked_points, 20)
        self.assertAlmostEqual(res2.median_dx, 3.0, delta=0.5)
        self.assertAlmostEqual(res2.median_dy, 1.0, delta=0.5)
        self.assertAlmostEqual(res2.magnitude, np.hypot(3.0, 1.0), delta=0.5)

    def test_motion_estimator_foreground_filtering(self):
        """Tests that points inside moving foreground vehicle boxes are excluded from motion estimation."""
        import cv2
        import numpy as np

        estimator = MotionEstimator(self.cfg)

        h, w = 540, 960
        f1 = np.zeros((h, w), dtype=np.uint8)
        # Background dots at left road side
        for y in range(250, 450, 30):
            for x in range(150, 300, 30):
                cv2.circle(f1, (x, y), 3, 255, -1)

        # Foreground dots inside a vehicle area (x: 450~650, y: 250~450)
        for y in range(250, 450, 30):
            for x in range(450, 650, 30):
                cv2.circle(f1, (x, y), 3, 255, -1)

        # Frame 1 initialization
        estimator.estimate(f1, detected_objects=[])

        # Shift only: background dots shift by dx=3, dy=0
        # Foreground object is placed at (440, 240, 660, 460)
        lead_car = ObjectBox(
            track_id=1,
            bbox=(440, 240, 660, 460),
            class_id=2,
            label="car",
            confidence=0.95,
            center=(550.0, 350.0)
        )

        f2 = cv2.warpAffine(f1, np.float32([[1, 0, 3], [0, 1, 0]]), (w, h))
        res = estimator.estimate(f2, detected_objects=[lead_car])

        self.assertTrue(res.is_reliable)
        self.assertAlmostEqual(res.median_dx, 3.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
