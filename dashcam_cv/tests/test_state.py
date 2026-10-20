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
from src.state.state_classifier import StateClassifier
from src.state.decision_fusion import DecisionFusion


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

    def test_state_machine_transitions(self):
        """Tests transitions: Driving -> Decelerating -> Stopped -> Starting."""
        classifier = StateClassifier(self.cfg)

        # 1. Normal driving with stable flow ~ 3.5 px/frame
        for _ in range(15):
            st, reason, conf = classifier.update(
                MotionResult(magnitude=3.5, is_reliable=True), self.quality_good
            )
        self.assertEqual(st, "NORMAL_DRIVING")

        # 2. Deceleration: flow declining from 3.5 down to 0.9
        for mag in [3.0, 2.5, 2.1, 1.8, 1.5, 1.3, 1.1, 0.95]:
            st, reason, conf = classifier.update(
                MotionResult(magnitude=mag, is_reliable=True), self.quality_good
            )
        self.assertEqual(st, "DECELERATION_SUSPECTED")

        # 3. Stopped: low flow sustained for 12 frames
        for _ in range(12):
            st, reason, conf = classifier.update(
                MotionResult(magnitude=0.3, is_reliable=True), self.quality_good
            )
        self.assertEqual(st, "STOPPED_SUSPECTED")

        # 4. Start: flow surge from 0.3 to 2.2
        st, reason, conf = classifier.update(
            MotionResult(magnitude=2.2, is_reliable=True), self.quality_good
        )
        self.assertEqual(st, "START_DETECTED")

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


if __name__ == "__main__":
    unittest.main()
