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


if __name__ == "__main__":
    unittest.main()
