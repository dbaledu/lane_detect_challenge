"""Unit tests for object detection, tracking, and road marking modules."""
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.data_models import (
    ObjectBox,
    LaneResult,
    RoadMarkingResult,
    MotionResult,
    DrivingStateResult,
    FrameTimings,
    QualityMetric,
)
from src.common.visualizer import HUDVisualizer
from src.objects.detector import ObjectDetector
from src.objects.object_tracker import ObjectTracker
from src.objects.road_markings import RoadMarkingDetector


class TestDetectorModule(unittest.TestCase):
    """Test suite for object perception and road markings."""

    def setUp(self):
        self.w = 960
        self.h = 540
        self.cfg = {
            "detector": {
                "enabled": True,
                "backend": "none",  # Unit test fallback
                "detect_interval": 3,
                "confidence_threshold": 0.30
            },
            "tracker": {
                "max_distance_px": 80.0,
                "max_missed_frames": 5,
                "lead_vehicle_corridor_margin_px": 120.0
            },
            "road_markings": {
                "enable": True,
                "crosswalk_min_stripes": 3,
                "stopline_min_width_ratio": 0.20,
                "stopline_max_slope": 0.20
            },
            "roi": {
                "top_ratio": 0.65,
                "bottom_ratio": 0.88,
                "top_left_ratio": 0.44,
                "top_right_ratio": 0.56,
                "bottom_left_ratio": 0.18,
                "bottom_right_ratio": 0.82
            }
        }

    def test_object_tracker_lead_vehicle_assignment(self):
        """Tests that vehicle directly in front within lane corridor is marked as lead."""
        tracker = ObjectTracker(self.cfg)

        # Vehicle 1: In center lane (x=480, y=380)
        obj1 = ObjectBox(
            track_id=-1,
            bbox=(430, 320, 530, 420),
            class_id=2,
            label="car",
            confidence=0.88,
            center=(480.0, 370.0)
        )
        # Vehicle 2: In adjacent right lane (x=750, y=350)
        obj2 = ObjectBox(
            track_id=-1,
            bbox=(700, 300, 800, 400),
            class_id=2,
            label="car",
            confidence=0.85,
            center=(750.0, 350.0)
        )

        lane = LaneResult(lane_center_x=480.0, status="BOTH_DETECTED")
        tracked = tracker.update([obj1, obj2], detection_executed=True, lane=lane, frame_shape=(self.h, self.w, 3))

        self.assertEqual(len(tracked), 2)
        lead_objs = [o for o in tracked if o.is_lead]
        self.assertEqual(len(lead_objs), 1)
        self.assertEqual(lead_objs[0].center[0], 480.0)

    def test_stopline_detection_on_synthetic_image(self):
        """Tests stopline horizontal edge line detection."""
        detector = RoadMarkingDetector(self.cfg)
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        gray = np.zeros((self.h, self.w), dtype=np.uint8)
        edges = np.zeros((self.h, self.w), dtype=np.uint8)

        # Draw a horizontal stopline in the lower road region
        y_stop = int(self.h * 0.78)
        cv2.line(edges, (int(self.w * 0.30), y_stop), (int(self.w * 0.70), y_stop), 255, 3)

        res = detector.detect(frame, gray, edges)
        self.assertTrue(res.stopline_detected)
        self.assertIsNotNone(res.stopline_y)
        self.assertAlmostEqual(res.stopline_y, y_stop, delta=5)

    def test_visualizer_show_object_labels_toggle(self):
        """Tests that object labels can be toggled on/off via configuration."""
        obj = ObjectBox(
            track_id=1,
            bbox=(100, 100, 200, 200),
            class_id=2,
            label="car",
            confidence=0.9,
            center=(150.0, 150.0)
        )
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        common_kwargs = dict(
            frame=frame,
            frame_idx=0,
            timestamp_sec=0.0,
            lane=LaneResult(status="LOST"),
            objects=[obj],
            markings=RoadMarkingResult(),
            motion=MotionResult(magnitude=0.0),
            state=DrivingStateResult(ego_state="STOPPED"),
            timings=FrameTimings(),
            quality=QualityMetric(brightness=100.0, blur_score=100.0, is_night=False, is_blurred=False)
        )

        # 1. With labels enabled
        cfg_with_labels = {"visualization": {"show_object_labels": True, "show_hud_card": False}}
        vis_with_labels = HUDVisualizer(cfg_with_labels).draw(**common_kwargs)

        # 2. With labels disabled
        cfg_no_labels = {"visualization": {"show_object_labels": False, "show_hud_card": False}}
        vis_no_labels = HUDVisualizer(cfg_no_labels).draw(**common_kwargs)

        # The image without labels should differ from the image with labels
        self.assertFalse(np.array_equal(vis_with_labels, vis_no_labels))


if __name__ == "__main__":
    unittest.main()
