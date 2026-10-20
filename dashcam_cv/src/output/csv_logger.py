"""CSV Logging module for frame-by-frame status and performance records."""
import csv
import json
from pathlib import Path
from typing import Dict, Any, List
from ..common.data_models import (
    LaneResult,
    ObjectBox,
    RoadMarkingResult,
    MotionResult,
    DrivingStateResult,
    FrameTimings
)


class CSVLogger:
    """Records detailed frame analysis and execution timing logs."""

    def __init__(self, frame_log_path: str, perf_log_path: str):
        self.frame_log_path = frame_log_path
        self.perf_log_path = perf_log_path

        Path(frame_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(perf_log_path).parent.mkdir(parents=True, exist_ok=True)

        self.f_frame = open(frame_log_path, "w", newline="", encoding="utf-8-sig")
        self.f_perf = open(perf_log_path, "w", newline="", encoding="utf-8-sig")

        # 1. Frame log columns
        self.frame_fields = [
            "frame_id", "timestamp_sec", "lane_status", "lane_confidence",
            "lane_center_x", "lane_error_px", "object_count", "lead_vehicle_id",
            "lead_state", "ego_driving_state", "fusion_decision", "decision_reason",
            "crosswalk_detected", "stopline_detected", "optical_flow_magnitude"
        ]
        self.writer_frame = csv.DictWriter(self.f_frame, fieldnames=self.frame_fields)
        self.writer_frame.writeheader()

        # 2. Performance log columns
        self.perf_fields = [
            "frame_id", "decode_ms", "preprocess_ms", "lane_ms", "detect_ms",
            "tracking_ms", "marking_ms", "motion_ms", "decision_ms", "render_ms",
            "total_ms", "detection_updated"
        ]
        self.writer_perf = csv.DictWriter(self.f_perf, fieldnames=self.perf_fields)
        self.writer_perf.writeheader()

    def log(
        self,
        frame_id: int,
        timestamp_sec: float,
        lane: LaneResult,
        objects: List[ObjectBox],
        markings: RoadMarkingResult,
        motion: MotionResult,
        state: DrivingStateResult,
        timings: FrameTimings
    ):
        """Writes current frame data to both logs."""
        lead_id = None
        for obj in objects:
            if obj.is_lead:
                lead_id = obj.track_id
                break

        self.writer_frame.writerow({
            "frame_id": frame_id,
            "timestamp_sec": round(timestamp_sec, 3),
            "lane_status": lane.status,
            "lane_confidence": round(lane.lane_confidence, 3),
            "lane_center_x": round(lane.lane_center_x, 1) if lane.lane_center_x is not None else "",
            "lane_error_px": round(lane.lane_error_px, 1) if lane.lane_error_px is not None else "",
            "object_count": len(objects),
            "lead_vehicle_id": lead_id if lead_id is not None else "",
            "lead_state": state.lead_state,
            "ego_driving_state": state.ego_state,
            "fusion_decision": state.fusion_state,
            "decision_reason": state.reason,
            "crosswalk_detected": markings.crosswalk_detected,
            "stopline_detected": markings.stopline_detected,
            "optical_flow_magnitude": round(motion.magnitude, 3)
        })

        self.writer_perf.writerow({
            "frame_id": frame_id,
            "decode_ms": round(timings.decode_ms, 2),
            "preprocess_ms": round(timings.preprocess_ms, 2),
            "lane_ms": round(timings.lane_ms, 2),
            "detect_ms": round(timings.detect_ms, 2),
            "tracking_ms": round(timings.tracking_ms, 2),
            "marking_ms": round(timings.marking_ms, 2),
            "motion_ms": round(timings.motion_ms, 2),
            "decision_ms": round(timings.decision_ms, 2),
            "render_ms": round(timings.render_ms, 2),
            "total_ms": round(timings.total_ms, 2),
            "detection_updated": timings.detection_updated
        })

    def close(self):
        """Flushes and closes both log files."""
        if self.f_frame and not self.f_frame.closed:
            self.f_frame.flush()
            self.f_frame.close()
        if self.f_perf and not self.f_perf.closed:
            self.f_perf.flush()
            self.f_perf.close()
