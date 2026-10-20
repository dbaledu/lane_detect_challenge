"""Metrics calculation and summary report generation module."""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List
from ..common.data_models import FrameTimings, LaneResult, DrivingStateResult, RoadMarkingResult


class MetricsCollector:
    """Aggregates execution timings and perception statistics to write summary.json."""

    def __init__(self, summary_path: str, video_meta: Dict[str, Any]):
        self.summary_path = summary_path
        self.video_meta = video_meta

        self.timings: List[FrameTimings] = []
        self.ego_states = Counter()
        self.fusion_states = Counter()
        self.lane_statuses = Counter()
        self.crosswalk_count = 0
        self.stopline_count = 0
        self.total_objects = 0

    def record(
        self,
        timings: FrameTimings,
        lane: LaneResult,
        state: DrivingStateResult,
        markings: RoadMarkingResult,
        num_objects: int
    ):
        """Records metrics for a processed frame."""
        self.timings.append(timings)
        self.ego_states[state.ego_state] += 1
        self.fusion_states[state.fusion_state] += 1
        self.lane_statuses[lane.status] += 1
        self.total_objects += num_objects
        if markings.crosswalk_detected:
            self.crosswalk_count += 1
        if markings.stopline_detected:
            self.stopline_count += 1

    def generate_summary(self) -> Dict[str, Any]:
        """Calculates statistics and writes summary.json."""
        n = len(self.timings)
        if n == 0:
            return {"error": "No frames processed"}

        total_times = [t.total_ms for t in self.timings]
        lane_times = [t.lane_ms for t in self.timings]
        detect_times = [t.detect_ms for t in self.timings]
        tracking_times = [t.tracking_ms for t in self.timings]
        motion_times = [t.motion_ms for t in self.timings]
        decision_times = [t.decision_ms for t in self.timings]
        render_times = [t.render_ms for t in self.timings]

        mean_total = float(np.mean(total_times))
        p95_total = float(np.percentile(total_times, 95))
        effective_fps = (1000.0 / mean_total) if mean_total > 0 else 0.0

        valid_lane_frames = self.lane_statuses.get("BOTH_DETECTED", 0) + \
                            self.lane_statuses.get("LEFT_ONLY", 0) + \
                            self.lane_statuses.get("RIGHT_ONLY", 0)
        lane_detection_rate = valid_lane_frames / float(n) if n > 0 else 0.0

        summary = {
            "video_metadata": self.video_meta,
            "performance": {
                "processed_frames": n,
                "effective_processing_fps": round(effective_fps, 2),
                "mean_total_ms": round(mean_total, 2),
                "p95_total_ms": round(p95_total, 2),
                "module_latency_mean_ms": {
                    "decode": round(float(np.mean([t.decode_ms for t in self.timings])), 2),
                    "preprocess": round(float(np.mean([t.preprocess_ms for t in self.timings])), 2),
                    "lane": round(float(np.mean(lane_times)), 2),
                    "detect": round(float(np.mean(detect_times)), 2),
                    "tracking": round(float(np.mean(tracking_times)), 2),
                    "motion": round(float(np.mean(motion_times)), 2),
                    "decision": round(float(np.mean(decision_times)), 2),
                    "render": round(float(np.mean(render_times)), 2)
                }
            },
            "perception_statistics": {
                "lane_detection_rate": round(lane_detection_rate, 4),
                "lane_status_distribution": dict(self.lane_statuses),
                "average_objects_per_frame": round(self.total_objects / float(n), 2),
                "crosswalk_detected_frames": self.crosswalk_count,
                "stopline_detected_frames": self.stopline_count
            },
            "driving_state_distribution": {
                "ego_state_frames": dict(self.ego_states),
                "ego_state_percent": {
                    k: round((v / float(n)) * 100.0, 1) for k, v in self.ego_states.items()
                },
                "fusion_decision_frames": dict(self.fusion_states),
                "fusion_decision_percent": {
                    k: round((v / float(n)) * 100.0, 1) for k, v in self.fusion_states.items()
                }
            }
        }

        Path(self.summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary
