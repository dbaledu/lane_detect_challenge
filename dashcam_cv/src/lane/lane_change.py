"""Lane departure and lane change detection module."""
from typing import Optional, Dict, Any
from ..common.data_models import LaneResult


class LaneChangeDetector:
    """Distinguishes genuine lane change maneuvers from temporary lane occlusions."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        lc_cfg = self.cfg.get("lane_change", {})
        self.drift_threshold_px = float(lc_cfg.get("drift_threshold_px", 35.0))
        self.candidate_frames = int(lc_cfg.get("candidate_frames", 3))
        self.progress_frames = int(lc_cfg.get("progress_frames", 7))

        self.prev_error: Optional[float] = None
        self.drift_count: int = 0
        self.status: str = "KEEPING"  # KEEPING, CANDIDATE, IN_PROGRESS, COMPLETE

    def update(self, lane: LaneResult) -> str:
        """Evaluates lateral drift trends to detect lane change state."""
        # If lane is lost, do not falsely claim lane change
        if lane.status in ("LOST", "TEMPORARY_LOST") or lane.lane_error_px is None:
            self.drift_count = max(0, self.drift_count - 1)
            if self.drift_count == 0:
                self.status = "KEEPING"
            return self.status

        error = lane.lane_error_px

        if self.prev_error is not None:
            diff = error - self.prev_error
            # Drift is occurring if lateral shift per frame is significant or total error is large
            if abs(error) > self.drift_threshold_px:
                self.drift_count += 1
            else:
                self.drift_count = max(0, self.drift_count - 1)
        else:
            self.drift_count = 0

        self.prev_error = error

        if self.drift_count >= self.progress_frames:
            self.status = "IN_PROGRESS"
        elif self.drift_count >= self.candidate_frames:
            self.status = "CANDIDATE"
        else:
            self.status = "KEEPING"

        return self.status
