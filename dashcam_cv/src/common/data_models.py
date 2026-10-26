"""Data models for dashcam_cv system.

Defines standardized data classes for inter-module communication.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict


@dataclass
class QualityMetric:
    """Video frame quality assessment metric."""
    brightness: float = 0.0
    is_night: bool = False
    blur_score: float = 0.0
    is_blurred: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brightness": round(self.brightness, 2),
            "is_night": self.is_night,
            "blur_score": round(self.blur_score, 2),
            "is_blurred": self.is_blurred,
        }


@dataclass
class LaneResult:
    """Lane detection and tracking result."""
    left_fit: Optional[List[float]] = None
    right_fit: Optional[List[float]] = None
    left_line: Optional[Tuple[int, int, int, int]] = None   # (x1, y1, x2, y2)
    right_line: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    center_path: List[Tuple[int, int]] = field(default_factory=list)
    lane_center_x: Optional[float] = None
    lane_error_px: Optional[float] = None
    lane_confidence: float = 0.0
    status: str = "LOST"  # BOTH_DETECTED, LEFT_ONLY, RIGHT_ONLY, TEMPORARY_LOST, LOST
    lost_frames: int = 0
    is_uncertain: bool = True
    lane_width_px: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "confidence": round(self.lane_confidence, 3),
            "lane_center_x": round(self.lane_center_x, 1) if self.lane_center_x is not None else None,
            "lane_error_px": round(self.lane_error_px, 1) if self.lane_error_px is not None else None,
            "lost_frames": self.lost_frames,
            "is_uncertain": self.is_uncertain,
        }


@dataclass
class ObjectBox:
    """Tracked road object representation."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    label: str
    confidence: float
    center: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    is_lead: bool = False
    is_on_crosswalk: bool = False
    relative_distance_trend: str = "UNKNOWN"  # APPROACHING, RECEDING, STABLE, UNKNOWN
    status: str = "UNKNOWN"                  # MOVING, STOPPED, UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.track_id,
            "bbox": [int(v) for v in self.bbox],
            "label": self.label,
            "confidence": round(self.confidence, 2),
            "is_lead": self.is_lead,
            "is_on_crosswalk": self.is_on_crosswalk,
            "status": self.status,
            "distance_trend": self.relative_distance_trend,
        }


@dataclass
class RoadMarkingResult:
    """Crosswalk and stopline detection result."""
    crosswalk_detected: bool = False
    crosswalk_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    stopline_detected: bool = False
    stopline_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    stopline_y: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crosswalk_detected": self.crosswalk_detected,
            "crosswalk_count": len(self.crosswalk_boxes),
            "stopline_detected": self.stopline_detected,
            "stopline_y": round(self.stopline_y, 1) if self.stopline_y is not None else None,
        }


@dataclass
class MotionResult:
    """Background optical flow motion estimation result."""
    magnitude: float = 0.0
    median_dx: float = 0.0
    median_dy: float = 0.0
    tracked_points: int = 0
    is_reliable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "magnitude": round(self.magnitude, 3),
            "median_dx": round(self.median_dx, 3),
            "median_dy": round(self.median_dy, 3),
            "tracked_points": self.tracked_points,
            "is_reliable": self.is_reliable,
        }


@dataclass
class TrafficLightResult:
    """Traffic light detection result based on color thresholding and circular light shape."""
    detected: bool = False
    color: Optional[str] = None  # "RED", "YELLOW", "GREEN", None
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    circle: Optional[Tuple[int, int, int]] = None  # (cx, cy, radius)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "color": self.color,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "circle": list(self.circle) if self.circle is not None else None,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class DrivingStateResult:
    """Ego vehicle and lead vehicle combined state determination."""
    ego_state: str = "UNCERTAIN"       # NORMAL_DRIVING, DECELERATION_SUSPECTED, STOPPED_SUSPECTED, START_DETECTED, UNCERTAIN
    lead_state: str = "NONE"           # NONE, MOVING, STOPPED, APPROACHING
    fusion_state: str = "UNCERTAIN"    # NORMAL, SLOW_DOWN, STOP_REQUIRED, STOPPED, LANE_CHANGE, UNCERTAIN
    reason: str = "Initializing"
    confidence: float = 0.5
    lane_change_status: str = "KEEPING" # KEEPING, CANDIDATE, IN_PROGRESS, COMPLETE
    traffic_light: TrafficLightResult = field(default_factory=TrafficLightResult)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ego_state": self.ego_state,
            "lead_state": self.lead_state,
            "fusion_state": self.fusion_state,
            "reason": self.reason,
            "confidence": round(self.confidence, 2),
            "lane_change_status": self.lane_change_status,
            "traffic_light": self.traffic_light.to_dict(),
        }


@dataclass
class FrameTimings:
    """Execution timing profile for a single frame (milliseconds)."""
    decode_ms: float = 0.0
    preprocess_ms: float = 0.0
    lane_ms: float = 0.0
    detect_ms: float = 0.0
    tracking_ms: float = 0.0
    marking_ms: float = 0.0
    motion_ms: float = 0.0
    decision_ms: float = 0.0
    render_ms: float = 0.0
    total_ms: float = 0.0
    detection_updated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decode_ms": round(self.decode_ms, 2),
            "preprocess_ms": round(self.preprocess_ms, 2),
            "lane_ms": round(self.lane_ms, 2),
            "detect_ms": round(self.detect_ms, 2),
            "tracking_ms": round(self.tracking_ms, 2),
            "marking_ms": round(self.marking_ms, 2),
            "motion_ms": round(self.motion_ms, 2),
            "decision_ms": round(self.decision_ms, 2),
            "render_ms": round(self.render_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "detection_updated": self.detection_updated,
        }
