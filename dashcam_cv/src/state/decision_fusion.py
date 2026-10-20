"""Decision fusion module combining ego state, lead vehicle, pedestrians, and road markings."""
from typing import List, Optional, Dict, Any, Tuple
from ..common.data_models import (
    MotionResult,
    DrivingStateResult,
    ObjectBox,
    RoadMarkingResult,
    LaneResult,
    QualityMetric
)
from .state_classifier import StateClassifier


class DecisionFusion:
    """Integrates multi-modal perceptions into unified situational decisions with anti-flicker filtering."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        st_cfg = self.cfg.get("state", {})
        self.state_classifier = StateClassifier(cfg)

        # Anti-flickering debounce and hold-time parameters
        self.state_hold_frames = int(st_cfg.get("state_hold_frames", 12))  # ~0.4s hold
        self.state_debounce_frames = int(st_cfg.get("state_debounce_frames", 2))

        self.current_fusion_state = "NORMAL"
        self.current_reason = "정상 주행 흐름 유지"
        self.state_hold_counter = 0

        self.pending_candidate_state = None
        self.pending_candidate_count = 0

        # Road marking temporal persistence
        self.stopline_persist = 0
        self.crosswalk_persist = 0

    def evaluate(
        self,
        motion: MotionResult,
        quality: QualityMetric,
        lane: LaneResult,
        objects: List[ObjectBox],
        markings: RoadMarkingResult,
        lane_change_status: str,
        frame_shape: Tuple[int, int, int]
    ) -> DrivingStateResult:
        """Evaluates full scene context and produces debounced, flicker-free driving decisions."""
        h, w = frame_shape[:2]

        # 1. Road marking temporal persistence (eliminates single-frame dropouts)
        if markings.stopline_detected:
            self.stopline_persist = min(10, self.stopline_persist + 3)
        else:
            self.stopline_persist = max(0, self.stopline_persist - 1)
        effective_stopline = (self.stopline_persist >= 3)

        if markings.crosswalk_detected:
            self.crosswalk_persist = min(10, self.crosswalk_persist + 3)
        else:
            self.crosswalk_persist = max(0, self.crosswalk_persist - 1)

        # 2. Get Ego State from State Machine
        ego_state, ego_reason, ego_conf = self.state_classifier.update(motion, quality)

        # 3. Extract Lead Vehicle and Pedestrian Dynamics
        lead_obj: Optional[ObjectBox] = None
        pedestrians_near = False
        pedestrians_mid = False

        lane_center = lane.lane_center_x if lane.lane_center_x is not None else (w / 2.0)
        lane_w = lane.lane_width_px if lane.lane_width_px is not None else (w * 0.40)

        for obj in objects:
            if obj.is_lead:
                lead_obj = obj

            if obj.label == "person":
                cx, cy = obj.center
                x1, y1, x2, y2 = obj.bbox
                # In or near our lane corridor
                if abs(cx - lane_center) < (lane_w * 0.7):
                    if y2 >= (h * 0.80):
                        pedestrians_near = True
                    elif y2 >= (h * 0.65):
                        pedestrians_mid = True

        lead_state = "NONE"
        if lead_obj is not None:
            if lead_obj.relative_distance_trend == "APPROACHING":
                lead_state = "APPROACHING"
            elif lead_obj.status == "STOPPED":
                lead_state = "STOPPED"
            else:
                lead_state = "MOVING"

        # 4. Instantaneous Decision Matrix (Raw Candidate)
        raw_state = "NORMAL"
        raw_reason = ego_reason

        pedestrian_on_crosswalk = any(obj.label == "person" and obj.is_on_crosswalk for obj in objects)

        # Hazard Priority 1: Pedestrian on crosswalk or near driving corridor
        if pedestrian_on_crosswalk:
            raw_state = "STOP_REQUIRED"
            raw_reason = "횡단보도 위 보행자 감지 (즉시 정지 필요)"
        elif pedestrians_near:
            raw_state = "STOP_REQUIRED"
            raw_reason = "전방 차로 내 근접 보행자 감지 (즉시 정지 필요)"

        # Hazard Priority 2: Stopline near while moving
        elif effective_stopline and ego_state in ("NORMAL_DRIVING", "DECELERATION_SUSPECTED"):
            if markings.stopline_y is not None and markings.stopline_y > (h * 0.70):
                raw_state = "SLOW_DOWN"
                raw_reason = "전방 정지선 근접 (감속 및 정지 준비)"

        # Hazard Priority 3: Pedestrian in mid zone
        elif pedestrians_mid:
            raw_state = "SLOW_DOWN"
            raw_reason = "전방 차로 주변 보행자 감지 (서행 필요)"

        # Lane Change (Trigger on IN_PROGRESS, avoid flickering on transient CANDIDATE)
        elif lane_change_status == "IN_PROGRESS":
            raw_state = "LANE_CHANGE"
            raw_reason = "차선 변경 동작 진행 중"

        # Multi-vehicle interaction matrix (Ego vs Lead)
        elif ego_state == "STOPPED_SUSPECTED":
            if lead_state in ("STOPPED", "NONE"):
                raw_state = "STOPPED"
                raw_reason = "내 차량 정지 상태 유지 (신호 대기 또는 정체)"
            elif lead_state in ("MOVING", "APPROACHING"):
                raw_state = "NORMAL"
                raw_reason = "선행 차량 출발 감지 (출발 준비 가능)"

        elif ego_state in ("NORMAL_DRIVING", "START_DETECTED"):
            if lead_state == "STOPPED" or lead_state == "APPROACHING":
                raw_state = "SLOW_DOWN"
                raw_reason = f"전방 선행 차량({lead_state}) 접근 주의 (감속 권고)"
            else:
                raw_state = "NORMAL"
                raw_reason = "정상 주행 흐름 유지"

        elif ego_state == "DECELERATION_SUSPECTED":
            raw_state = "SLOW_DOWN"
            if lead_state == "STOPPED":
                raw_reason = "전방 정지 차량을 향해 감속 주행 중"
            else:
                raw_reason = "주행 속도 감속 진행 중"

        elif ego_state == "UNCERTAIN":
            raw_state = "UNCERTAIN"
            raw_reason = "영상 품질 또는 신뢰도 저하로 판단 보류"

        # 5. Debounce & State Hold Filter (Flicker-Free Stabilization)
        # Critical hazard (STOP_REQUIRED): switch immediately without debounce delay!
        if raw_state == "STOP_REQUIRED":
            self.current_fusion_state = raw_state
            self.current_reason = raw_reason
            self.state_hold_counter = self.state_hold_frames
            self.pending_candidate_state = None
            self.pending_candidate_count = 0

        # If holding a warning/active state and raw falls back to NORMAL: hold state for minimum duration
        elif self.state_hold_counter > 0 and raw_state == "NORMAL":
            self.state_hold_counter -= 1
            # Maintain current_fusion_state and current_reason

        # Transitioning to another state (e.g. SLOW_DOWN, LANE_CHANGE, STOPPED): require debounce
        elif raw_state != self.current_fusion_state:
            if raw_state == self.pending_candidate_state:
                self.pending_candidate_count += 1
            else:
                self.pending_candidate_state = raw_state
                self.pending_candidate_count = 1

            if self.pending_candidate_count >= self.state_debounce_frames:
                self.current_fusion_state = raw_state
                self.current_reason = raw_reason
                self.state_hold_counter = self.state_hold_frames
                self.pending_candidate_state = None
                self.pending_candidate_count = 0
        else:
            self.current_fusion_state = raw_state
            self.current_reason = raw_reason
            self.pending_candidate_state = None
            self.pending_candidate_count = 0

        return DrivingStateResult(
            ego_state=ego_state,
            lead_state=lead_state,
            fusion_state=self.current_fusion_state,
            reason=self.current_reason,
            confidence=ego_conf,
            lane_change_status=lane_change_status
        )
