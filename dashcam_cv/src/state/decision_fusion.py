"""Decision fusion module combining ego state, lead vehicle, pedestrians, road markings, and traffic lights."""
import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from ..common.data_models import (
    MotionResult,
    DrivingStateResult,
    ObjectBox,
    RoadMarkingResult,
    LaneResult,
    QualityMetric,
    TrafficLightResult
)


class DecisionFusion:
    """Integrates multi-modal perceptions into unified situational decisions with anti-flicker filtering."""

    # [NEW] Traffic light HSV color ranges (Phase 1 / Phase 2 통합 상수)
    _TL_RED_LOW = (0, 10)       # Hue range: lower red wrap
    _TL_RED_HIGH = (165, 180)   # Hue range: upper red wrap
    _TL_GREEN = (40, 95)        # Hue range: green

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        st_cfg = self.cfg.get("state", {})
        self.stopped_th = float(st_cfg.get("flow_stopped_threshold", 0.85))

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

        # Traffic light detection configuration and temporal state
        tl_cfg = st_cfg.get("traffic_light", {})
        self.tl_roi_top = float(tl_cfg.get("roi_top", 0.10))
        self.tl_roi_bottom = float(tl_cfg.get("roi_bottom", 0.50))
        self.tl_roi_left = float(tl_cfg.get("roi_left", 0.15))
        self.tl_roi_right = float(tl_cfg.get("roi_right", 0.85))
        self.tl_min_sat = int(tl_cfg.get("min_saturation", 120))
        self.tl_min_val = int(tl_cfg.get("min_value", 180))
        self.tl_debounce_frames = int(tl_cfg.get("debounce_frames", 2))

        self.last_detected_tl: TrafficLightResult = TrafficLightResult()
        self.tl_candidate_color: Optional[str] = None
        self.tl_candidate_count: int = 0

    # [NEW] Phase 1 / Phase 2 공용 HSV 마스크 빌더 (중복 제거 및 범위 통일)
    def _build_color_masks(
        self, hsv: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Builds unified R/G binary masks with clamped saturation/value thresholds.

        Note: Yellow mask removed to eliminate false positives from streetlights
        and electronic billboards.
        """
        min_s = min(self.tl_min_sat, 110)
        min_v = min(self.tl_min_val, 150)

        mask_r = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([self._TL_RED_LOW[0], min_s, min_v]),
                             np.array([self._TL_RED_LOW[1], 255, 255])),
            cv2.inRange(hsv, np.array([self._TL_RED_HIGH[0], min_s, min_v]),
                             np.array([self._TL_RED_HIGH[1], 255, 255]))
        )
        mask_g = cv2.inRange(hsv, np.array([self._TL_GREEN[0], min_s, min_v]),
                                  np.array([self._TL_GREEN[1], 255, 255]))
        return mask_r, mask_g

    def _classify_tl_crop(
        self,
        crop: np.ndarray,
        ox1: int,
        oy1: int
    ) -> Tuple[Optional[str], float, Optional[Tuple[int, int, int]]]:
        """Classifies lit lamp color inside a YOLO-detected traffic light bounding box (Top-Down)."""
        ch, cw = crop.shape[:2]
        if ch < 4 or cw < 4:
            return None, 0.0, None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_r, mask_g = self._build_color_masks(hsv)

        counts = {
            "RED": cv2.countNonZero(mask_r),
            "GREEN": cv2.countNonZero(mask_g)
        }
        best_color = max(counts, key=counts.get)
        max_cnt = counts[best_color]

        # Require at least 4 lit pixels or 2.5% of the bounding box
        box_area = cw * ch
        if max_cnt >= max(4, int(box_area * 0.025)):
            target_mask = mask_r if best_color == "RED" else mask_g
            pts = cv2.findNonZero(target_mask)
            if pts is not None and len(pts) > 0:
                pts_2d = pts.reshape(-1, 2)
                mcx = int(np.mean(pts_2d[:, 0]))
                mcy = int(np.mean(pts_2d[:, 1]))
            else:
                mcx = cw // 2
                mcy = ch // 2

            gcx = ox1 + mcx
            gcy = oy1 + mcy
            gr = max(4, min(cw, ch) // 3)
            confidence = min(0.98, max_cnt / float(box_area) + 0.4)
            return best_color, confidence, (gcx, gcy, gr)

        return None, 0.0, None

    def detect_traffic_light(
        self,
        frame: np.ndarray,
        detected_objects: Optional[List[ObjectBox]] = None
    ) -> TrafficLightResult:
        """Detects traffic light colors prioritizing YOLO-detected bounding boxes (Top-Down).

        Falls back to lightweight contour analysis with dark-housing and lateral-ROI validation
        to eliminate false positives from streetlights and electronic billboards without heavy HoughCircles.
        """
        if frame is None or frame.size == 0:
            return TrafficLightResult()

        h, w = frame.shape[:2]
        detected_candidates = []

        # ====================================================================
        # Phase 1: [Top-Down] Prioritize YOLO traffic light detections (Fast & Accurate)
        # ====================================================================
        tl_boxes = [obj for obj in (detected_objects or []) if obj.label == "traffic light"]
        if tl_boxes:
            for obj in tl_boxes:
                ox1, oy1, ox2, oy2 = obj.bbox
                # Filter out boxes completely outside upper driving corridor
                if ox2 < w * self.tl_roi_left or ox1 > w * self.tl_roi_right:
                    continue
                if oy2 < h * 0.05 or oy1 > h * self.tl_roi_bottom * 1.2:
                    continue

                crop = frame[max(0, oy1):min(h, oy2), max(0, ox1):min(w, ox2)]
                if crop.size == 0:
                    continue

                color, conf, circle = self._classify_tl_crop(crop, ox1, oy1)
                if color:
                    detected_candidates.append({
                        "color": color,
                        "bbox": (ox1, oy1, ox2, oy2),
                        "circle": circle,
                        "confidence": conf,
                        "count": 100  # High priority for YOLO-grounded detection
                    })

        # ====================================================================
        # Phase 2: [Fallback] Lightweight Contour & Housing Validation
        # (Executes only when YOLO detected no traffic lights; no heavy HoughCircles)
        # ====================================================================
        if not detected_candidates:
            y_start = int(h * self.tl_roi_top)
            y_end = int(h * self.tl_roi_bottom)
            x_start = int(w * self.tl_roi_left)
            x_end = int(w * self.tl_roi_right)

            if y_start < y_end and x_start < x_end:
                roi = frame[y_start:y_end, x_start:x_end]
                roi_h, roi_w = roi.shape[:2]

                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask_red, mask_green = self._build_color_masks(hsv)
                combined_mask = mask_red | mask_green

                if cv2.countNonZero(combined_mask) >= 6:
                    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if not (12 <= area <= 800):
                            continue

                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter <= 0:
                            continue

                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        if circularity < 0.65:
                            continue

                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        icx, icy, r = int(cx), int(cy), max(4, int(radius))
                        if not (0 <= icx < roi_w and 0 <= icy < roi_h):
                            continue

                        # [Dark Housing Validation] Check surrounding area brightness
                        # Streetlights and billboard graphics lack dark housing/visors
                        pad = int(r * 1.8)
                        py1, py2 = max(0, icy - pad), min(roi_h, icy + pad)
                        px1, px2 = max(0, icx - pad), min(roi_w, icx + pad)
                        sub_v = hsv[py1:py2, px1:px2, 2]
                        sub_lit = combined_mask[py1:py2, px1:px2]
                        non_lit_v = sub_v[sub_lit == 0]
                        if len(non_lit_v) > 0 and np.mean(non_lit_v) > 195:
                            # Rejected: surrounded by blindingly bright pixels (billboard / streetlight halo)
                            continue

                        sub_r = mask_red[max(0, icy - r):min(roi_h, icy + r), max(0, icx - r):min(roi_w, icx + r)]
                        sub_g = mask_green[max(0, icy - r):min(roi_h, icy + r), max(0, icx - r):min(roi_w, icx + r)]

                        r_cnt = cv2.countNonZero(sub_r)
                        g_cnt = cv2.countNonZero(sub_g)
                        max_c = max(r_cnt, g_cnt)

                        circle_area = np.pi * (r ** 2)
                        if max_c >= max(5, int(circle_area * 0.20)):
                            c_map = {r_cnt: "RED", g_cnt: "GREEN"}
                            best_color = c_map[max_c]
                            gcx = icx + x_start
                            gcy = icy + y_start
                            confidence = round(min(0.92, circularity * 0.8), 2)
                            bx1 = max(0, gcx - int(r * 3.5))
                            by1 = max(0, gcy - int(r * 1.6))
                            bx2 = min(w, gcx + int(r * 3.5))
                            by2 = min(h, gcy + int(r * 1.6))
                            detected_candidates.append({
                                "color": best_color,
                                "bbox": (bx1, by1, bx2, by2),
                                "circle": (gcx, gcy, r),
                                "confidence": confidence,
                                "count": max_c
                            })

        if not detected_candidates:
            self.tl_candidate_color = None
            self.tl_candidate_count = 0
            return TrafficLightResult()

        best_cand = max(detected_candidates, key=lambda c: (c["count"], c["confidence"]))
        tl_color = best_cand["color"]
        conf = best_cand["confidence"]
        matched_bbox = best_cand["bbox"]
        circle = best_cand.get("circle")

        # Temporal debounce
        if tl_color == self.tl_candidate_color:
            self.tl_candidate_count += 1
        else:
            self.tl_candidate_color = tl_color
            self.tl_candidate_count = 1

        is_confirmed = (self.tl_candidate_count >= self.tl_debounce_frames)
        result = TrafficLightResult(
            detected=is_confirmed,
            color=tl_color if is_confirmed else None,
            bbox=matched_bbox if is_confirmed else None,
            circle=circle if is_confirmed else None,
            confidence=conf if is_confirmed else 0.0
        )
        if is_confirmed:
            self.last_detected_tl = result
        return result

    def evaluate(
        self,
        motion: MotionResult,
        quality: QualityMetric,
        lane: LaneResult,
        objects: List[ObjectBox],
        markings: RoadMarkingResult,
        lane_change_status: str,
        frame_shape: Tuple[int, int, int],
        frame: Optional[np.ndarray] = None
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

        # 1.5 Traffic light detection within upper ROI (0.10h ~ 0.50h)
        tl_result = TrafficLightResult()
        if frame is not None:
            tl_result = self.detect_traffic_light(frame, objects)
        elif self.last_detected_tl.detected:
            tl_result = self.last_detected_tl

        # 2. Get Ego State directly from Motion and Quality
        if not motion.is_reliable or quality.is_blurred:
            ego_state = "UNCERTAIN"
            ego_reason = "영상 흔들림 또는 특징점 소실로 모션 측정 신뢰도 낮음"
            ego_conf = 0.30
        elif motion.magnitude < self.stopped_th:
            ego_state = "STOPPED_SUSPECTED"
            ego_reason = f"배경 이동량({motion.magnitude:.2f}px) 미미하여 정지 판단"
            ego_conf = 0.90
        else:
            ego_state = "NORMAL_DRIVING"
            ego_reason = f"유의미한 배경 이동량({motion.magnitude:.2f}px) 유지"
            ego_conf = 0.88

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
                if abs(cx - lane_center) < (lane_w * 0.45): # (보행자 침범 범위 축소 - 261029)
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

        # Hazard Priority 1.5: Traffic Light Red
        elif tl_result.detected and tl_result.color == "RED":
            raw_state = "STOP_REQUIRED"
            raw_reason = "전방 적색 신호등 감지 (즉시 정지 필요)"

        # Hazard Priority 2: Stopline near while moving
        elif effective_stopline and ego_state == "NORMAL_DRIVING":  # [CHANGED] "DECELERATION_SUSPECTED" 제거 (도달 불가 값)
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

        # Multi-vehicle interaction matrix (Ego vs Lead vs Traffic Light Green)
        elif ego_state == "STOPPED_SUSPECTED":
            if tl_result.detected and tl_result.color == "GREEN":
                raw_state = "NORMAL"
                raw_reason = "전방 녹색 신호등 점등 (출발 준비 가능)"
            elif lead_state in ("STOPPED", "NONE"):
                raw_state = "STOPPED"
                raw_reason = "내 차량 정지 상태 유지 (신호 대기 또는 정체)"
            elif lead_state in ("MOVING", "APPROACHING"):
                raw_state = "NORMAL"
                raw_reason = "선행 차량 출발 감지 (출발 준비 가능)"

        elif ego_state == "NORMAL_DRIVING":  # [CHANGED] "START_DETECTED" 제거 (도달 불가 값)
            if lead_state == "STOPPED" or lead_state == "APPROACHING":
                raw_state = "SLOW_DOWN"
                raw_reason = f"전방 선행 차량({lead_state}) 접근 주의 (감속 권고)"
            else:
                raw_state = "NORMAL"
                raw_reason = "정상 주행 흐름 유지"

        # [REMOVED] ego_state == "DECELERATION_SUSPECTED" 블록 전체 제거
        # (ego_state는 "UNCERTAIN" / "STOPPED_SUSPECTED" / "NORMAL_DRIVING"만 할당되어 도달 불가)

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
            lane_change_status=lane_change_status,
            traffic_light=tl_result
        )
