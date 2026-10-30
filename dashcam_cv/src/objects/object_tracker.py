"""Object tracking module with ID persistence and lead vehicle dynamics analysis.

특징:
1. 프레임 간격(dt) 기반 정규화 속도 계산 및 최대 속도 클램핑 (박스 허공 날림 원천 방지)
2. EMA(Exponential Moving Average) 스무딩으로 부드러운 바운딩 박스 연속 추적
3. 신호등 등 정적 객체 고정(속도=0) 처리로 하늘/허공 날림 방지
4. 미탐지 프레임 속도 감쇠(Damping) 및 화면 경계 클리핑/유효성 검사
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
from ..common.data_models import ObjectBox, LaneResult
from ..common.geometry import calculate_iou


class TrackRecord:
    """Historical state of an individual tracked object with EMA smoothing and velocity damping."""

    def __init__(self, track_id: int, obj: ObjectBox):
        self.track_id = track_id
        self.class_id = obj.class_id
        self.label = obj.label
        self.confidence = float(obj.confidence)

        # 바운딩 박스를 float형 배열로 관리하여 보간 및 EMA 스무딩 오차 방지
        self.bbox_f = np.array(obj.bbox, dtype=np.float32)
        self.bbox = tuple(int(round(v)) for v in self.bbox_f)
        self.center = obj.center

        self.missed_frames = 0
        self.frames_since_detect = 0
        self.hits = 1

        self.history = deque(maxlen=20)
        self.history.append((self.bbox, self.center))
        self.velocity = (0.0, 0.0)  # 프레임당 정규화 속도 (vx_per_frame, vy_per_frame)

        # 신호등 등 고정 객체 여부 (신호등은 허공으로 날아가지 않도록 속도 항상 0 고정)
        self.is_static = (
            self.label in ("traffic light", "traffic_light")
            or self.class_id == 9
        )

    def update(
        self,
        obj: ObjectBox,
        ema_alpha: float = 0.65,
        max_speed: float = 15.0,
    ):
        """새로 감지된 객체 박스와 매칭되었을 때 위치와 속도를 부드럽게 갱신."""
        dt = max(1, self.frames_since_detect)
        prev_cx, prev_cy = self.center
        curr_cx, curr_cy = obj.center

        if self.is_static:
            # 신호등 등 정적 객체는 속도 0 고정 (허공 날림 방지)
            self.velocity = (0.0, 0.0)
        else:
            # 1프레임당 실제 이동 속도 계산 (경과 프레임 dt로 정규화)
            raw_vx = (curr_cx - prev_cx) / float(dt)
            raw_vy = (curr_cy - prev_cy) / float(dt)

            # 급격한 노이즈나 오매칭으로 인한 이상 속도 폭주 방지 (클램핑)
            raw_vx = float(np.clip(raw_vx, -max_speed, max_speed))
            raw_vy = float(np.clip(raw_vy, -max_speed, max_speed))

            # 속도 평활화 (EMA)
            self.velocity = (
                0.35 * raw_vx + 0.65 * self.velocity[0],
                0.35 * raw_vy + 0.65 * self.velocity[1],
            )

        # 바운딩 박스 EMA 스무딩 (떨림/지터 방지 및 매끄러운 추종)
        det_bbox_f = np.array(obj.bbox, dtype=np.float32)
        self.bbox_f = ema_alpha * det_bbox_f + (1.0 - ema_alpha) * self.bbox_f
        self.bbox = tuple(int(round(v)) for v in self.bbox_f)
        self.center = (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

        self.confidence = 0.7 * float(obj.confidence) + 0.3 * self.confidence
        self.missed_frames = 0
        self.frames_since_detect = 0
        self.hits += 1
        self.history.append((self.bbox, self.center))

    def predict(self, velocity_decay: float = 0.88):
        """검출 건너뛰기/미탐지 프레임 동안 위치 보간 및 속도 감쇠 적용."""
        self.missed_frames += 1
        self.frames_since_detect += 1

        if self.is_static:
            # 신호등은 외삽 이동 없이 현재 위치 유지
            self.history.append((self.bbox, self.center))
            return

        # 미탐지 프레임이 누적될수록 감쇠를 강하게 하여 허공으로 날아가는 현상 원천 방지
        decay = velocity_decay if self.missed_frames <= 2 else (velocity_decay * 0.7)
        vx = self.velocity[0] * decay
        vy = self.velocity[1] * decay
        self.velocity = (vx, vy)

        # 보간 위치 이동
        self.bbox_f[0] += vx
        self.bbox_f[1] += vy
        self.bbox_f[2] += vx
        self.bbox_f[3] += vy

        self.bbox = tuple(int(round(v)) for v in self.bbox_f)
        self.center = (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )
        self.history.append((self.bbox, self.center))

    def clip_and_validate(self, frame_w: int, frame_h: int) -> bool:
        """박스가 화면 내부의 유효한 범위에 있는지 검증하고 경계 클리핑."""
        x1, y1, x2, y2 = self.bbox_f
        bw = x2 - x1
        bh = y2 - y1

        # 너무 작거나 음수인 박스 제거
        if bw < 8.0 or bh < 8.0:
            return False

        # 화면 밖으로 완전히 나간 박스 제거
        if x2 <= 0 or x1 >= frame_w or y2 <= 0 or y1 >= frame_h:
            return False

        # 차량/보행자가 하늘 상단 허공으로 솟구치는 비정상 상태 제거
        if not self.is_static and y2 < (frame_h * 0.15):
            return False

        # 화면 경계 클리핑
        cx1 = float(np.clip(x1, 0, frame_w - 1))
        cy1 = float(np.clip(y1, 0, frame_h - 1))
        cx2 = float(np.clip(x2, 0, frame_w - 1))
        cy2 = float(np.clip(y2, 0, frame_h - 1))

        if (cx2 - cx1) < 6.0 or (cy2 - cy1) < 6.0:
            return False

        self.bbox_f = np.array([cx1, cy1, cx2, cy2], dtype=np.float32)
        self.bbox = (int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2)))
        self.center = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
        return True


class ObjectTracker:
    """Maintains consistent IDs across frames and analyzes lead vehicle movement."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        tr_cfg = self.cfg.get("tracker", {})

        self.max_distance = float(tr_cfg.get("max_distance_px", 90.0))
        self.max_missed_frames = int(tr_cfg.get("max_missed_frames", 10))
        self.lead_margin = float(tr_cfg.get("lead_vehicle_corridor_margin_px", 130.0))
        self.ema_alpha = float(tr_cfg.get("ema_alpha", 0.65))
        self.velocity_decay = float(tr_cfg.get("velocity_decay", 0.88))
        self.max_speed = float(tr_cfg.get("max_speed_px", 15.0))

        self.tracks: Dict[int, TrackRecord] = {}
        self.next_id: int = 1

    def _is_class_compatible(self, track_label: str, det_label: str, track_cid: int, det_cid: int) -> bool:
        # 1. 신호등은 신호등끼리만 매칭
        is_tr_tl = (track_label in ("traffic light", "traffic_light") or track_cid == 9)
        is_det_tl = (det_label in ("traffic light", "traffic_light") or det_cid == 9)
        if is_tr_tl or is_det_tl:
            return is_tr_tl == is_det_tl

        # 2. 보행자는 보행자끼리만 매칭
        if track_label == "person" or det_label == "person" or track_cid == 0 or det_cid == 0:
            return (track_label == "person" or track_cid == 0) and (det_label == "person" or det_cid == 0)

        # 3. 차량류 (car, truck, bus 상호 매칭 허용)
        vehicle_set = {"car", "truck", "bus"}
        if track_label in vehicle_set and det_label in vehicle_set:
            return True

        # 그 외 (오토바이, 자전거 등) 동일 라벨
        return track_label == det_label or track_cid == det_cid

    def update(
        self,
        detections: List[ObjectBox],
        detection_executed: bool,
        lane: Optional[LaneResult],
        frame_shape: Tuple[int, int, int],
        road_markings: Optional[Any] = None
    ) -> List[ObjectBox]:
        """Matches detections to existing tracks and computes dynamics."""
        h, w = frame_shape[:2]
        lane_center = lane.lane_center_x if (lane and lane.lane_center_x is not None) else (w / 2.0)

        # 1. If detection was executed, match detections to tracks
        if detection_executed:
            matched_track_ids = set()
            unmatched_dets = []

            # 점수 기반 후보 페어링
            candidates = []
            for det_idx, det in enumerate(detections):
                for tid, track in self.tracks.items():
                    if not self._is_class_compatible(track.label, det.label, track.class_id, det.class_id):
                        continue

                    iou = calculate_iou(track.bbox, det.bbox)
                    dist = np.hypot(track.center[0] - det.center[0], track.center[1] - det.center[1])

                    # 신호등은 위치가 거의 고정이므로 엄격한 매칭
                    if track.is_static:
                        if iou > 0.10 or dist < 40.0:
                            score = iou * 3.0 + (1.0 - dist / 40.0)
                            candidates.append((score, tid, det_idx))
                    else:
                        if iou > 0.20:
                            score = iou * 2.0 + (1.0 - min(1.0, dist / self.max_distance))
                            candidates.append((score, tid, det_idx))
                        elif dist < self.max_distance:
                            # 박스 크기 비율 검사 (너무 다르면 오매칭 차단)
                            t_w = max(1, track.bbox[2] - track.bbox[0])
                            t_h = max(1, track.bbox[3] - track.bbox[1])
                            d_w = max(1, det.bbox[2] - det.bbox[0])
                            d_h = max(1, det.bbox[3] - det.bbox[1])
                            ratio = (t_w * t_h) / float(d_w * d_h)
                            if 0.30 <= ratio <= 3.3:
                                score = 1.0 - (dist / self.max_distance)
                                candidates.append((score, tid, det_idx))

            # 점수 높은 순으로 매칭 (Greedy Matching)
            candidates.sort(key=lambda x: x[0], reverse=True)
            matched_dets = set()

            for score, tid, det_idx in candidates:
                if tid in matched_track_ids or det_idx in matched_dets:
                    continue
                self.tracks[tid].update(
                    detections[det_idx],
                    ema_alpha=self.ema_alpha,
                    max_speed=self.max_speed,
                )
                matched_track_ids.add(tid)
                matched_dets.add(det_idx)

            for det_idx, det in enumerate(detections):
                if det_idx not in matched_dets:
                    unmatched_dets.append(det)

            # 매칭되지 않은 기존 트랙 예측 및 감쇠
            for tid in list(self.tracks.keys()):
                if tid not in matched_track_ids:
                    self.tracks[tid].predict(velocity_decay=self.velocity_decay)
                    if (
                        self.tracks[tid].missed_frames > self.max_missed_frames
                        or not self.tracks[tid].clip_and_validate(w, h)
                    ):
                        del self.tracks[tid]

            # 신규 트랙 등록
            for det in unmatched_dets:
                new_track = TrackRecord(self.next_id, det)
                if new_track.clip_and_validate(w, h):
                    self.tracks[self.next_id] = new_track
                    self.next_id += 1

        else:
            # Predict for all tracks during skipped frames
            for tid in list(self.tracks.keys()):
                self.tracks[tid].predict(velocity_decay=self.velocity_decay)
                if (
                    self.tracks[tid].missed_frames > self.max_missed_frames
                    or not self.tracks[tid].clip_and_validate(w, h)
                ):
                    del self.tracks[tid]

        # 2. Build tracked objects list and identify Lead Vehicle
        active_objects: List[ObjectBox] = []
        lead_candidate_id = None
        max_bottom_y = -1

        for tid, track in self.tracks.items():
            if not track.clip_and_validate(w, h):
                continue

            cx, cy = track.center
            x1, y1, x2, y2 = track.bbox

            # Evaluate distance trend over history
            trend = "STABLE"
            if len(track.history) >= 8:
                past_bbox, _ = track.history[0]
                past_area = max(1, (past_bbox[2] - past_bbox[0]) * (past_bbox[3] - past_bbox[1]))
                curr_area = max(1, (x2 - x1) * (y2 - y1))
                area_ratio = curr_area / float(past_area)

                if area_ratio > 1.15:
                    trend = "APPROACHING"
                elif area_ratio < 0.85:
                    trend = "RECEDING"

            # Determine motion status
            speed = np.hypot(track.velocity[0], track.velocity[1])
            status = "STOPPED" if speed < 1.0 else "MOVING"

            obj_box = ObjectBox(
                track_id=tid,
                bbox=track.bbox,
                class_id=track.class_id,
                label=track.label,
                confidence=track.confidence,
                center=track.center,
                velocity=track.velocity,
                is_lead=False,
                relative_distance_trend=trend,
                status=status
            )
            active_objects.append(obj_box)

            # Check if candidate for Lead Vehicle (in front of our lane)
            if track.label in ("car", "truck", "bus"):
                if abs(cx - lane_center) < self.lead_margin and y2 > (h * 0.50):
                    if y2 > max_bottom_y:
                        max_bottom_y = y2
                        lead_candidate_id = tid

        # Mark lead vehicle
        if lead_candidate_id is not None:
            for obj in active_objects:
                if obj.track_id == lead_candidate_id:
                    obj.is_lead = True

        # Mark pedestrians situated on/near detected crosswalks
        if road_markings and road_markings.crosswalk_detected:
            for obj in active_objects:
                if obj.label == "person" or obj.class_id == 0:
                    ox1, oy1, ox2, oy2 = obj.bbox
                    for cx1, cy1, cx2, cy2 in road_markings.crosswalk_boxes:
                        # Check horizontal and vertical bounding overlap
                        if (ox2 >= cx1 - 15 and ox1 <= cx2 + 15) and (oy2 >= cy1 - 30 and oy1 <= cy2 + 30):
                            obj.is_on_crosswalk = True
                            break

        return active_objects
