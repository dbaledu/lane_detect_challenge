"""Lane tracking module based on cv1021.py 4D Kalman Filter and temporal smoothing."""
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from ..common.data_models import LaneResult
from ..common.geometry import evaluate_poly_x


class SingleLaneKalmanFilter:
    """
    개별 차선(1차 직선 x = m*y + c) 추적을 위한 4차원 칼만 필터 (cv1021.py):
    - 상태 벡터 x: [m, c, dm, dc]^T (기울기, 절편, 기울기 변화율, 절편 변화율)
    - 관측 벡터 z: [m, c]^T
    - 등속도(Constant Velocity) 물리 모델을 적용하여 프레임 간 떨림(Jitter)을 완벽히 제거하고,
      보행자나 전방 차량 가림 시에도 직전 속도 벡터를 기반으로 차선 위치를 정밀하게 예측(Predict)
    """
    def __init__(self, dt: float = 1.0, max_lost_frames: int = 30):
        self.dt = dt
        self.max_lost_frames = max_lost_frames
        self.kf = cv2.KalmanFilter(4, 2)

        # 1. 상태 전이 행렬 (F: 등속도 운동 모델)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 2. 관측 행렬 (H: m과 c를 직접 관측)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 3. 프로세스 노이즈 공분산 행렬 (Q: 시스템 모델의 불확실성)
        self.kf.processNoiseCov = np.array([
            [1e-4, 0, 0, 0],
            [0, 1e-1, 0, 0],
            [0, 0, 1e-4, 0],
            [0, 0, 0, 1e-1]
        ], dtype=np.float32)

        # 4. 측정 노이즈 공분산 행렬 (R: 허프 변환 검출 노이즈)
        self.kf.measurementNoiseCov = np.array([
            [5e-3, 0],
            [0, 5.0]
        ], dtype=np.float32)

        # 5. 오차 공분산 행렬 (P)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.is_initialized = False
        self.lost_frames = 0
        self.current_fit: Optional[List[float]] = None

    def reset(self):
        """필터 상태 완전 리셋."""
        self.is_initialized = False
        self.lost_frames = 0
        self.current_fit = None
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

    def update(self, measurement_fit: Optional[List[float]]) -> Tuple[Optional[List[float]], int]:
        """측정값([m, c])을 바탕으로 칼만 필터 예측(Predict) 및 갱신(Correct) 수행."""
        # 1. 등속도 모델 기반 다음 상태 예측
        pred = self.kf.predict()
        pred_m = float(pred[0, 0])
        pred_c = float(pred[1, 0])

        if measurement_fit is not None:
            if len(measurement_fit) >= 3:
                y1, y2 = 350.0, 480.0
                x1 = measurement_fit[0] * (y1 ** 2) + measurement_fit[1] * y1 + measurement_fit[2]
                x2 = measurement_fit[0] * (y2 ** 2) + measurement_fit[1] * y2 + measurement_fit[2]
                m_meas = (x2 - x1) / (y2 - y1)
                c_meas = x1 - m_meas * y1
            else:
                m_meas, c_meas = float(measurement_fit[0]), float(measurement_fit[1])

            if not self.is_initialized:
                # 첫 측정값으로 상태 벡터 초기화
                self.kf.statePost = np.array([[m_meas], [c_meas], [0.0], [0.0]], dtype=np.float32)
                self.is_initialized = True
                self.lost_frames = 0
                self.current_fit = [m_meas, c_meas]
            else:
                # 횡단보도/노면 마킹 등 비정상적인 급격한 점프(이상치) 필터링 (Innovation Gating)
                diff_m = abs(m_meas - pred_m)
                diff_c = abs(c_meas - pred_c)

                if diff_m > 0.45 or diff_c > 160.0:
                    # 이상치 관측 시: 관측값 배제하고 칼만 예측값 유지
                    self.lost_frames += 1
                    self.current_fit = [pred_m, pred_c]
                else:
                    # 정상 관측 시: 칼만 보정(Correct)
                    z = np.array([[m_meas], [c_meas]], dtype=np.float32)
                    estimated = self.kf.correct(z)
                    self.lost_frames = 0
                    self.current_fit = [float(estimated[0, 0]), float(estimated[1, 0])]
        else:
            # 측정 결측 시 (가림/교차로/횡단보도 등): 예측 상태 유지
            if self.is_initialized:
                self.lost_frames += 1
                if self.lost_frames <= self.max_lost_frames:
                    self.current_fit = [pred_m, pred_c]
                else:
                    self.reset()
            else:
                self.current_fit = None

        return self.current_fit, self.lost_frames


class LaneTracker:
    """좌/우 차선 각각에 독립된 칼만 필터(cv1021.py)를 탑재한 시계열 차선 추적기."""

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        dt: float = 1.0,
        max_lost_frames: Optional[int] = None
    ):
        self.cfg = cfg or {}
        lane_cfg = self.cfg.get("lane", {})
        roi_cfg = self.cfg.get("roi", {})

        self.max_lost_frames = max_lost_frames or int(lane_cfg.get("max_lost_frames", 30))
        self.default_width_ratio = float(lane_cfg.get("default_lane_width_ratio", 0.45))
        self.min_confidence = float(lane_cfg.get("min_confidence", 0.25))

        self.top_ratio = float(roi_cfg.get("top_ratio", 0.67))
        self.bot_ratio = float(roi_cfg.get("bottom_ratio", 0.86))

        self.left_kf = SingleLaneKalmanFilter(dt=dt, max_lost_frames=self.max_lost_frames)
        self.right_kf = SingleLaneKalmanFilter(dt=dt, max_lost_frames=self.max_lost_frames)
        self.lost_frames: int = 0
        self.learned_lane_width: Optional[float] = None

    def reset(self):
        """모든 칼만 필터 및 추적 상태 초기화."""
        self.left_kf.reset()
        self.right_kf.reset()
        self.lost_frames = 0
        self.learned_lane_width = None

    def update(
        self,
        curr_left_fit: Optional[List[float]] = None,
        curr_right_fit: Optional[List[float]] = None,
        raw_confidence: float = 1.0,
        w: Optional[int] = None,
        h: Optional[int] = None,
        is_crosswalk: bool = False,
        debug_info: Optional[Dict[str, Any]] = None
    ) -> LaneResult:
        """칼만 필터 기반 좌/우 차선 상태 예측 및 갱신 수행 후 LaneResult 반환."""
        frame_w = w or 960
        frame_h = h or 540

        top_y = int(frame_h * self.top_ratio)
        bot_y = int(frame_h * self.bot_ratio)

        # 횡단보도 구간에서는 가로선 오검출을 방지하기 위해 신규 측정을 배제하고 칼만 예측 유지
        if is_crosswalk:
            meas_left = None
            meas_right = None
        else:
            meas_left = curr_left_fit
            meas_right = curr_right_fit

        # 1. 칼만 필터 업데이트
        left_fit, left_lost = self.left_kf.update(meas_left)
        right_fit, right_lost = self.right_kf.update(meas_right)

        # 둘 다 미검출된 경우 누적 프레임 카운트
        if meas_left is None and meas_right is None:
            self.lost_frames += 1
        else:
            self.lost_frames = 0

        # 2. 양쪽 차선이 모두 측정된 경우 차로 폭(lane width) 학습
        both_measured_now = (meas_left is not None and meas_right is not None)
        left_only_now = (meas_left is not None and meas_right is None)
        right_only_now = (meas_left is None and meas_right is not None)

        if both_measured_now and left_fit is not None and right_fit is not None:
            lx_bot = evaluate_poly_x(left_fit, bot_y)
            rx_bot = evaluate_poly_x(right_fit, bot_y)
            if lx_bot is not None and rx_bot is not None and rx_bot > lx_bot:
                measured_width = rx_bot - lx_bot
                if self.learned_lane_width is None:
                    self.learned_lane_width = measured_width
                else:
                    self.learned_lane_width = 0.1 * measured_width + 0.9 * self.learned_lane_width

        # 3. 편측만 검출되고 반대편이 아직 초기화되지 않은 경우 학습된 차로폭으로 보정
        if left_only_now and right_fit is None and left_fit is not None:
            width_px = self.learned_lane_width if self.learned_lane_width is not None else (frame_w * self.default_width_ratio)
            synth_right = [left_fit[0], left_fit[1] + width_px]
            right_fit, right_lost = self.right_kf.update(synth_right)

        elif right_only_now and left_fit is None and right_fit is not None:
            width_px = self.learned_lane_width if self.learned_lane_width is not None else (frame_w * self.default_width_ratio)
            synth_left = [right_fit[0], right_fit[1] - width_px]
            left_fit, left_lost = self.left_kf.update(synth_left)

        # 4. 상태 판별
        has_left = left_fit is not None
        has_right = right_fit is not None

        if is_crosswalk:
            status = "CROSSWALK_DETECTED"
            effective_conf = 0.50
        elif both_measured_now or (has_left and has_right and self.lost_frames == 0):
            status = "BOTH_DETECTED"
            effective_conf = raw_confidence
        elif left_only_now or (has_left and not has_right and self.lost_frames == 0):
            status = "LEFT_ONLY"
            effective_conf = max(0.35, raw_confidence * 0.8)
        elif right_only_now or (has_right and not has_left and self.lost_frames == 0):
            status = "RIGHT_ONLY"
            effective_conf = max(0.35, raw_confidence * 0.8)
        elif (has_left or has_right) and self.lost_frames <= self.max_lost_frames:
            status = "TEMPORARY_LOST"
            decay = max(0.1, 1.0 - (self.lost_frames / float(self.max_lost_frames)))
            effective_conf = 0.5 * decay
        else:
            status = "LOST"
            effective_conf = 0.0
            left_fit = None
            right_fit = None

        # 5. 좌표 및 중심 경로 계산
        left_line: Optional[Tuple[int, int, int, int]] = None
        right_line: Optional[Tuple[int, int, int, int]] = None
        center_path: List[Tuple[int, int]] = []
        lane_center_x: Optional[float] = None
        lane_error_px: Optional[float] = None
        current_width_px: Optional[float] = None

        if left_fit is not None:
            lx_top = evaluate_poly_x(left_fit, top_y)
            lx_bot = evaluate_poly_x(left_fit, bot_y)
            if lx_top is not None and lx_bot is not None:
                left_line = (lx_top, top_y, lx_bot, bot_y)

        if right_fit is not None:
            rx_top = evaluate_poly_x(right_fit, top_y)
            rx_bot = evaluate_poly_x(right_fit, bot_y)
            if rx_top is not None and rx_bot is not None:
                right_line = (rx_top, top_y, rx_bot, bot_y)

        if left_line is not None and right_line is not None:
            lx_top, _, lx_bot, _ = left_line
            rx_top, _, rx_bot, _ = right_line
            if lx_bot > rx_bot:
                # 좌우가 반대로 인식된 경우 교환
                left_line, right_line = right_line, left_line
                left_fit, right_fit = right_fit, left_fit
                lx_top, _, lx_bot, _ = left_line
                rx_top, _, rx_bot, _ = right_line

            current_width_px = float(rx_bot - lx_bot)
            lane_center_x = (lx_bot + rx_bot) / 2.0
            lane_error_px = lane_center_x - (frame_w / 2.0)

            # 주행 중심 경로 (6단계 보간)
            steps = 6
            for i in range(steps + 1):
                y_val = top_y + (bot_y - top_y) * (i / float(steps))
                lx = evaluate_poly_x(left_fit, y_val)
                rx = evaluate_poly_x(right_fit, y_val)
                if lx is not None and rx is not None:
                    center_path.append((int((lx + rx) / 2), int(y_val)))

        is_uncertain = (status == "LOST" or effective_conf < self.min_confidence)

        # 디버그 정보에서 ROI 지오메트리 추출
        left_corners = debug_info.get("left_corners") if debug_info else None
        right_corners = debug_info.get("right_corners") if debug_info else None
        left_bbox = debug_info.get("left_bbox") if debug_info else None
        right_bbox = debug_info.get("right_bbox") if debug_info else None

        return LaneResult(
            left_fit=left_fit,
            right_fit=right_fit,
            left_line=left_line,
            right_line=right_line,
            center_path=center_path,
            lane_center_x=lane_center_x,
            lane_error_px=lane_error_px,
            lane_confidence=effective_conf,
            status=status,
            lost_frames=self.lost_frames,
            is_uncertain=is_uncertain,
            lane_width_px=current_width_px,
            left_corners=left_corners,
            right_corners=right_corners,
            left_bbox=left_bbox,
            right_bbox=right_bbox,
            is_crosswalk=is_crosswalk
        )
