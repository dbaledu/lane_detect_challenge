"""Driving state classifier implementing time-window state machine."""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque
from ..common.data_models import MotionResult, QualityMetric


class StateClassifier:
    """Classifies ego-vehicle driving state (Driving, Decelerating, Stopped, Starting) using hysteresis."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        st_cfg = self.cfg.get("state", {})

        self.stopped_th = float(st_cfg.get("flow_stopped_threshold", 0.85))
        self.decel_slope = float(st_cfg.get("flow_decel_slope", -0.04))
        self.start_th = float(st_cfg.get("flow_start_threshold", 1.25))
        self.window_size = int(st_cfg.get("window_size_frames", 25))
        self.persistence = int(st_cfg.get("stopped_persistence_frames", 15))

        self.motion_history = deque(maxlen=self.window_size)
        self.current_state = "NORMAL_DRIVING"
        self.low_motion_frames = 0
        self.frames_in_state = 0
        self.smoothed_mag: Optional[float] = None

    def update(self, motion: MotionResult, quality: QualityMetric) -> Tuple[str, str, float]:
        """Updates state machine with current motion and quality metrics.

        Returns:
            ego_state: "NORMAL_DRIVING", "DECELERATION_SUSPECTED", "STOPPED_SUSPECTED", "START_DETECTED", "UNCERTAIN"
            reason: explanatory rationale
            confidence: decision confidence (0.0 ~ 1.0)
        """
        self.frames_in_state += 1

        # Check for unreliability (extreme blur, occlusion)
        if not motion.is_reliable or quality.is_blurred:
            return "UNCERTAIN", "영상 흔들림 또는 특징점 소실로 모션 측정 신뢰도 낮음", 0.30

        # Smooth motion magnitude to suppress instantaneous asphalt/camera jitter
        if self.smoothed_mag is None:
            self.smoothed_mag = motion.magnitude
        else:
            self.smoothed_mag = 0.40 * motion.magnitude + 0.60 * self.smoothed_mag
        mag = self.smoothed_mag

        self.motion_history.append(mag)

        # Track consecutive low-motion frames
        if mag < self.stopped_th:
            self.low_motion_frames += 1
        else:
            self.low_motion_frames = max(0, self.low_motion_frames - 2)

        # State transition logic
        new_state = self.current_state
        reason = "주행 상태 유지"
        confidence = 0.85

        # 1. Start Detected: was stopped, now motion surges
        if self.current_state == "STOPPED_SUSPECTED" and mag >= self.start_th:
            new_state = "START_DETECTED"
            reason = f"정지 후 배경 이동량({mag:.2f}px) 급증하여 출발 감지"
            confidence = 0.88
            self.low_motion_frames = 0

        # 2. Stopped Suspected: sustained low motion
        elif self.low_motion_frames >= self.persistence:
            new_state = "STOPPED_SUSPECTED"
            reason = f"배경 이동량({mag:.2f}px)이 기준({self.stopped_th}) 미만으로 {self.low_motion_frames}프레임 지속"
            confidence = min(0.95, 0.70 + (self.low_motion_frames / 40.0))

        # 3. Deceleration Suspected: motion steadily decreasing across time window
        elif len(self.motion_history) >= 15:
            # Linear regression slope over recent motion history
            y = np.array(self.motion_history)
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)

            if slope < self.decel_slope and np.mean(y[-5:]) < np.mean(y[:5]) * 0.75:
                new_state = "DECELERATION_SUSPECTED"
                reason = f"시간 창({len(y)}f) 내 이동량 감소 추세 감지 (기울기: {slope:.3f})"
                confidence = 0.78
            elif self.current_state == "START_DETECTED" and self.frames_in_state > 12:
                new_state = "NORMAL_DRIVING"
                reason = "출발 후 정상 주행 흐름으로 안착"
                confidence = 0.85
            elif mag >= self.stopped_th and slope >= self.decel_slope:
                new_state = "NORMAL_DRIVING"
                reason = f"유의미한 배경 이동량({mag:.2f}px) 유지"
                confidence = 0.88
        else:
            new_state = "NORMAL_DRIVING"
            reason = "초기 프레임 모션 누적 중"
            confidence = 0.60

        if new_state != self.current_state:
            self.current_state = new_state
            self.frames_in_state = 0

        return self.current_state, reason, confidence
