"""Lane tracking module for temporal smoothing and handling lane occlusion."""
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from ..common.data_models import LaneResult
from ..common.geometry import evaluate_poly_x


class LaneTracker:
    """Maintains temporal lane continuity, handles occlusion, and calculates ego lane center."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        lane_cfg = self.cfg.get("lane", {})
        roi_cfg = self.cfg.get("roi", {})

        self.alpha = float(lane_cfg.get("smoothing_alpha", 0.20))
        self.max_lost_frames = int(lane_cfg.get("max_lost_frames", 30))
        self.default_width_ratio = float(lane_cfg.get("default_lane_width_ratio", 0.45))
        self.min_confidence = float(lane_cfg.get("min_confidence", 0.25))

        self.top_ratio = float(roi_cfg.get("top_ratio", 0.65))
        self.bot_ratio = float(roi_cfg.get("bottom_ratio", 0.88))

        self.left_fit: Optional[List[float]] = None
        self.right_fit: Optional[List[float]] = None
        self.lost_frames: int = 0
        self.learned_lane_width: Optional[float] = None

    def update(
        self,
        curr_left_fit: Optional[List[float]],
        curr_right_fit: Optional[List[float]],
        raw_confidence: float,
        w: int,
        h: int
    ) -> LaneResult:
        """Updates lane models with smoothing, width estimation, and occlusion fail-safe."""
        top_y = int(h * self.top_ratio)
        bot_y = int(h * self.bot_ratio)

        # 1. Update Left Lane with EMA
        if curr_left_fit is not None:
            if self.left_fit is None:
                self.left_fit = list(curr_left_fit)
            else:
                self.left_fit = [
                    self.alpha * curr + (1.0 - self.alpha) * prev
                    for curr, prev in zip(curr_left_fit, self.left_fit)
                ]

        # 2. Update Right Lane with EMA
        if curr_right_fit is not None:
            if self.right_fit is None:
                self.right_fit = list(curr_right_fit)
            else:
                self.right_fit = [
                    self.alpha * curr + (1.0 - self.alpha) * prev
                    for curr, prev in zip(curr_right_fit, self.right_fit)
                ]

        # 3. Determine Lane Status & Manage Occlusion
        both_detected_now = (curr_left_fit is not None and curr_right_fit is not None)
        left_only_now = (curr_left_fit is not None and curr_right_fit is None)
        right_only_now = (curr_left_fit is None and curr_right_fit is not None)
        neither_detected_now = (curr_left_fit is None and curr_right_fit is None)

        status = "LOST"
        effective_conf = raw_confidence

        if both_detected_now:
            self.lost_frames = 0
            status = "BOTH_DETECTED"
            # Learn current pixel lane width at bottom
            lx_bot = evaluate_poly_x(self.left_fit, bot_y)
            rx_bot = evaluate_poly_x(self.right_fit, bot_y)
            if lx_bot is not None and rx_bot is not None and rx_bot > lx_bot:
                measured_width = rx_bot - lx_bot
                if self.learned_lane_width is None:
                    self.learned_lane_width = measured_width
                else:
                    self.learned_lane_width = 0.1 * measured_width + 0.9 * self.learned_lane_width

        elif left_only_now:
            self.lost_frames = 0
            status = "LEFT_ONLY"
            effective_conf = max(0.35, raw_confidence * 0.8)
            # If right fit is missing, estimate right lane using learned width
            if self.right_fit is None:
                width_px = self.learned_lane_width if self.learned_lane_width is not None else (w * self.default_width_ratio)
                # Parallel offset approximation: x_right = m * y + (c + width_px)
                if self.left_fit is not None:
                    self.right_fit = [self.left_fit[0], self.left_fit[1] + width_px]

        elif right_only_now:
            self.lost_frames = 0
            status = "RIGHT_ONLY"
            effective_conf = max(0.35, raw_confidence * 0.8)
            # If left fit is missing, estimate left lane using learned width
            if self.left_fit is None:
                width_px = self.learned_lane_width if self.learned_lane_width is not None else (w * self.default_width_ratio)
                if self.right_fit is not None:
                    self.left_fit = [self.right_fit[0], self.right_fit[1] - width_px]

        else: # Neither detected in current frame
            self.lost_frames += 1
            if self.lost_frames <= self.max_lost_frames and (self.left_fit is not None or self.right_fit is not None):
                status = "TEMPORARY_LOST"
                # Decay confidence gradually
                decay = max(0.1, 1.0 - (self.lost_frames / float(self.max_lost_frames)))
                effective_conf = 0.5 * decay
            else:
                status = "LOST"
                self.left_fit = None
                self.right_fit = None
                effective_conf = 0.0

        # 4. Generate Output Coordinates & Center Path
        left_line = None
        right_line = None
        center_path: List[Tuple[int, int]] = []
        lane_center_x = None
        lane_error_px = None
        current_width_px = None

        if self.left_fit is not None:
            lx_top = evaluate_poly_x(self.left_fit, top_y)
            lx_bot = evaluate_poly_x(self.left_fit, bot_y)
            if lx_top is not None and lx_bot is not None:
                left_line = (lx_top, top_y, lx_bot, bot_y)

        if self.right_fit is not None:
            rx_top = evaluate_poly_x(self.right_fit, top_y)
            rx_bot = evaluate_poly_x(self.right_fit, bot_y)
            if rx_top is not None and rx_bot is not None:
                right_line = (rx_top, top_y, rx_bot, bot_y)

        if left_line and right_line:
            lx_top, _, lx_bot, _ = left_line
            rx_top, _, rx_bot, _ = right_line
            if lx_bot > rx_bot:
                # Swap lines if detected reversed
                left_line, right_line = right_line, left_line
                self.left_fit, self.right_fit = self.right_fit, self.left_fit
                lx_top, _, lx_bot, _ = left_line
                rx_top, _, rx_bot, _ = right_line

            current_width_px = rx_bot - lx_bot
            lane_center_x = (lx_bot + rx_bot) / 2.0
            lane_error_px = lane_center_x - (w / 2.0)

            # Generate representative trajectory path
            steps = 6
            for i in range(steps + 1):
                y_val = top_y + (bot_y - top_y) * (i / float(steps))
                lx = evaluate_poly_x(self.left_fit, y_val)
                rx = evaluate_poly_x(self.right_fit, y_val)
                if lx is not None and rx is not None:
                    center_path.append((int((lx + rx) / 2), int(y_val)))

        is_uncertain = (status == "LOST" or effective_conf < self.min_confidence)

        return LaneResult(
            left_fit=self.left_fit,
            right_fit=self.right_fit,
            left_line=left_line,
            right_line=right_line,
            center_path=center_path,
            lane_center_x=lane_center_x,
            lane_error_px=lane_error_px,
            lane_confidence=effective_conf,
            status=status,
            lost_frames=self.lost_frames,
            is_uncertain=is_uncertain,
            lane_width_px=current_width_px
        )
