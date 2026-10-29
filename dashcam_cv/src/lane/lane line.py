#!/usr/bin/env python3
"""run_antigravity.py

고안정성 자율주행 차선 및 횡단보도 추적기 (High-Stability Lane & Crosswalk Tracker)
=============================================================================
[해결된 5대 핵심 문제]
1. (m, c) 파라미터 시소 현상 완전 제거 -> 화면 좌표계(x_top, x_bottom) 직접 스무딩
2. 이상치 급격한 점프 방지 -> Slew-rate(최대 이동 속도) 클램핑 적용
3. 점선(Dashed) 구간 깜빡임 방지 -> 지수 감쇠 메모리 홀드(Graceful Decay Hold)
4. 곡선 구간 ROI 클리핑 방지 -> 차선 위치 기반 동적 마진(Adaptive ROI Padding)
5. 화면 바닥(100%) 무리한 외삽 널뛰기 방지 -> 유효 노면 영역(0.67h ~ 0.88h) 정밀 렌더링
6. 횡단보도 감지 시 즉각적인 차선/ROI 억제 및 안전 정지 배너 표출
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any


# ── 안전 기하 연산 함수 ────────────────────────────────────────────────────────
try:
    from ..common.geometry import compute_slope, evaluate_poly_x
except Exception:
    def compute_slope(x1: float, y1: float, x2: float, y2: float) -> Optional[float]:
        if abs(x2 - x1) < 1e-6:
            return None
        return (y2 - y1) / (x2 - x1)

    def evaluate_poly_x(fit: List[float], y: float) -> float:
        if fit is None or len(fit) < 2:
            return 0.0
        if len(fit) == 2:
            return fit[0] * y + fit[1]
        elif len(fit) == 3:
            return fit[0] * (y ** 2) + fit[1] * y + fit[2]
        return 0.0


# ── 듀얼 ROI 및 마스킹 기하학 ──────────────────────────────────────────────────
def get_dual_roi_geometry(
    h: int,
    w: int,
    top_ratio: float = 0.67,
    bot_ratio: float = 0.88,
    padding_x: int = 24,
    prev_left_pts: Optional[Tuple[float, float]] = None,
    prev_right_pts: Optional[Tuple[float, float]] = None,
) -> Tuple[Tuple[np.ndarray, Tuple[int, int, int, int]], Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """곡선 및 차선 이탈에도 잘리지 않도록 동적 마진(Adaptive Margin)을 포함한 듀얼 ROI 산출."""
    top_y = int(h * top_ratio)
    bot_y = int(h * bot_ratio)

    # 1. 좌측 차선 슬림 사다리꼴 (이전 추적 위치가 있으면 동적 추종, 없으면 기본 기하학)
    if prev_left_pts is not None:
        px_top, px_bot = prev_left_pts
        l_tl_x = max(0, int(px_top - padding_x * 1.5))
        l_tr_x = min(w // 2, int(px_top + padding_x * 1.5))
        l_bl_x = max(0, int(px_bot - padding_x * 2.0))
        l_br_x = min(w // 2 + 50, int(px_bot + padding_x * 1.5))
    else:
        l_tl_x = int(w * 0.36)
        l_tr_x = int(w * 0.47)
        l_br_x = int(w * 0.42)
        l_bl_x = int(w * 0.24)

    left_tl = (l_tl_x, top_y)
    left_tr = (l_tr_x, top_y)
    left_br = (l_br_x, bot_y)
    left_bl = (l_bl_x, bot_y)
    left_corners = np.array([[left_tl, left_tr, left_br, left_bl]], dtype=np.int32)

    l_xs = [left_tl[0], left_tr[0], left_br[0], left_bl[0]]
    left_xmin = max(0, min(l_xs) - 6)
    left_xmax = min(w, max(l_xs) + 6)
    left_bbox = (left_xmin, top_y, left_xmax, bot_y)

    # 2. 우측 차선 슬림 사다리꼴
    if prev_right_pts is not None:
        px_top, px_bot = prev_right_pts
        r_tl_x = max(w // 2 - 50, int(px_top - padding_x * 1.5))
        r_tr_x = min(w, int(px_top + padding_x * 1.5))
        r_bl_x = max(w // 2, int(px_bot - padding_x * 1.5))
        r_br_x = min(w, int(px_bot + padding_x * 2.0))
    else:
        r_tl_x = int(w * 0.53)
        r_tr_x = int(w * 0.64)
        r_br_x = int(w * 0.76)
        r_bl_x = int(w * 0.58)

    right_tl = (r_tl_x, top_y)
    right_tr = (r_tr_x, top_y)
    right_br = (r_br_x, bot_y)
    right_bl = (r_bl_x, bot_y)
    right_corners = np.array([[right_tl, right_tr, right_br, right_bl]], dtype=np.int32)

    r_xs = [right_tl[0], right_tr[0], right_br[0], right_bl[0]]
    right_xmin = max(0, min(r_xs) - 6)
    right_xmax = min(w, max(r_xs) + 6)
    right_bbox = (right_xmin, top_y, right_xmax, bot_y)

    return (left_corners, left_bbox), (right_corners, right_bbox)


def crop_and_resize_roi(
    frame: np.ndarray,
    roi_corners: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Tuple[int, int, float, float]]:
    """Bounding Box 영역 슬라이싱 후 사다리꼴 마스크 적용."""
    h_img, w_img = frame.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    xmin = max(0, min(xmin, w_img - 1))
    xmax = max(xmin + 1, min(xmax, w_img))
    ymin = max(0, min(ymin, h_img - 1))
    ymax = max(ymin + 1, min(ymax, h_img))

    crop_w = xmax - xmin
    crop_h = ymax - ymin

    cropped = frame[ymin:ymax, xmin:xmax]

    rel_corners = roi_corners.copy()
    rel_corners[0, :, 0] = np.clip(rel_corners[0, :, 0] - xmin, 0, crop_w - 1)
    rel_corners[0, :, 1] = np.clip(rel_corners[0, :, 1] - ymin, 0, crop_h - 1)

    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(mask, rel_corners, 255)
    masked_crop = cv2.bitwise_and(cropped, cropped, mask=mask)

    if target_size is not None and (crop_w != target_size[0] or crop_h != target_size[1]):
        target_w, target_h = target_size
        resized_roi = cv2.resize(masked_crop, target_size, interpolation=cv2.INTER_AREA)
        scale_x = crop_w / float(target_w)
        scale_y = crop_h / float(target_h)
    else:
        resized_roi = masked_crop
        scale_x = 1.0
        scale_y = 1.0

    return resized_roi, (xmin, ymin, scale_x, scale_y)


def preprocess_roi(roi_img: np.ndarray, clahe_obj: Optional[cv2.CLAHE] = None) -> np.ndarray:
    """노면 조명 변화에 강인한 적응형 히스토그램 평활화(CLAHE) 및 Canny 에지 추출."""
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    if clahe_obj is not None:
        enhanced = clahe_obj.apply(gray)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 130)
    return edges


def scale_lines_to_original(
    lines: Optional[np.ndarray],
    transform_params: Tuple[int, int, float, float]
) -> Optional[np.ndarray]:
    """로컬 ROI 좌표계의 선분들을 원본 프레임 좌표계로 역투영."""
    if lines is None or len(lines) == 0:
        return None

    xmin, ymin, scale_x, scale_y = transform_params
    orig_lines = []

    for line in lines:
        pts = line.flatten()
        if len(pts) < 4:
            continue
        x1, y1, x2, y2 = pts[:4]
        ox1 = int(round(x1 * scale_x + xmin))
        oy1 = int(round(y1 * scale_y + ymin))
        ox2 = int(round(x2 * scale_x + xmin))
        oy2 = int(round(y2 * scale_y + ymin))
        orig_lines.append([[ox1, oy1, ox2, oy2]])

    if len(orig_lines) == 0:
        return None

    return np.array(orig_lines, dtype=np.int32)


# ── 강건한 차선 피팅 (WLS + Outlier Rejection) ──────────────────────────────────
def fit_lane_line_robust(
    lines: Optional[np.ndarray],
    is_left: bool = True,
    min_slope: float = 0.40,
    max_slope: float = 3.00,
    min_line_count: int = 2,
    min_total_length: float = 30.0,
    min_y_span_ratio: float = 0.20,
    roi_h: int = 100
) -> Tuple[Optional[List[float]], Dict[str, Any]]:
    """가중 최소자승법(WLS)과 잔차 필터링을 결합한 1차 차선 방정식 피팅."""
    metrics = {
        "line_count": 0,
        "total_length": 0.0,
        "y_span": 0,
        "rmse": 999.0,
        "valid": False
    }

    if lines is None or len(lines) == 0:
        return None, metrics

    pts_x = []
    pts_y = []
    weights = []
    valid_lines = 0
    total_length = 0.0

    for line in lines:
        pts = line.flatten()
        if len(pts) < 4:
            continue
        x1, y1, x2, y2 = pts[:4]
        if x1 == x2:
            continue

        slope = (y2 - y1) / float(x2 - x1)
        abs_slope = abs(slope)

        if abs_slope < min_slope or abs_slope > max_slope:
            continue

        line_len = float(np.hypot(x2 - x1, y2 - y1))

        # 좌측 차선(음의 기울기) / 우측 차선(양의 기울기)
        if is_left and slope <= -min_slope:
            pts_x.extend([x1, x2])
            pts_y.extend([y1, y2])
            weights.extend([line_len, line_len])
            total_length += line_len
            valid_lines += 1
        elif not is_left and slope >= min_slope:
            pts_x.extend([x1, x2])
            pts_y.extend([y1, y2])
            weights.extend([line_len, line_len])
            total_length += line_len
            valid_lines += 1

    metrics["line_count"] = valid_lines
    metrics["total_length"] = total_length

    if valid_lines < min_line_count or total_length < min_total_length or len(pts_x) < 4:
        return None, metrics

    y_min, y_max = min(pts_y), max(pts_y)
    y_span = y_max - y_min
    metrics["y_span"] = y_span

    if y_span < int(roi_h * min_y_span_ratio):
        return None, metrics

    try:
        w_arr = np.array(weights, dtype=np.float32)
        fit = np.polyfit(pts_y, pts_x, 1, w=w_arr)

        pred_x = fit[0] * np.array(pts_y) + fit[1]
        residuals = np.abs(pred_x - np.array(pts_x))
        inliers_mask = residuals < 20.0  # 잔차 임계값 20px

        if np.sum(inliers_mask) >= 4:
            clean_y = np.array(pts_y)[inliers_mask]
            clean_x = np.array(pts_x)[inliers_mask]
            clean_w = w_arr[inliers_mask]
            refined_fit = np.polyfit(clean_y, clean_x, 1, w=clean_w)
            pred_x_clean = refined_fit[0] * clean_y + refined_fit[1]
            rmse = float(np.sqrt(np.mean((pred_x_clean - clean_x) ** 2)))
            metrics["rmse"] = rmse
            metrics["valid"] = True
            return list(refined_fit), metrics
        else:
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            metrics["rmse"] = rmse
            metrics["valid"] = True
            return list(fit), metrics
    except Exception:
        return None, metrics


# ── 횡단보도 검출 함수 ────────────────────────────────────────────────────────
def detect_crosswalk(
    frame: np.ndarray,
    top_ratio: float = 0.65,
    bot_ratio: float = 0.88,
    left_ratio: float = 0.22,
    right_ratio: float = 0.78,
    thresh_val: int = 175,
    clahe_obj: Optional[cv2.CLAHE] = None
) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
    """해상도 스케일링을 고려한 횡단보도(Zebra Stripe Pattern) 검출."""
    h, w = frame.shape[:2]
    scale = w / 640.0
    y1 = int(h * top_ratio)
    y2 = int(h * bot_ratio)
    x1 = int(w * left_ratio)
    x2 = int(w * right_ratio)

    road = frame[y1:y2, x1:x2]
    if road.size == 0:
        return False, []

    gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    if clahe_obj is not None:
        enhanced = clahe_obj.apply(gray)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

    _, thresh = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)
    k_w = max(5, int(15 * scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripes = 0
    boxes = []
    stripe_y_coords = []

    min_w = int(24 * scale)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (80 * scale) or area > (14000 * scale * scale):
            continue
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        aspect = rw / float(max(1, rh))
        if 1.8 <= aspect <= 16.0 and rw >= min_w:
            stripes += 1
            boxes.append((x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh))
            stripe_y_coords.append(y1 + ry)

    is_crosswalk = stripes >= 3
    if is_crosswalk and len(stripe_y_coords) >= 3:
        y_span = max(stripe_y_coords) - min(stripe_y_coords)
        if y_span < int((y2 - y1) * 0.20):
            is_crosswalk = False

    return is_crosswalk, boxes


# ── [핵심 클래스] 고안정성 차선 추적 엔진 ─────────────────────────────────────
class HighStabilityLaneTracker:
    """좌표 공간 평활화, Slew-Rate 제한, 동적 ROI를 갖춘 무흔들림 차선 추적기."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        lane_cfg = self.cfg.get("lane", {})
        roi_cfg = self.cfg.get("roi", {})
        cw_cfg = self.cfg.get("crosswalk", {})

        # 기하학 파라미터
        self.top_ratio = float(roi_cfg.get("top_ratio", 0.67))
        self.bot_ratio = float(roi_cfg.get("bottom_ratio", 0.88))
        self.padding_x = int(roi_cfg.get("padding_x", 24))

        # 허프 파라미터
        self.hough_threshold = int(lane_cfg.get("hough_threshold", 22))
        self.min_line_length = int(lane_cfg.get("min_line_length", 18))
        self.max_line_gap = int(lane_cfg.get("max_line_gap", 45))
        self.min_slope = float(lane_cfg.get("min_slope", 0.40))
        self.max_slope = float(lane_cfg.get("max_slope", 3.00))

        # -------------------------------------------------------------
        # [핵심 파라미터 1] 좌표계 기반 EMA 평활화 계수 (0.18: 부드러운 유도)
        # -------------------------------------------------------------
        self.coord_alpha = float(lane_cfg.get("coord_alpha", 0.18))

        # -------------------------------------------------------------
        # [핵심 파라미터 2] Slew-Rate 제한 (프레임당 최대 이동 픽셀 클램핑)
        # 30fps 기준 급격한 노이즈로 차선이 수십 픽셀 순간 이동하는 것을 방지
        # -------------------------------------------------------------
        self.max_slew_top = float(lane_cfg.get("max_slew_top", 12.0))   # 상단 최대 12px/프레임
        self.max_slew_bot = float(lane_cfg.get("max_slew_bot", 18.0))   # 하단 최대 18px/프레임

        # -------------------------------------------------------------
        # [핵심 파라미터 3] 감쇠 홀드 (Graceful Decay Hold)
        # 점선 구간에서 검출이 순간 끊겨도 최대 12프레임 동안 부드럽게 지탱
        # -------------------------------------------------------------
        self.max_hold_frames = int(lane_cfg.get("max_hold_frames", 12))
        self.decay_factor = float(lane_cfg.get("decay_factor", 0.94))

        # 좌표 공간 상태 변수 (x_top, x_bot)
        self.smooth_left_pts: Optional[Tuple[float, float]] = None
        self.smooth_right_pts: Optional[Tuple[float, float]] = None
        self.lost_left_count: int = 0
        self.lost_right_count: int = 0
        self.left_confidence: float = 0.0
        self.right_confidence: float = 0.0

        # 횡단보도 상태 머신 (히스테리시스)
        self.cw_detect_count = 0
        self.is_crosswalk_state = False
        self.cw_hysteresis_enter = int(cw_cfg.get("hysteresis_enter", 2))
        self.cw_hysteresis_exit = int(cw_cfg.get("hysteresis_exit", 2))
        self.cw_thresh = int(cw_cfg.get("threshold", 175))

        # CLAHE 객체 사전 할당
        self._clahe_left = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._clahe_right = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._clahe_cw = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect_crosswalk(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
        raw_detected, boxes = detect_crosswalk(
            frame,
            top_ratio=0.65,
            bot_ratio=0.88,
            thresh_val=self.cw_thresh,
            clahe_obj=self._clahe_cw
        )

        if raw_detected:
            self.cw_detect_count = min(self.cw_hysteresis_enter, self.cw_detect_count + 1)
            if self.cw_detect_count >= self.cw_hysteresis_enter:
                self.is_crosswalk_state = True
        else:
            self.cw_detect_count = max(0, self.cw_detect_count - 1)
            if self.cw_detect_count == 0:
                self.is_crosswalk_state = False

        return self.is_crosswalk_state, boxes

    def _detect_single_roi(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        bbox: Tuple[int, int, int, int],
        clahe_obj: cv2.CLAHE,
        is_left: bool,
        cw_boxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[Optional[List[float]], Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
        h, w = frame.shape[:2]
        roi_img, params = crop_and_resize_roi(frame, corners, bbox)
        edges = preprocess_roi(roi_img, clahe_obj=clahe_obj)

        xmin, ymin, scale_x, scale_y = params
        if cw_boxes and len(cw_boxes) > 0:
            for bx1, by1, bx2, by2 in cw_boxes:
                lx1 = int(round(max(0, bx1 - xmin) / scale_x))
                lx2 = int(round(max(0, bx2 - xmin) / scale_x))
                ly1 = int(round(max(0, by1 - ymin) / scale_y))
                ly2 = int(round(max(0, by2 - ymin) / scale_y))
                eh, ew = edges.shape[:2]
                if lx1 < ew and ly1 < eh and lx2 > 0 and ly2 > 0:
                    edges[max(0, ly1):min(eh, ly2), max(0, lx1):min(ew, lx2)] = 0

        lines_roi = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        lines_orig = scale_lines_to_original(lines_roi, params)

        fit, metrics = fit_lane_line_robust(
            lines_orig,
            is_left=is_left,
            min_slope=self.min_slope,
            max_slope=self.max_slope,
            roi_h=int(h * (self.bot_ratio - self.top_ratio))
        )

        return fit, lines_orig, edges, metrics

    # -------------------------------------------------------------------------
    # [핵심 알고리즘] 좌표 공간(Coordinate-Space) 평활화 및 Slew-Rate 제한
    # -------------------------------------------------------------------------
    def _smooth_lane_coordinates(
        self,
        meas_fit: Optional[List[float]],
        prev_pts: Optional[Tuple[float, float]],
        lost_count: int,
        prev_conf: float,
        y_top: float,
        y_bot: float
    ) -> Tuple[Optional[Tuple[float, float]], int, float]:
        """기울기/절편 대신 실제 노면 상단/하단 x좌표를 직접 평활화하여 시소 흔들림 방지."""
        if meas_fit is not None:
            # 1. 측정된 직선을 y_top, y_bot에서의 픽셀 좌표로 환산
            meas_xt = evaluate_poly_x(meas_fit, y_top)
            meas_xb = evaluate_poly_x(meas_fit, y_bot)

            if prev_pts is None:
                # 첫 검출 시에는 즉시 수렴
                return (meas_xt, meas_xb), 0, 0.90
            else:
                prev_xt, prev_xb = prev_pts

                # 2. Slew-Rate Clamping (한 프레임당 최대 이동 픽셀 제한)
                d_xt = np.clip(meas_xt - prev_xt, -self.max_slew_top, self.max_slew_top)
                d_xb = np.clip(meas_xb - prev_xb, -self.max_slew_bot, self.max_slew_bot)

                target_xt = prev_xt + d_xt
                target_xb = prev_xb + d_xb

                # 3. 부드러운 지수이동평균(EMA) 융합
                smooth_xt = self.coord_alpha * target_xt + (1.0 - self.coord_alpha) * prev_xt
                smooth_xb = self.coord_alpha * target_xb + (1.0 - self.coord_alpha) * prev_xb

                new_conf = min(1.0, prev_conf * 0.8 + 0.20)
                return (smooth_xt, smooth_xb), 0, new_conf

        else:
            # 미검출 시 (점선 사이 빈 공간 등) -> 이전 위치를 유지하며 신뢰도를 점진적 감쇠
            new_lost = lost_count + 1
            if prev_pts is not None and new_lost <= self.max_hold_frames:
                decayed_conf = prev_conf * self.decay_factor
                return prev_pts, new_lost, decayed_conf
            else:
                return None, new_lost, 0.0

    @staticmethod
    def _pts_to_polyfit(pts: Optional[Tuple[float, float]], y_top: float, y_bot: float) -> Optional[List[float]]:
        """평활화된 (x_top, x_bot) 좌표를 다시 x = m*y + c 표준 계수 형태로 역환산."""
        if pts is None:
            return None
        xt, xb = pts
        dy = y_bot - y_top
        if abs(dy) < 1e-4:
            return None
        m = (xb - xt) / dy
        c = xt - m * y_top
        return [float(m), float(c)]

    def detect(
        self,
        frame: np.ndarray,
        is_crosswalk: Optional[bool] = None
    ) -> Tuple[Optional[List[float]], Optional[List[float]], bool, Dict[str, Any]]:
        """메인 차선 검출 파이프라인 호출."""
        h, w = frame.shape[:2]
        y_top = float(h * self.top_ratio)
        y_bot = float(h * self.bot_ratio)

        # 1. 횡단보도 상태 확인
        cw_boxes = []
        if is_crosswalk is None:
            is_crosswalk_detected, cw_boxes = self.detect_crosswalk(frame)
        else:
            is_crosswalk_detected = is_crosswalk
            _, cw_boxes = detect_crosswalk(frame, top_ratio=0.65, bot_ratio=0.88, clahe_obj=self._clahe_cw)

        # 횡단보도 확정 시: 차선 탐색 즉시 중단 및 ROI 화면 숨김
        if is_crosswalk_detected:
            self.smooth_left_pts = None
            self.smooth_right_pts = None
            self.lost_left_count = 0
            self.lost_right_count = 0
            debug_info = {
                "cw_boxes": cw_boxes,
                "confidence": 0.0,
                "is_crosswalk": True,
                "left_edges": np.zeros((100, 100), dtype=np.uint8),
                "right_edges": np.zeros((100, 100), dtype=np.uint8),
                "combined_edges": np.zeros((100, 200), dtype=np.uint8),
                "held_left": False,
                "held_right": False,
            }
            return None, None, True, debug_info

        # 2. 동적 마진(Adaptive ROI) 생성
        (left_corners, left_bbox), (right_corners, right_bbox) = get_dual_roi_geometry(
            h, w,
            top_ratio=self.top_ratio,
            bot_ratio=self.bot_ratio,
            padding_x=self.padding_x,
            prev_left_pts=self.smooth_left_pts,
            prev_right_pts=self.smooth_right_pts,
        )

        # 3. 좌/우 단일 ROI 에지 및 선분 추출
        raw_left_fit, left_lines_orig, left_edges, left_metrics = self._detect_single_roi(
            frame, left_corners, left_bbox, self._clahe_left, is_left=True, cw_boxes=cw_boxes
        )

        raw_right_fit, right_lines_orig, right_edges, right_metrics = self._detect_single_roi(
            frame, right_corners, right_bbox, self._clahe_right, is_left=False, cw_boxes=cw_boxes
        )

        # 4. 좌표 공간 기반 고안정성 평활화 (Coordinate-Space Smoothing)
        self.smooth_left_pts, self.lost_left_count, self.left_confidence = self._smooth_lane_coordinates(
            raw_left_fit, self.smooth_left_pts, self.lost_left_count, self.left_confidence, y_top, y_bot
        )

        self.smooth_right_pts, self.lost_right_count, self.right_confidence = self._smooth_lane_coordinates(
            raw_right_fit, self.smooth_right_pts, self.lost_right_count, self.right_confidence, y_top, y_bot
        )

        # 5. 차선 폭 기하학 유효성 검사 (좌우 교차 방지)
        if self.smooth_left_pts is not None and self.smooth_right_pts is not None:
            l_xt, l_xb = self.smooth_left_pts
            r_xt, r_xb = self.smooth_right_pts
            # 차선이 서로 교차하거나 폭이 너무 좁은 비정상 상태 방지
            if l_xt >= r_xt or l_xb >= r_xb or (r_xb - l_xb) < (w * 0.20):
                if self.left_confidence < self.right_confidence:
                    self.smooth_left_pts = None
                else:
                    self.smooth_right_pts = None

        final_left_fit = self._pts_to_polyfit(self.smooth_left_pts, y_top, y_bot)
        final_right_fit = self._pts_to_polyfit(self.smooth_right_pts, y_top, y_bot)

        # 통합 신뢰도
        if final_left_fit is not None and final_right_fit is not None:
            overall_conf = 0.5 * (self.left_confidence + self.right_confidence)
        elif final_left_fit is not None:
            overall_conf = 0.6 * self.left_confidence
        elif final_right_fit is not None:
            overall_conf = 0.6 * self.right_confidence
        else:
            overall_conf = 0.0

        if left_edges.shape == right_edges.shape:
            combined_edges = np.hstack([left_edges, right_edges])
        else:
            combined_edges = left_edges

        debug_info = {
            "left_corners": left_corners,
            "right_corners": right_corners,
            "left_bbox": left_bbox,
            "right_bbox": right_bbox,
            "left_edges": left_edges,
            "right_edges": right_edges,
            "combined_edges": combined_edges,
            "cw_boxes": cw_boxes,
            "confidence": float(overall_conf),
            "is_crosswalk": False,
            "held_left": self.lost_left_count > 0 and final_left_fit is not None,
            "held_right": self.lost_right_count > 0 and final_right_fit is not None,
        }

        return final_left_fit, final_right_fit, False, debug_info

    def reset(self):
        self.smooth_left_pts = None
        self.smooth_right_pts = None
        self.lost_left_count = 0
        self.lost_right_count = 0
        self.left_confidence = 0.0
        self.right_confidence = 0.0
        self.cw_detect_count = 0
        self.is_crosswalk_state = False


# ── HUD 렌더링 및 디스플레이 함수 ─────────────────────────────────────────────
def draw_lane_hud(
    frame: np.ndarray,
    left_fit: Optional[List[float]],
    right_fit: Optional[List[float]],
    is_crosswalk: bool,
    debug_info: Dict[str, Any],
    fps: float = 0.0,
    show_roi_boxes: bool = True
) -> np.ndarray:
    """프레임 위에 유도로, 안티 앨리어싱 차선, HUD 및 텔레메트리를 렌더링."""
    out = frame.copy()
    h, w = frame.shape[:2]
    top_y = int(h * 0.67)
    bot_y = int(h * 0.88)  # 무리한 100% 외삽 대신 유효 노면 영역(88%)까지만 안전 렌더링

    # 1. 듀얼 ROI 및 BBox 시각화 (횡단보도가 아닐 때만 표출)
    if show_roi_boxes and not is_crosswalk:
        if "left_corners" in debug_info and "right_corners" in debug_info:
            cv2.polylines(out, [debug_info["left_corners"]], isClosed=True, color=(255, 200, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(out, [debug_info["right_corners"]], isClosed=True, color=(0, 230, 100), thickness=2, lineType=cv2.LINE_AA)
        if "left_bbox" in debug_info and "right_bbox" in debug_info:
            lx1, ly1, lx2, ly2 = debug_info["left_bbox"]
            rx1, ry1, rx2, ry2 = debug_info["right_bbox"]
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (255, 120, 0), 1)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 180, 80), 1)

    # 2. 횡단보도 줄무늬 박스 시각화
    if "cw_boxes" in debug_info and debug_info["cw_boxes"]:
        for bx1, by1, bx2, by2 in debug_info["cw_boxes"]:
            cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            cv2.putText(out, "ZEBRA", (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)

    # 3. 주행 통로(Driving Corridor) 및 차선 곡선 그리기
    if not is_crosswalk:
        if left_fit is not None and right_fit is not None:
            # 부드러운 다각형 통로 계산
            y_samples = np.linspace(top_y, bot_y, 16)
            left_pts = [(int(round(evaluate_poly_x(left_fit, y))), int(round(y))) for y in y_samples]
            right_pts = [(int(round(evaluate_poly_x(right_fit, y))), int(round(y))) for y in reversed(y_samples)]
            poly_pts = np.array(left_pts + right_pts, dtype=np.int32)

            # 반투명 에메랄드 주행 통로
            overlay = out.copy()
            cv2.fillPoly(overlay, [poly_pts], (0, 210, 90))
            cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)

            # 좌측 차선 (청록색) & 우측 차선 (네온 그린)
            lx_top = int(round(evaluate_poly_x(left_fit, top_y)))
            lx_bot = int(round(evaluate_poly_x(left_fit, bot_y)))
            rx_top = int(round(evaluate_poly_x(right_fit, top_y)))
            rx_bot = int(round(evaluate_poly_x(right_fit, bot_y)))

            cv2.line(out, (lx_top, top_y), (lx_bot, bot_y), (255, 255, 0), thickness=4, lineType=cv2.LINE_AA)
            cv2.line(out, (rx_top, top_y), (rx_bot, bot_y), (0, 255, 120), thickness=4, lineType=cv2.LINE_AA)

            # 목표 주행 궤적 중심선
            cx_top = int((lx_top + rx_top) / 2)
            cx_bot = int((lx_bot + rx_bot) / 2)
            cv2.line(out, (cx_top, top_y), (cx_bot, bot_y), (0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(out, (cx_top, top_y), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (w // 2, bot_y), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        elif left_fit is not None:
            lx_top = int(round(evaluate_poly_x(left_fit, top_y)))
            lx_bot = int(round(evaluate_poly_x(left_fit, bot_y)))
            cv2.line(out, (lx_top, top_y), (lx_bot, bot_y), (255, 255, 0), thickness=4, lineType=cv2.LINE_AA)

        elif right_fit is not None:
            rx_top = int(round(evaluate_poly_x(right_fit, top_y)))
            rx_bot = int(round(evaluate_poly_x(right_fit, bot_y)))
            cv2.line(out, (rx_top, top_y), (rx_bot, bot_y), (0, 255, 120), thickness=4, lineType=cv2.LINE_AA)

    else:
        # 횡단보도 정지 안내 배너 표시
        banner_w, banner_h = 380, 42
        bx = (w - banner_w) // 2
        by = int(h * 0.52)
        banner_bg = out.copy()
        cv2.rectangle(banner_bg, (bx, by), (bx + banner_w, by + banner_h), (0, 0, 220), -1)
        cv2.addWeighted(banner_bg, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (bx, by), (bx + banner_w, by + banner_h), (0, 0, 255), 2)
        cv2.putText(
            out, "CROSSWALK DETECTED - HOLD / STOP",
            (bx + 18, by + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA
        )

    # 4. 오른쪽 상단 PiP 에지 맵 렌더링
    if "combined_edges" in debug_info and debug_info["combined_edges"] is not None:
        edge_map = debug_info["combined_edges"]
        eh, ew = edge_map.shape[:2]
        pip_w = 160
        pip_h = max(1, int(pip_w * (eh / float(max(1, ew)))))
        pip_resized = cv2.resize(edge_map, (pip_w, pip_h), interpolation=cv2.INTER_NEAREST)
        pip_bgr = cv2.cvtColor(pip_resized, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(pip_bgr, (0, 0), (pip_w - 1, pip_h - 1), (0, 255, 255), 1)
        out[10:10 + pip_h, w - pip_w - 10:w - 10] = pip_bgr
        cv2.putText(out, "EDGE PiP", (w - pip_w - 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # 5. 왼쪽 상단 텔레메트리 HUD 박스
    hud_bg = out.copy()
    cv2.rectangle(hud_bg, (10, 10), (280, 140), (15, 23, 42), -1)
    cv2.addWeighted(hud_bg, 0.85, out, 0.15, 0, out)
    cv2.rectangle(out, (10, 10), (280, 140), (51, 65, 85), 1)

    conf = debug_info.get("confidence", 0.0)
    conf_color = (0, 255, 120) if conf >= 0.7 else ((0, 200, 255) if conf >= 0.4 else (0, 0, 255))
    cv2.putText(out, f"FPS: {fps:.1f}  |  CONF: {conf*100:.0f}%", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, conf_color, 1, lineType=cv2.LINE_AA)

    l_str = f"L: x = {left_fit[0]:.2f}y + {left_fit[1]:.0f}" if left_fit else "L: LOST / SUPPRESSED"
    r_str = f"R: x = {right_fit[0]:.2f}y + {right_fit[1]:.0f}" if right_fit else "R: LOST / SUPPRESSED"
    cv2.putText(out, l_str, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.putText(out, r_str, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1, lineType=cv2.LINE_AA)

    if is_crosswalk:
        cw_text = "CROSSWALK: DETECTED"
        cw_color = (0, 0, 255)
        state_text = "STATE: HOLD / STOP (CW ACTIVE)"
        state_color = (0, 0, 255)
    else:
        cw_text = "CROSSWALK: CLEAR"
        cw_color = (150, 150, 150)
        held_str = ""
        if debug_info.get("held_left"):
            held_str += "L-HOLD "
        if debug_info.get("held_right"):
            held_str += "R-HOLD "
        if held_str:
            state_text = f"STATE: {held_str}(SMOOTH)"
            state_color = (0, 200, 255)
        else:
            state_text = "STATE: ACTIVE TRACKING"
            state_color = (0, 255, 120)

    cv2.putText(out, cw_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, cw_color, 1, lineType=cv2.LINE_AA)
    cv2.putText(out, state_text, (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.40, state_color, 1, lineType=cv2.LINE_AA)

    return out


# ── 비디오 파이프라인 러너 ───────────────────────────────────────────────────
def run_video_pipeline(
    video_path: str,
    output_path: Optional[str] = None,
    show_window: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
    max_frames: Optional[int] = None
):
    """비디오를 로드하여 고안정성 차선 추적 및 HUD 오버레이를 실시간 수행."""
    resolved_path = video_path
    if not os.path.exists(resolved_path):
        candidates = [
            os.path.join(os.path.dirname(__file__), video_path),
            os.path.join(os.getcwd(), video_path),
            f"/public/assets/aistudio/{video_path}",
            f"public/assets/aistudio/{video_path}",
            "/public/assets/aistudio/Test_Video_2_05bb47dc.mp4",
            "public/assets/aistudio/Test_Video_2_05bb47dc.mp4",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                resolved_path = cand
                break

    if not os.path.exists(resolved_path):
        print(f"[Error] 비디오 파일을 찾을 수 없습니다: {video_path}")
        return

    print("=" * 75)
    print(f"[*] 고안정성 차선 추적기 (High-Stability Lane Tracker) 시작: {resolved_path}")
    print("=" * 75)

    cap = cv2.VideoCapture(resolved_path)
    if not cap.isOpened():
        print(f"[Error] 비디오 파일을 열 수 없습니다: {resolved_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"    - 비디오 해상도: {width} x {height}")
    print(f"    - 입력 FPS: {fps_in:.1f}")
    print(f"    - 총 프레임 수: {total_frames}")

    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
        print(f"    - 결과 저장 경로: {output_path}")

    detector = HighStabilityLaneTracker(cfg=cfg)

    frame_idx = 0
    t_start = time.time()
    prev_time = time.time()
    fps_display = fps_in
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n[*] 비디오 재생 완료 (End of Video).")
                    break
                frame_idx += 1

                # 고안정성 추적 파이프라인 실행
                left_fit, right_fit, is_cw, debug = detector.detect(frame)

                # FPS 계산
                cur_time = time.time()
                dt = cur_time - prev_time
                prev_time = cur_time
                if dt > 0:
                    fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

                # HUD 오버레이 렌더링
                annotated = draw_lane_hud(
                    frame, left_fit, right_fit, is_cw, debug, fps=fps_display, show_roi_boxes=True
                )

                if writer is not None:
                    writer.write(annotated)

                if frame_idx % 30 == 0 or frame_idx == total_frames:
                    cw_badge = "[CW ACTIVE]" if is_cw else "[CLEAR]"
                    sys.stdout.write(f"\r진행률: [{frame_idx}/{total_frames}] {frame_idx/max(1, total_frames)*100:.1f}% | {cw_badge} | FPS: {fps_display:.1f}")
                    sys.stdout.flush()

                if max_frames and frame_idx >= max_frames:
                    print(f"\n[*] 지정된 최대 프레임 수({max_frames}) 도달.")
                    break

            if show_window:
                try:
                    cv2.imshow("High-Stability Lane Tracker - Test_Video_2", annotated)
                    key = cv2.waitKey(1) if not paused else 30 & 0xFF
                    if key == ord('q'):
                        print("\n[*] 사용자에 의해 재생 중단 (Quit).")
                        break
                    elif key == ord(' '):  # 일시정지/재개
                        paused = not paused
                    elif key == ord('r'):  # 리셋
                        detector.reset()
                except cv2.error:
                    show_window = False

    except KeyboardInterrupt:
        print("\n[*] 강제 종료되었습니다.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        print(f"\n[완료] 총 {frame_idx} 프레임 처리 완료 (소요 시간: {elapsed:.2f}초, 평균 {frame_idx/max(0.001, elapsed):.1f} FPS)")


# ── MAIN CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Stability Lane & Crosswalk Tracker")
    parser.add_argument("--video", type=str, default="Test_Video_2_05bb47dc.mp4", help="비디오 파일 경로")
    parser.add_argument("--output", type=str, default=None, help="처리된 결과 비디오 저장 경로")
    parser.add_argument("--no-show", action="store_true", help="화면 창 띄우지 않고 백그라운드 처리")
    parser.add_argument("--max-frames", type=int, default=None, help="처리할 최대 프레임 수")
    args = parser.parse_args()

    custom_cfg = {
        "lane": {
            "coord_alpha": 0.18,        # 좌표계 EMA 평활 계수 (낮을수록 더욱 부드러움)
            "max_slew_top": 12.0,       # 상단 프레임당 최대 이동 12px (급격한 점프 방지)
            "max_slew_bot": 18.0,       # 하단 프레임당 최대 이동 18px
            "max_hold_frames": 12,      # 점선 누락 시 감쇠 유지 프레임
            "decay_factor": 0.94,       # 유지 시 신뢰도 감쇠율
            "hough_threshold": 22,
            "min_line_length": 18,
            "max_line_gap": 45,
            "min_slope": 0.40,
            "max_slope": 3.00,
        },
        "roi": {
            "top_ratio": 0.67,
            "bottom_ratio": 0.88,
            "padding_x": 24             # 곡선 도로 추종용 동적 패딩 마진
        },
        "crosswalk": {
            "threshold": 175,
            "hysteresis_enter": 2,
            "hysteresis_exit": 2
        }
    }

    video_to_run = args.video
    if not os.path.exists(video_to_run):
        candidates = [
            video_to_run,
            "Test_Video_2.mp4",
            "Test_Video_2_05bb47dc.mp4",
            "dashcam_cv/data/Test_Video_2.mp4",
        ]
        for c in candidates:
            if os.path.exists(c):
                video_to_run = c
                break

    run_video_pipeline(
        video_path=video_to_run,
        output_path=args.output,
        show_window=not args.no_show,
        cfg=custom_cfg,
        max_frames=args.max_frames
    )