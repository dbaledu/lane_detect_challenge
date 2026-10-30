#!/usr/bin/env python3
"""run_antigravity.py

[완전 개정판] 전천후 곡선 적응형 고안정성 자율주행 차선 및 횡단보도 추적기
=============================================================================
(All-Weather High-Stability Adaptive Curve Lane & Crosswalk Tracker v2.0)

[기존 6대 치명적 실패 원인 완전 해결]
1. [곡선 에지 탈락 해결] 2차 다항식 (x = ay² + by + c) 피팅 + 35px 적응형 인라이어 잔차 필터
2. [영구 락아웃(Lock-Out) 차단] 광각 폴백(Wide Fallback) 동적 ROI 및 중앙선(w//2) 제약 해제
3. [그림자 오인 & 황색선 미검출 해결] HLS(Hue-Lightness-Saturation) 채널 결합 컬러-에지 마스크
4. [노면 기호 오탐 & 강제 셧다운 해결] 스트라이프 주기성(Spacing Uniformity) 검증 및 차선 상태 동결(Freeze)
5. [곡선 추종 지연(Lag) 해결] 곡률 및 오차 크기 연동 가변 Slew-Rate & 적응형 EMA
6. [완만한 곡선 기각 해결] 적응형 기울기 대역(0.18 ~ 4.50) 허용
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any


# ── 1. 기하 안전 연산 함수 ───────────────────────────────────────────────────
def evaluate_poly_x(fit: Optional[List[float]], y: float) -> float:
    """다항식 계수(1차 직선 또는 2차 곡선)로부터 y값에 대응하는 x 좌표를 안전하게 계산."""
    if fit is None or len(fit) < 2:
        return 0.0
    if len(fit) == 2:  # x = m*y + c
        return float(fit[0] * y + fit[1])
    elif len(fit) >= 3:  # x = a*y² + b*y + c
        return float(fit[0] * (y ** 2) + fit[1] * y + fit[2])
    return 0.0


# ── 2. HLS + CLAHE 하이브리드 전처리 (그림자 제거 & 황색/백색 차선 강화) ──────
def preprocess_roi_advanced(
    roi_img: np.ndarray,
    clahe_obj: Optional[cv2.CLAHE] = None
) -> np.ndarray:
    """그레이스케일의 한계를 극복하기 위해 HLS 채널 마스킹과 적응형 Canny를 결합."""
    if roi_img is None or roi_img.size == 0:
        return np.zeros((10, 10), dtype=np.uint8)

    # 1. HLS 색상 변환
    hls = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # 노란색 차선 마스크: Hue(14~38) & Saturation(>= 70)
    yellow_mask = cv2.inRange(hls, np.array([14, 50, 70]), np.array([38, 255, 255]))

    # 흰색 차선 마스크: 밝기(Lightness >= 175)
    white_mask = cv2.inRange(l_channel, 175, 255)

    # 색상 마스크 결합 (노란색 + 흰색)
    color_mask = cv2.bitwise_or(yellow_mask, white_mask)

    # 2. 조명 보정을 위한 CLAHE 적용
    if clahe_obj is not None:
        enhanced_l = clahe_obj.apply(l_channel)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)

    blurred = cv2.GaussianBlur(enhanced_l, (5, 5), 0)
    canny_edges = cv2.Canny(blurred, 40, 120)

    # 3. 색상 마스크 확장 후 Canny 에지와 교집합을 취해 그림자 경계선 배제
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_color = cv2.dilate(color_mask, kernel, iterations=1)
    filtered_edges = cv2.bitwise_and(canny_edges, canny_edges, mask=dilated_color)

    # 차선 도색이 심하게 마모되어 색상 마스크 에지가 부족할 경우 일반 Canny로 자연스럽게 폴백
    if np.count_nonzero(filtered_edges) < 25:
        return canny_edges

    return filtered_edges


# ── 3. 광각 폴백(Wide Fallback) 동적 ROI 생성 ─────────────────────────────
def get_adaptive_roi_geometry_v2(
    h: int,
    w: int,
    top_ratio: float = 0.60,
    bot_ratio: float = 0.88,
    padding_x: int = 42,
    prev_left_pts: Optional[Tuple[float, float, float]] = None,
    prev_right_pts: Optional[Tuple[float, float, float]] = None,
    is_left_lost: bool = False,
    is_right_lost: bool = False,
) -> Tuple[Tuple[np.ndarray, Tuple[int, int, int, int]], Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """급커브/차선변경 시 영구 락아웃을 방지하는 Wide-Fallback 탑재 동적 ROI."""
    top_y = int(h * top_ratio)
    bot_y = int(h * bot_ratio)

    # [좌측 차선 ROI]
    # 이전 추적 상태가 있고 정상 추종 중일 때: 좁은 추종 사다리꼴
    if prev_left_pts is not None and not is_left_lost:
        px_top, px_mid, px_bot = prev_left_pts
        l_tl_x = max(0, int(px_top - padding_x * 1.6))
        l_tr_x = min(w - 20, int(px_top + padding_x * 1.6))
        l_bl_x = max(0, int(px_bot - padding_x * 2.2))
        l_br_x = min(w - 10, int(px_bot + padding_x * 2.0))
    else:
        # [핵심] 차선을 놓쳤거나 초기 상태일 때: 전체 노면 좌반부 광각 폴백 ROI
        l_tl_x = int(w * 0.08)
        l_tr_x = int(w * 0.52)
        l_br_x = int(w * 0.55)
        l_bl_x = int(w * 0.02)

    left_tl = (l_tl_x, top_y)
    left_tr = (l_tr_x, top_y)
    left_br = (l_br_x, bot_y)
    left_bl = (l_bl_x, bot_y)
    left_corners = np.array([[left_tl, left_tr, left_br, left_bl]], dtype=np.int32)

    l_xs = [left_tl[0], left_tr[0], left_br[0], left_bl[0]]
    left_bbox = (max(0, min(l_xs) - 4), top_y, min(w, max(l_xs) + 4), bot_y)

    # [우측 차선 ROI]
    if prev_right_pts is not None and not is_right_lost:
        px_top, px_mid, px_bot = prev_right_pts
        r_tl_x = max(10, int(px_top - padding_x * 1.6))
        r_tr_x = min(w, int(px_top + padding_x * 1.6))
        r_bl_x = max(10, int(px_bot - padding_x * 2.0))
        r_br_x = min(w, int(px_bot + padding_x * 2.2))
    else:
        # 전체 노면 우반부 광각 폴백 ROI
        r_tl_x = int(w * 0.48)
        r_tr_x = int(w * 0.92)
        r_br_x = int(w * 0.98)
        r_bl_x = int(w * 0.45)

    right_tl = (r_tl_x, top_y)
    right_tr = (r_tr_x, top_y)
    right_br = (r_br_x, bot_y)
    right_bl = (r_bl_x, bot_y)
    right_corners = np.array([[right_tl, right_tr, right_br, right_bl]], dtype=np.int32)

    r_xs = [right_tl[0], right_tr[0], right_br[0], right_bl[0]]
    right_bbox = (max(0, min(r_xs) - 4), top_y, min(w, max(r_xs) + 4), bot_y)

    return (left_corners, left_bbox), (right_corners, right_bbox)


def crop_and_mask_roi(
    frame: np.ndarray,
    roi_corners: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, Tuple[int, int, float, float]]:
    """Bounding Box 영역 슬라이싱 후 사다리꼴 마스크를 안전하게 적용."""
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

    return masked_crop, (xmin, ymin, 1.0, 1.0)


def scale_lines_to_original(
    lines: Optional[np.ndarray],
    transform_params: Tuple[int, int, float, float]
) -> Optional[np.ndarray]:
    """로컬 ROI 좌표계의 선분들을 원본 프레임 좌표계로 투영 변환."""
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


# ── 4. 강건한 2차 다항식 곡선 피팅 (2nd-Order Robust WLS Curve Fit) ─────────
def fit_lane_curve_robust(
    lines: Optional[np.ndarray],
    is_left: bool = True,
    min_slope: float = 0.18,          # 완만한 곡선 선분도 포용 (기존 0.40 -> 0.18)
    max_slope: float = 4.50,          # 수직에 가까운 선분 허용 (기존 3.00 -> 4.50)
    min_line_count: int = 2,
    min_total_length: float = 24.0,
    min_y_span_ratio: float = 0.18,
    roi_h: int = 100
) -> Tuple[Optional[List[float]], Dict[str, Any]]:
    """직선(1차) 대신 2차 포물선(x = ay² + by + c)을 피팅하여 급커브 에지 아웃라이어 기각 해결."""
    metrics = {
        "line_count": 0,
        "total_length": 0.0,
        "y_span": 0,
        "rmse": 999.0,
        "valid": False,
        "degree": 1
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

        # 곡선에서는 부호가 일시적으로 뒤집힐 수 있으므로 절대 기울기가 큰 수직 성분은 양측 모두 유효 인정
        if is_left and (slope <= -min_slope or abs_slope >= 1.6):
            pts_x.extend([x1, x2])
            pts_y.extend([y1, y2])
            weights.extend([line_len, line_len])
            total_length += line_len
            valid_lines += 1
        elif not is_left and (slope >= min_slope or abs_slope >= 1.6):
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
        y_arr = np.array(pts_y, dtype=np.float32)
        x_arr = np.array(pts_x, dtype=np.float32)

        # 1. 2차 다항식(포물선) 초기 피팅: x = a*y² + b*y + c
        fit2 = np.polyfit(y_arr, x_arr, 2, w=w_arr)
        pred_x = fit2[0] * (y_arr ** 2) + fit2[1] * y_arr + fit2[2]
        residuals = np.abs(pred_x - x_arr)

        # 잔차 임계치를 20px에서 35px로 현실화하여 곡선의 유효 에지 보존
        inliers_mask = residuals < 35.0

        if np.sum(inliers_mask) >= 4:
            clean_y = y_arr[inliers_mask]
            clean_x = x_arr[inliers_mask]
            clean_w = w_arr[inliers_mask]

            # 인라이어로 정제된 2차 곡선 재피팅
            refined_fit = np.polyfit(clean_y, clean_x, 2, w=clean_w)
            pred_clean = refined_fit[0] * (clean_y ** 2) + refined_fit[1] * clean_y + refined_fit[2]
            rmse = float(np.sqrt(np.mean((pred_clean - clean_x) ** 2)))

            # 곡률 계수 'a'가 극도로 작으면 직선(1차)으로 단순화하여 과적합 방지
            if abs(refined_fit[0]) < 1e-4:
                linear_fit = np.polyfit(clean_y, clean_x, 1, w=clean_w)
                metrics["degree"] = 1
                metrics["rmse"] = rmse
                metrics["valid"] = True
                return [0.0, float(linear_fit[0]), float(linear_fit[1])], metrics

            metrics["degree"] = 2
            metrics["rmse"] = rmse
            metrics["valid"] = True
            return [float(refined_fit[0]), float(refined_fit[1]), float(refined_fit[2])], metrics
        else:
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            metrics["degree"] = 2
            metrics["rmse"] = rmse
            metrics["valid"] = True
            return [float(fit2[0]), float(fit2[1]), float(fit2[2])], metrics

    except Exception:
        return None, metrics


# ── 5. 노면 기호(화살표/숫자) 오탐 방지 횡단보도 검출기 ───────────────────────
def detect_crosswalk_robust(
    frame: np.ndarray,
    top_ratio: float = 0.65,
    bot_ratio: float = 0.88,
    thresh_val: int = 195,            # 임계치 상향 (175 -> 195)으로 아스팔트 잔상 제거
    clahe_obj: Optional[cv2.CLAHE] = None
) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
    """화살표 1~2개로 인한 오탐을 배제하기 위해 스트라이프 5개 이상 & 가로 주기성(Spacing) 검증."""
    h, w = frame.shape[:2]
    scale = w / 640.0
    y1 = int(h * top_ratio)
    y2 = int(h * bot_ratio)
    x1 = int(w * 0.20)
    x2 = int(w * 0.80)

    road = frame[y1:y2, x1:x2]
    if road.size == 0:
        return False, []

    gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    if clahe_obj is not None:
        enhanced = clahe_obj.apply(gray)
    else:
        enhanced = gray

    _, thresh = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)
    k_w = max(5, int(15 * scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripes = 0
    boxes = []
    x_centers = []

    min_w = int(22 * scale)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (70 * scale) or area > (15000 * scale * scale):
            continue
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        aspect = rw / float(max(1, rh))
        if 1.8 <= aspect <= 14.0 and rw >= min_w and rh >= int(7 * scale):
            stripes += 1
            boxes.append((x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh))
            x_centers.append(x1 + rx + rw / 2.0)

    # 횡단보도는 최소 5개 이상의 스트라이프가 가로로 균일한 간격(Periodic Spacing)을 형성해야 함
    is_crosswalk = False
    if stripes >= 5 and len(x_centers) >= 5:
        x_centers.sort()
        spacings = np.diff(x_centers)
        mean_sp = np.mean(spacings)
        std_sp = np.std(spacings)
        # 간격 변동 계수(CV)가 0.48 이하일 때만 진정한 횡단보도로 확정
        if (std_sp / (mean_sp + 1e-5)) < 0.48:
            is_crosswalk = True

    return is_crosswalk, boxes


# ── 6. [차세대 엔진] 곡선 적응형 고안정성 차선 추적기 클래스 ──────────────────
class AdaptiveHighStabilityLaneTracker:
    """2차 곡선 피팅, Wide-Fallback ROI, 상태 동결(Freeze) 메모리를 탑재한 차선 추적기."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        lane_cfg = self.cfg.get("lane", {})
        roi_cfg = self.cfg.get("roi", {})
        cw_cfg = self.cfg.get("crosswalk", {})

        # 기하학 파라미터
        self.top_ratio = float(roi_cfg.get("top_ratio", 0.60))
        self.bot_ratio = float(roi_cfg.get("bottom_ratio", 0.88))
        self.padding_x = int(roi_cfg.get("padding_x", 36))

        # 허프 파라미터
        self.hough_threshold = int(lane_cfg.get("hough_threshold", 20))
        self.min_line_length = int(lane_cfg.get("min_line_length", 16))
        self.max_line_gap = int(lane_cfg.get("max_line_gap", 50))
        self.min_slope = float(lane_cfg.get("min_slope", 0.18))
        self.max_slope = float(lane_cfg.get("max_slope", 4.50))

        # [적응형 스무딩 파라미터]
        self.base_coord_alpha = float(lane_cfg.get("coord_alpha", 0.25))
        self.max_slew_top = float(lane_cfg.get("max_slew_top", 26.0))
        self.max_slew_bot = float(lane_cfg.get("max_slew_bot", 36.0))

        # 점선 구간 감쇠 홀드
        self.max_hold_frames = int(lane_cfg.get("max_hold_frames", 16))
        self.decay_factor = float(lane_cfg.get("decay_factor", 0.95))

        # 3점 화면 좌표계 상태 변수 (x_top, x_mid, x_bot) -> 2차 곡선 완전 결정
        self.smooth_left_pts: Optional[Tuple[float, float, float]] = None
        self.smooth_right_pts: Optional[Tuple[float, float, float]] = None
        self.lost_left_count: int = 0
        self.lost_right_count: int = 0
        self.left_confidence: float = 0.0
        self.right_confidence: float = 0.0

        # 횡단보도 상태 머신
        self.cw_detect_count = 0
        self.is_crosswalk_state = False
        self.cw_hysteresis_enter = int(cw_cfg.get("hysteresis_enter", 3))
        self.cw_hysteresis_exit = int(cw_cfg.get("hysteresis_exit", 2))
        self.cw_thresh = int(cw_cfg.get("threshold", 195))

        # CLAHE 객체 사전 할당
        self._clahe_left = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._clahe_right = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._clahe_cw = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect_crosswalk(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
        raw_detected, boxes = detect_crosswalk_robust(
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
            if self.cw_detect_count <= 0:
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
        roi_img, params = crop_and_mask_roi(frame, corners, bbox)

        # HLS + CLAHE 하이브리드 전처리로 그림자 제거 및 황색/백색 차선 강화
        edges = preprocess_roi_advanced(roi_img, clahe_obj=clahe_obj)

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

        fit, metrics = fit_lane_curve_robust(
            lines_orig,
            is_left=is_left,
            min_slope=self.min_slope,
            max_slope=self.max_slope,
            roi_h=int(h * (self.bot_ratio - self.top_ratio))
        )

        return fit, lines_orig, edges, metrics

    # ── [핵심 개선] 3점 좌표계(x_top, x_mid, x_bot) 적응형 평활화 ───────────
    def _smooth_curve_coordinates(
        self,
        meas_fit: Optional[List[float]],
        prev_pts: Optional[Tuple[float, float, float]],
        lost_count: int,
        prev_conf: float,
        y_top: float,
        y_mid: float,
        y_bot: float
    ) -> Tuple[Optional[Tuple[float, float, float]], int, float]:
        """곡선의 3개 앵커 포인트(상단, 중간, 하단)를 직접 평활화하여 시소 떨림 없이 곡률 추종."""
        if meas_fit is not None:
            # 1. 측정된 2차 곡선을 3개 앵커 좌표로 변환
            meas_xt = evaluate_poly_x(meas_fit, y_top)
            meas_xm = evaluate_poly_x(meas_fit, y_mid)
            meas_xb = evaluate_poly_x(meas_fit, y_bot)

            if prev_pts is None:
                return (meas_xt, meas_xm, meas_xb), 0, 0.92
            else:
                prev_xt, prev_xm, prev_xb = prev_pts

                # 2. 적응형 Slew-Rate 제한 (오차 크기에 따라 유연하게 대응)
                diff_top = meas_xt - prev_xt
                diff_mid = meas_xm - prev_xm
                diff_bot = meas_xb - prev_xb

                # 오차가 클 경우(코너 진입) 클램핑 한도를 1.5배까지 자동 확장하여 랙(Lag) 방지
                scale_top = self.max_slew_top * (1.5 if abs(diff_top) > self.max_slew_top else 1.0)
                scale_bot = self.max_slew_bot * (1.5 if abs(diff_bot) > self.max_slew_bot else 1.0)
                scale_mid = (scale_top + scale_bot) / 2.0

                d_xt = np.clip(diff_top, -scale_top, scale_top)
                d_xm = np.clip(diff_mid, -scale_mid, scale_mid)
                d_xb = np.clip(diff_bot, -scale_bot, scale_bot)

                target_xt = prev_xt + d_xt
                target_xm = prev_xm + d_xm
                target_xb = prev_xb + d_xb

                # 3. 곡선 적응형 EMA 가중치 (오차에 비례해 alpha를 0.25~0.55로 가변)
                error_norm = (abs(d_xt) + abs(d_xb)) / (scale_top + scale_bot + 1e-4)
                dynamic_alpha = min(0.55, self.base_coord_alpha + 0.30 * error_norm)

                smooth_xt = dynamic_alpha * target_xt + (1.0 - dynamic_alpha) * prev_xt
                smooth_xm = dynamic_alpha * target_xm + (1.0 - dynamic_alpha) * prev_xm
                smooth_xb = dynamic_alpha * target_xb + (1.0 - dynamic_alpha) * prev_xb

                new_conf = min(1.0, prev_conf * 0.75 + 0.25)
                return (smooth_xt, smooth_xm, smooth_xb), 0, new_conf

        else:
            # 점선 빈 공간 등 일시 미검출 시 점진적 감쇠 유지
            new_lost = lost_count + 1
            if prev_pts is not None and new_lost <= self.max_hold_frames:
                decayed_conf = prev_conf * self.decay_factor
                return prev_pts, new_lost, decayed_conf
            else:
                return None, new_lost, 0.0

    @staticmethod
    def _pts_to_polyfit2(
        pts: Optional[Tuple[float, float, float]],
        y_top: float,
        y_mid: float,
        y_bot: float
    ) -> Optional[List[float]]:
        """평활화된 3개 점으로부터 최종 2차 곡선 계수 [a, b, c] 산출."""
        if pts is None:
            return None
        xt, xm, xb = pts
        try:
            fit = np.polyfit([y_top, y_mid, y_bot], [xt, xm, xb], 2)
            return [float(fit[0]), float(fit[1]), float(fit[2])]
        except Exception:
            return None

    def detect(
        self,
        frame: np.ndarray,
        is_crosswalk: Optional[bool] = None
    ) -> Tuple[Optional[List[float]], Optional[List[float]], bool, Dict[str, Any]]:
        """메인 차선 검출 파이프라인."""
        h, w = frame.shape[:2]
        y_top = float(h * self.top_ratio)
        y_bot = float(h * self.bot_ratio)
        y_mid = float((y_top + y_bot) / 2.0)

        # 1. 횡단보도 상태 확인
        cw_boxes = []
        if is_crosswalk is None:
            is_crosswalk_detected, cw_boxes = self.detect_crosswalk(frame)
        else:
            is_crosswalk_detected = is_crosswalk
            _, cw_boxes = detect_crosswalk_robust(frame, top_ratio=0.65, bot_ratio=0.88, clahe_obj=self._clahe_cw)

        # [핵심 개선] 횡단보도 감지 시 기존 차선 상태를 None으로 파괴하지 않고 '동결(Freeze)' 유지!
        if is_crosswalk_detected:
            final_left_fit = self._pts_to_polyfit2(self.smooth_left_pts, y_top, y_mid, y_bot)
            final_right_fit = self._pts_to_polyfit2(self.smooth_right_pts, y_top, y_mid, y_bot)
            debug_info = {
                "cw_boxes": cw_boxes,
                "confidence": 0.5,
                "is_crosswalk": True,
                "left_edges": np.zeros((100, 100), dtype=np.uint8),
                "right_edges": np.zeros((100, 100), dtype=np.uint8),
                "combined_edges": np.zeros((100, 200), dtype=np.uint8),
                "held_left": True,
                "held_right": True,
            }
            return final_left_fit, final_right_fit, True, debug_info

        # 2. 광각 폴백(Wide Fallback) 동적 ROI 생성
        (left_corners, left_bbox), (right_corners, right_bbox) = get_adaptive_roi_geometry_v2(
            h, w,
            top_ratio=self.top_ratio,
            bot_ratio=self.bot_ratio,
            padding_x=self.padding_x,
            prev_left_pts=self.smooth_left_pts,
            prev_right_pts=self.smooth_right_pts,
            is_left_lost=self.lost_left_count > 3,
            is_right_lost=self.lost_right_count > 3,
        )

        # 3. 좌/우 단일 ROI 에지 및 2차 곡선 추출
        raw_left_fit, left_lines_orig, left_edges, left_metrics = self._detect_single_roi(
            frame, left_corners, left_bbox, self._clahe_left, is_left=True, cw_boxes=cw_boxes
        )

        raw_right_fit, right_lines_orig, right_edges, right_metrics = self._detect_single_roi(
            frame, right_corners, right_bbox, self._clahe_right, is_left=False, cw_boxes=cw_boxes
        )

        # 4. 곡선 3점 좌표계 평활화
        self.smooth_left_pts, self.lost_left_count, self.left_confidence = self._smooth_curve_coordinates(
            raw_left_fit, self.smooth_left_pts, self.lost_left_count, self.left_confidence, y_top, y_mid, y_bot
        )

        self.smooth_right_pts, self.lost_right_count, self.right_confidence = self._smooth_curve_coordinates(
            raw_right_fit, self.smooth_right_pts, self.lost_right_count, self.right_confidence, y_top, y_mid, y_bot
        )

        # 5. 차선 폭 유효성 검사 (좌우 교차 방지)
        if self.smooth_left_pts is not None and self.smooth_right_pts is not None:
            l_xt, _, l_xb = self.smooth_left_pts
            r_xt, _, r_xb = self.smooth_right_pts
            if l_xt >= r_xt or l_xb >= r_xb or (r_xb - l_xb) < (w * 0.15):
                if self.left_confidence < self.right_confidence:
                    self.smooth_left_pts = None
                else:
                    self.smooth_right_pts = None

        final_left_fit = self._pts_to_polyfit2(self.smooth_left_pts, y_top, y_mid, y_bot)
        final_right_fit = self._pts_to_polyfit2(self.smooth_right_pts, y_top, y_mid, y_bot)

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


# ── 7. HUD 렌더링 및 디스플레이 함수 ─────────────────────────────────────────
def draw_lane_hud_v2(
    frame: np.ndarray,
    left_fit: Optional[List[float]],
    right_fit: Optional[List[float]],
    is_crosswalk: bool,
    debug_info: Dict[str, Any],
    fps: float = 0.0,
    show_roi_boxes: bool = True
) -> np.ndarray:
    """부드러운 2차 곡선 다각형 주행 통로와 안티 앨리어싱 HUD를 렌더링."""
    out = frame.copy()
    h, w = frame.shape[:2]
    top_y = int(h * 0.60)
    bot_y = int(h * 0.88)

    # 1. 듀얼 ROI 박스 시각화
    if show_roi_boxes:
        if "left_corners" in debug_info and "right_corners" in debug_info:
            cv2.polylines(out, [debug_info["left_corners"]], isClosed=True, color=(255, 200, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(out, [debug_info["right_corners"]], isClosed=True, color=(0, 230, 100), thickness=2, lineType=cv2.LINE_AA)

    # 2. 횡단보도 줄무늬 박스 시각화
    if "cw_boxes" in debug_info and debug_info["cw_boxes"]:
        for bx1, by1, bx2, by2 in debug_info["cw_boxes"]:
            cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            cv2.putText(out, "ZEBRA", (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)

    # 3. 2차 곡선 주행 통로(Driving Corridor) 그리기
    steps = 20
    y_samples = np.linspace(top_y, bot_y, steps)

    if left_fit is not None and right_fit is not None:
        left_pts = [(int(round(evaluate_poly_x(left_fit, y))), int(round(y))) for y in y_samples]
        right_pts = [(int(round(evaluate_poly_x(right_fit, y))), int(round(y))) for y in reversed(y_samples)]
        poly_pts = np.array(left_pts + right_pts, dtype=np.int32)

        # 에메랄드 반투명 주행 회랑
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly_pts], (0, 220, 100))
        cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)

        # 좌측 곡선 (청록색) & 우측 곡선 (네온 그린)
        for i in range(len(left_pts) - 1):
            cv2.line(out, left_pts[i], left_pts[i + 1], (255, 255, 0), thickness=4, lineType=cv2.LINE_AA)
            cv2.line(out, right_pts[i], right_pts[i + 1], (0, 255, 120), thickness=4, lineType=cv2.LINE_AA)

        # 조향 유도 중심 궤적선
        center_pts = [
            (int((left_pts[i][0] + right_pts[len(right_pts) - 1 - i][0]) / 2), int(y_samples[i]))
            for i in range(steps)
        ]
        for i in range(len(center_pts) - 1):
            cv2.line(out, center_pts[i], center_pts[i + 1], (0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(out, center_pts[0], 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    elif left_fit is not None:
        left_pts = [(int(round(evaluate_poly_x(left_fit, y))), int(round(y))) for y in y_samples]
        for i in range(len(left_pts) - 1):
            cv2.line(out, left_pts[i], left_pts[i + 1], (255, 255, 0), thickness=4, lineType=cv2.LINE_AA)

    elif right_fit is not None:
        right_pts = [(int(round(evaluate_poly_x(right_fit, y))), int(round(y))) for y in y_samples]
        for i in range(len(right_pts) - 1):
            cv2.line(out, right_pts[i], right_pts[i + 1], (0, 255, 120), thickness=4, lineType=cv2.LINE_AA)

    # 4. 횡단보도 정지 안내 배너 (차선 오버레이와 공존)
    if is_crosswalk:
        banner_w, banner_h = 380, 42
        bx = (w - banner_w) // 2
        by = int(h * 0.50)
        banner_bg = out.copy()
        cv2.rectangle(banner_bg, (bx, by), (bx + banner_w, by + banner_h), (0, 0, 220), -1)
        cv2.addWeighted(banner_bg, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (bx, by), (bx + banner_w, by + banner_h), (0, 0, 255), 2)
        cv2.putText(
            out, "CROSSWALK DETECTED - HOLD / STOP",
            (bx + 18, by + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA
        )

    # 5. 오른쪽 상단 PiP 에지 맵 렌더링
    if "combined_edges" in debug_info and debug_info["combined_edges"] is not None:
        edge_map = debug_info["combined_edges"]
        eh, ew = edge_map.shape[:2]
        pip_w = 160
        pip_h = max(1, int(pip_w * (eh / float(max(1, ew)))))
        pip_resized = cv2.resize(edge_map, (pip_w, pip_h), interpolation=cv2.INTER_NEAREST)
        pip_bgr = cv2.cvtColor(pip_resized, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(pip_bgr, (0, 0), (pip_w - 1, pip_h - 1), (0, 255, 255), 1)
        out[10:10 + pip_h, w - pip_w - 10:w - 10] = pip_bgr
        cv2.putText(out, "HLS-EDGE PiP", (w - pip_w - 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # 6. 왼쪽 상단 텔레메트리 HUD 박스
    hud_bg = out.copy()
    cv2.rectangle(hud_bg, (10, 10), (280, 140), (15, 23, 42), -1)
    cv2.addWeighted(hud_bg, 0.85, out, 0.15, 0, out)
    cv2.rectangle(out, (10, 10), (280, 140), (51, 65, 85), 1)

    conf = debug_info.get("confidence", 0.0)
    conf_color = (0, 255, 120) if conf >= 0.7 else ((0, 200, 255) if conf >= 0.4 else (0, 0, 255))
    cv2.putText(out, f"FPS: {fps:.1f}  |  CONF: {conf*100:.0f}% (v2.0)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46, conf_color, 1, lineType=cv2.LINE_AA)

    l_str = "L: 2nd-Order CURVE" if left_fit and abs(left_fit[0]) > 1e-4 else ("L: LINEAR" if left_fit else "L: LOST")
    r_str = "R: 2nd-Order CURVE" if right_fit and abs(right_fit[0]) > 1e-4 else ("R: LINEAR" if right_fit else "R: LOST")
    cv2.putText(out, l_str, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.putText(out, r_str, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1, lineType=cv2.LINE_AA)

    cw_text = "CROSSWALK: DETECTED" if is_crosswalk else "CROSSWALK: CLEAR"
    cw_color = (0, 0, 255) if is_crosswalk else (150, 150, 150)
    state_text = "STATE: CURVE ADAPTIVE ACTIVE" if conf > 0.4 else "STATE: SEARCHING (WIDE ROI)"
    state_color = (0, 255, 120) if conf > 0.4 else (0, 150, 255)

    cv2.putText(out, cw_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.40, cw_color, 1, lineType=cv2.LINE_AA)
    cv2.putText(out, state_text, (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.38, state_color, 1, lineType=cv2.LINE_AA)

    return out


# ── 8. 비디오 파이프라인 러너 ─────────────────────────────────────────────────
def run_video_pipeline(
    video_path: str,
    output_path: Optional[str] = None,
    show_window: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
    max_frames: Optional[int] = None
):
    """비디오를 로드하여 개선된 고안정성 차선 추적 및 HUD 오버레이를 실시간 수행."""
    resolved_path = video_path
    if not os.path.exists(resolved_path):
        candidates = [
            video_path,
            os.path.join(os.path.dirname(__file__), video_path),
            os.path.join(os.getcwd(), video_path),
            f"/public/assets/aistudio/{video_path}",
            f"public/assets/aistudio/{video_path}",
            "Test_Video_2.mp4",
            "Test_Video_2_05bb47dc.mp4",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                resolved_path = cand
                break

    if not os.path.exists(resolved_path):
        print(f"[Error] 비디오 파일을 찾을 수 없습니다: {video_path}")
        return

    print("=" * 75)
    print(f"[*] [v2.0] 곡선 적응형 차선 추적기 (Adaptive Curve Tracker) 시작: {resolved_path}")
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

    detector = AdaptiveHighStabilityLaneTracker(cfg=cfg)

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

                # 2차 곡선 적응형 추적 파이프라인 호출
                left_fit, right_fit, is_cw, debug = detector.detect(frame)

                # FPS 계산
                cur_time = time.time()
                dt = cur_time - prev_time
                prev_time = cur_time
                if dt > 0:
                    fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

                # HUD 렌더링
                annotated = draw_lane_hud_v2(
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
                    cv2.imshow("Adaptive Curve Lane Tracker v2.0", annotated)
                    key = cv2.waitKey(1) if not paused else 30 & 0xFF
                    if key == ord('q'):
                        print("\n[*] 사용자에 의해 중단 (Quit).")
                        break
                    elif key == ord(' '):
                        paused = not paused
                    elif key == ord('r'):
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
    parser = argparse.ArgumentParser(description="Adaptive Curve Lane & Crosswalk Tracker v2.0")
    parser.add_argument("--video", type=str, default="Test_Video_2_05bb47dc.mp4", help="비디오 파일 경로")
    parser.add_argument("--output", type=str, default=None, help="결과 비디오 저장 경로")
    parser.add_argument("--no-show", action="store_true", help="창 띄우지 않고 백그라운드 처리")
    parser.add_argument("--max-frames", type=int, default=None, help="처리할 최대 프레임 수")
    args = parser.parse_args()

    custom_cfg = {
        "lane": {
            "coord_alpha": 0.25,        # 적응형 EMA 기본 계수 (0.25~0.55 가변)
            "max_slew_top": 26.0,       # 곡선 추종 속도 확보를 위해 상향 (12 -> 26px)
            "max_slew_bot": 36.0,       # 하단 최대 속도 (18 -> 36px)
            "max_hold_frames": 16,      # 점선 홀드 프레임 확장
            "decay_factor": 0.95,
            "hough_threshold": 20,
            "min_line_length": 16,
            "max_line_gap": 50,
            "min_slope": 0.18,          # 완만한 곡선 선분 수용 (0.40 -> 0.18)
            "max_slope": 4.50,          # 수직에 가까운 차선 허용 (3.00 -> 4.50)
        },
        "roi": {
            "top_ratio": 0.60,          # 원거리 곡선까지 시야각 확장 (0.67 -> 0.60)
            "bottom_ratio": 0.88,
            "padding_x": 36             # 곡선 여유 마진 확장 (24 -> 36px)
        },
        "crosswalk": {
            "threshold": 195,           # 노면 도색 오탐 방지를 위해 상향 (175 -> 195)
            "hysteresis_enter": 3,      # 안정성 강화 (2 -> 3)
            "hysteresis_exit": 2
        }
    }

    run_video_pipeline(
        video_path=args.video,
        output_path=args.output,
        show_window=not args.no_show,
        cfg=custom_cfg,
        max_frames=args.max_frames
    )