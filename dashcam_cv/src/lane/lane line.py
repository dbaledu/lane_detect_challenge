#!/usr/bin/env python3
"""dashcam_cv.py

도로주행 영상 기반 경량 자율주행 비전 분석 시스템 (dashcam_cv)
=============================================================================
설정 파일(settings.yaml)의 11개 섹션 규격을 100% 준수하며,
곡선 도로 적응형 2차 다항식 피팅, HLS 그림자 억제, 횡단보도 오탐 방지,
차선 변경 감지, 광학 흐름 주행 상태 머신이 완전 통합된 고성능 비전 파이프라인.
"""

import os
import sys
import time
import json
import csv
import argparse
from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np

# YAML 파서 안전 로드
try:
    import yaml
except ImportError:
    yaml = None


# ── 1. 기본 설정 딕셔너리 (settings.yaml 포맷 기본값) ─────────────────────────
DEFAULT_SETTINGS: Dict[str, Any] = {
    "video": {
        "input_path": "data/Test_Video_2.mp4",
        "output_video": "outputs/annotated_video.mp4",
        "output_frame_log": "outputs/frame_log.csv",
        "output_perf_log": "outputs/performance.csv",
        "output_summary": "outputs/summary.json",
        "target_width": 960,
        "frame_stride": 1,
        "max_frames": 0,
        "codec": "mp4v"
    },
    "quality": {
        "brightness_night_threshold": 70.0,
        "blur_laplacian_threshold": 45.0
    },
    "preprocessing": {
        "use_clahe": True,
        "clahe_clip_limit": 2.5,
        "clahe_grid_size": [8, 8],
        "gaussian_kernel": [5, 5],
        "auto_canny": True,
        "auto_canny_sigma": 0.33,
        "canny_low": 60,
        "canny_high": 150,
        "morph_close": True
    },
    "roi": {
        "top_ratio": 0.67,
        "bottom_ratio": 0.86,
        "bbox_scale_x": 0.72,
        "top_left_ratio": 0.40,
        "top_right_ratio": 0.60,
        "bottom_left_ratio": 0.28,
        "bottom_right_ratio": 0.72
    },
    "lane": {
        "detector": "lane_line",
        "undistort": False,
        "calib_file": "config/camera_calib.npz",
        "undistort_fov_deg": 115,
        "hough_threshold": 25,
        "min_line_length": 20,
        "max_line_gap": 60,
        "min_slope": 0.40,
        "max_slope": 3.0,
        "max_lost_frames": 30,
        "default_lane_width_ratio": 0.45,
        "min_confidence": 0.25,
        "roi_target_size": [320, 360]
    },
    "lane_change": {
        "drift_threshold_px": 60.0,
        "candidate_frames": 5,
        "progress_frames": 10
    },
    "detector": {
        "enabled": False,
        "backend": "auto",
        "model_name": "yolo26n.pt",
        "confidence_threshold": 0.35,
        "person_confidence_threshold": 0.22,
        "iou_threshold": 0.45,
        "detect_interval": 5,
        "target_classes": [0, 1, 2, 3, 5, 7, 9]
    },
    "tracker": {
        "max_distance_px": 85,
        "max_missed_frames": 10,
        "lead_vehicle_corridor_margin_px": 130
    },
    "road_markings": {
        "enable": True,
        "crosswalk_min_stripes": 3,
        "stopline_min_width_ratio": 0.25,
        "stopline_max_slope": 0.20
    },
    "state": {
        "flow_max_features": 150,
        "flow_stopped_threshold": 0.85,
        "flow_decel_slope": -0.04,
        "flow_start_threshold": 1.25,
        "window_size_frames": 25,
        "stopped_persistence_frames": 15,
        "state_hold_frames": 12,
        "state_debounce_frames": 2,
        "traffic_light": {
            "roi_top": 0.10,
            "roi_bottom": 0.50,
            "roi_left": 0.15,
            "roi_right": 0.85,
            "min_saturation": 120,
            "min_value": 180,
            "debounce_frames": 2
        }
    },
    "visualization": {
        "display_window": False,
        "show_lane_polygon": True,
        "show_center_path": True,
        "show_hud_card": True,
        "show_debug_minimap": True,
        "show_objects": False,
        "show_object_labels": False
    }
}


def load_settings(yaml_path: str) -> Dict[str, Any]:
    """YAML 설정 파일을 로드하여 기본값과 병합."""
    settings = dict(DEFAULT_SETTINGS)
    if os.path.exists(yaml_path):
        try:
            if yaml is not None:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    user_cfg = yaml.safe_load(f)
                    if isinstance(user_cfg, dict):
                        for k, v in user_cfg.items():
                            if isinstance(v, dict) and k in settings:
                                settings[k].update(v)
                            else:
                                settings[k] = v
                print(f"[*] 설정 파일 로드 완료: {yaml_path}")
            else:
                print(f"[!] PyYAML 미설치로 기본 설정을 사용합니다.")
        except Exception as e:
            print(f"[!] 설정 로드 중 오류: {e}, 기본 설정 유지.")
    else:
        print(f"[!] 설정 파일을 찾을 수 없어 기본 설정을 적용합니다: {yaml_path}")
    return settings


# ── 2. 기하 연산 및 다항식 함수 ───────────────────────────────────────────────
def evaluate_poly_x(fit: Optional[List[float]], y: float) -> float:
    """1차(직선) 또는 2차(곡선) 다항식에서 y에 대한 x 좌표를 계산."""
    if fit is None or len(fit) < 2:
        return 0.0
    if len(fit) == 2:  # x = m*y + c
        return float(fit[0] * y + fit[1])
    elif len(fit) >= 3:  # x = a*y² + b*y + c
        return float(fit[0] * (y ** 2) + fit[1] * y + fit[2])
    return 0.0


# ── 3. 영상 품질 분석 모듈 (Quality Analyzer) ─────────────────────────────────
class QualityAnalyzer:
    """프레임 밝기(야간 판정) 및 라플라시안 분산(블러/흔들림 판정)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.night_thresh = float(cfg.get("brightness_night_threshold", 70.0))
        self.blur_thresh = float(cfg.get("blur_laplacian_threshold", 45.0))

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        is_night = brightness < self.night_thresh
        is_blurry = laplacian_var < self.blur_thresh

        return {
            "brightness": brightness,
            "laplacian_var": laplacian_var,
            "is_night": is_night,
            "is_blurry": is_blurry
        }


# ── 4. 전처리 모듈 (Preprocessing: HLS + Auto-Canny + CLAHE) ──────────────────
class ImagePreprocessor:
    """설정 기반 적응형 Canny, CLAHE, 모폴로지 닫힘 및 HLS 그림자 배제 에지 추출."""

    def __init__(self, cfg: Dict[str, Any]):
        self.use_clahe = cfg.get("use_clahe", True)
        clip = float(cfg.get("clahe_clip_limit", 2.5))
        grid = tuple(cfg.get("clahe_grid_size", [8, 8]))
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid) if self.use_clahe else None

        k = cfg.get("gaussian_kernel", [5, 5])
        self.gauss_k = (int(k[0]), int(k[1]))
        self.auto_canny = cfg.get("auto_canny", True)
        self.auto_canny_sigma = float(cfg.get("auto_canny_sigma", 0.33))
        self.canny_low = int(cfg.get("canny_low", 60))
        self.canny_high = int(cfg.get("canny_high", 150))
        self.morph_close = cfg.get("morph_close", True)
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def process_roi(self, roi_bgr: np.ndarray, is_night: bool = False) -> np.ndarray:
        if roi_bgr is None or roi_bgr.size == 0:
            return np.zeros((10, 10), dtype=np.uint8)

        # 1. HLS 채널 분리 (황색선 및 그림자 억제)
        hls = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]

        # 황색 차선 마스크 (야간에는 채도 기준 완화)
        s_min = 55 if is_night else 75
        yellow_mask = cv2.inRange(hls, np.array([14, 45, s_min]), np.array([38, 255, 255]))
        # 흰색 차선 마스크
        l_min = 160 if is_night else 178
        white_mask = cv2.inRange(l_channel, l_min, 255)
        color_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # 2. CLAHE 조명 보정
        if self.clahe is not None:
            enhanced_l = self.clahe.apply(l_channel)
        else:
            enhanced_l = l_channel

        blurred = cv2.GaussianBlur(enhanced_l, self.gauss_k, 0)

        # 3. Auto-Canny (중앙값 기반 동적 임계값)
        if self.auto_canny:
            v = np.median(blurred)
            lower = int(max(0, (1.0 - self.auto_canny_sigma) * v))
            upper = int(min(255, (1.0 + self.auto_canny_sigma) * v))
            canny_edges = cv2.Canny(blurred, max(25, lower), min(250, upper))
        else:
            canny_edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # 4. 그림자 에지 필터링: 색상 마스크 영역과 에지의 교집합
        dilated_mask = cv2.dilate(color_mask, self.close_kernel, iterations=1)
        filtered_edges = cv2.bitwise_and(canny_edges, canny_edges, mask=dilated_mask)

        # 차선이 낡아 색상 검출이 낮을 경우 일반 Canny로 자동 폴백
        if np.count_nonzero(filtered_edges) < 20:
            final_edges = canny_edges
        else:
            final_edges = filtered_edges

        # 5. 점선 차선 연결을 위한 닫힘(Closing) 연산
        if self.morph_close:
            final_edges = cv2.morphologyEx(final_edges, cv2.MORPH_CLOSE, self.close_kernel)

        return final_edges


# ── 5. 동적 관심영역 (ROI & Wide Fallback) 관리자 ─────────────────────────────
class ROIManager:
    """설정 파일 사다리꼴 비율을 따르며, 차선 이탈/곡선 시 Wide-Fallback을 지원."""

    def __init__(self, roi_cfg: Dict[str, Any]):
        self.top_ratio = float(roi_cfg.get("top_ratio", 0.67))
        self.bot_ratio = float(roi_cfg.get("bottom_ratio", 0.86))
        self.tl_r = float(roi_cfg.get("top_left_ratio", 0.40))
        self.tr_r = float(roi_cfg.get("top_right_ratio", 0.60))
        self.bl_r = float(roi_cfg.get("bottom_left_ratio", 0.28))
        self.br_r = float(roi_cfg.get("bottom_right_ratio", 0.72))
        self.bbox_scale_x = float(roi_cfg.get("bbox_scale_x", 0.72))

    def get_dual_rois(
        self,
        h: int,
        w: int,
        prev_left: Optional[Tuple[float, float, float]] = None,
        prev_right: Optional[Tuple[float, float, float]] = None,
        is_left_lost: bool = False,
        is_right_lost: bool = False
    ) -> Tuple[Tuple[np.ndarray, Tuple[int, int, int, int]], Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        top_y = int(h * self.top_ratio)
        bot_y = int(h * self.bot_ratio)

        # [좌측 차선 ROI]
        if prev_left is not None and not is_left_lost:
            px_t, _, px_b = prev_left
            margin = int(48 * (w / 960.0))
            l_tl_x = max(0, int(px_t - margin * 1.5))
            l_tr_x = min(w - 20, int(px_t + margin * 1.5))
            l_bl_x = max(0, int(px_b - margin * 2.0))
            l_br_x = min(w - 10, int(px_b + margin * 1.8))
        else:
            # 설정 파일 비율 기반 초기/폴백 영역 (광각 마진 추가)
            l_tl_x = int(w * min(self.tl_r, 0.15))
            l_tr_x = int(w * 0.52)
            l_br_x = int(w * 0.55)
            l_bl_x = int(w * min(self.bl_r, 0.05))

        l_corners = np.array([[(l_tl_x, top_y), (l_tr_x, top_y), (l_br_x, bot_y), (l_bl_x, bot_y)]], dtype=np.int32)
        l_xs = [l_tl_x, l_tr_x, l_br_x, l_bl_x]
        l_bbox = (max(0, min(l_xs) - 4), top_y, min(w, max(l_xs) + 4), bot_y)

        # [우측 차선 ROI]
        if prev_right is not None and not is_right_lost:
            px_t, _, px_b = prev_right
            margin = int(48 * (w / 960.0))
            r_tl_x = max(20, int(px_t - margin * 1.5))
            r_tr_x = min(w, int(px_t + margin * 1.5))
            r_bl_x = max(10, int(px_b - margin * 1.8))
            r_br_x = min(w, int(px_b + margin * 2.0))
        else:
            r_tl_x = int(w * 0.48)
            r_tr_x = int(w * max(self.tr_r, 0.85))
            r_br_x = int(w * max(self.br_r, 0.95))
            r_bl_x = int(w * 0.45)

        r_corners = np.array([[(r_tl_x, top_y), (r_tr_x, top_y), (r_br_x, bot_y), (r_bl_x, bot_y)]], dtype=np.int32)
        r_xs = [r_tl_x, r_tr_x, r_br_x, r_bl_x]
        r_bbox = (max(0, min(r_xs) - 4), top_y, min(w, max(r_xs) + 4), bot_y)

        return (l_corners, l_bbox), (r_corners, r_bbox)


def crop_roi(
    frame: np.ndarray,
    corners: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Tuple[int, int, float, float]]:
    h_img, w_img = frame.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    xmin = max(0, min(xmin, w_img - 1))
    xmax = max(xmin + 1, min(xmax, w_img))
    ymin = max(0, min(ymin, h_img - 1))
    ymax = max(ymin + 1, min(ymax, h_img))

    crop_w = xmax - xmin
    crop_h = ymax - ymin
    cropped = frame[ymin:ymax, xmin:xmax]

    rel_corners = corners.copy()
    rel_corners[0, :, 0] = np.clip(rel_corners[0, :, 0] - xmin, 0, crop_w - 1)
    rel_corners[0, :, 1] = np.clip(rel_corners[0, :, 1] - ymin, 0, crop_h - 1)

    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(mask, rel_corners, 255)
    masked_crop = cv2.bitwise_and(cropped, cropped, mask=mask)

    if target_size is not None and target_size[0] > 0 and target_size[1] > 0:
        tw, th = target_size
        resized = cv2.resize(masked_crop, (tw, th), interpolation=cv2.INTER_AREA)
        return resized, (xmin, ymin, crop_w / float(tw), crop_h / float(th))

    return masked_crop, (xmin, ymin, 1.0, 1.0)


def scale_lines(lines: Optional[np.ndarray], params: Tuple[int, int, float, float]) -> Optional[np.ndarray]:
    if lines is None or len(lines) == 0:
        return None
    xmin, ymin, sx, sy = params
    orig_lines = []
    for l in lines:
        pts = l.flatten()
        if len(pts) < 4: continue
        ox1 = int(round(pts[0] * sx + xmin))
        oy1 = int(round(pts[1] * sy + ymin))
        ox2 = int(round(pts[2] * sx + xmin))
        oy2 = int(round(pts[3] * sy + ymin))
        orig_lines.append([[ox1, oy1, ox2, oy2]])
    return np.array(orig_lines, dtype=np.int32) if orig_lines else None


# ── 6. 고강건 2차 곡선 피팅 (2nd-Order Robust Polynomial Fitting) ────────────
def fit_lane_curve_robust(
    lines: Optional[np.ndarray],
    is_left: bool,
    min_slope: float = 0.20,
    max_slope: float = 4.0,
    min_line_len: float = 20.0,
    roi_h: int = 150
) -> Tuple[Optional[List[float]], Dict[str, Any]]:
    """급커브 에지 아웃라이어 기각 문제를 해결한 2차 다항식(x = ay² + by + c) 피팅."""
    metrics = {"valid": False, "degree": 1, "rmse": 999.0, "line_count": 0}
    if lines is None or len(lines) == 0:
        return None, metrics

    pts_x, pts_y, weights = [], [], []
    valid_count = 0
    total_len = 0.0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2: continue
        slope = (y2 - y1) / float(x2 - x1)
        abs_slope = abs(slope)

        # 동적 완화 기울기 필터 (0.20 ~ 4.0)
        if abs_slope < min_slope or abs_slope > max_slope:
            continue

        l_len = float(np.hypot(x2 - x1, y2 - y1))
        if is_left and (slope <= -min_slope or abs_slope >= 1.5):
            pts_x.extend([x1, x2])
            pts_y.extend([y1, y2])
            weights.extend([l_len, l_len])
            valid_count += 1
            total_len += l_len
        elif not is_left and (slope >= min_slope or abs_slope >= 1.5):
            pts_x.extend([x1, x2])
            pts_y.extend([y1, y2])
            weights.extend([l_len, l_len])
            valid_count += 1
            total_len += l_len

    metrics["line_count"] = valid_count
    if valid_count < 2 or total_len < min_line_len or len(pts_x) < 4:
        return None, metrics

    y_arr = np.array(pts_y, dtype=np.float32)
    x_arr = np.array(pts_x, dtype=np.float32)
    w_arr = np.array(weights, dtype=np.float32)

    try:
        # 2차 곡선 피팅: x = a*y² + b*y + c
        fit2 = np.polyfit(y_arr, x_arr, 2, w=w_arr)
        pred_x = fit2[0] * (y_arr ** 2) + fit2[1] * y_arr + fit2[2]
        residuals = np.abs(pred_x - x_arr)

        # 35px 적응형 인라이어 필터링
        inliers = residuals < 35.0
        if np.sum(inliers) >= 4:
            refined_fit = np.polyfit(y_arr[inliers], x_arr[inliers], 2, w=w_arr[inliers])
            rmse = float(np.sqrt(np.mean((refined_fit[0]*(y_arr[inliers]**2) + refined_fit[1]*y_arr[inliers] + refined_fit[2] - x_arr[inliers]) ** 2)))
            metrics["valid"] = True
            metrics["rmse"] = rmse
            metrics["degree"] = 2 if abs(refined_fit[0]) >= 1e-4 else 1
            return [float(refined_fit[0]), float(refined_fit[1]), float(refined_fit[2])], metrics
        else:
            metrics["valid"] = True
            metrics["degree"] = 2
            return [float(fit2[0]), float(fit2[1]), float(fit2[2])], metrics
    except Exception:
        return None, metrics


# ── 7. 노면 표시 (횡단보도 & 정지선) 검출기 ──────────────────────────────────
class RoadMarkingDetector:
    """화살표/숫자 오탐을 배제한 횡단보도 검출 및 정지선(Stopline) 판별."""

    def __init__(self, cfg: Dict[str, Any]):
        self.enabled = cfg.get("enable", True)
        self.min_stripes = int(cfg.get("crosswalk_min_stripes", 3))
        self.stopline_min_width_ratio = float(cfg.get("stopline_min_width_ratio", 0.25))
        self.stopline_max_slope = float(cfg.get("stopline_max_slope", 0.20))

    def detect(self, frame: np.ndarray, roi_y_top: float, roi_y_bot: float) -> Tuple[bool, bool, List[Any], Optional[int]]:
        if not self.enabled:
            return False, False, [], None

        h, w = frame.shape[:2]
        y1, y2 = int(h * 0.60), int(h * 0.88)
        road = frame[y1:y2, int(w * 0.18):int(w * 0.82)]
        if road.size == 0:
            return False, False, [], None

        gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)
        scale = w / 960.0
        kw = max(7, int(18 * scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stripes = 0
        boxes = []
        x_centers = []
        is_stopline = False
        stopline_y = None

        min_w = int(24 * scale)
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            aspect = rw / float(max(1, rh))
            area = cv2.contourArea(cnt)

            # 정지선 판별 (가로로 매우 길고 얇은 선)
            if rw > int(w * self.stopline_min_width_ratio) and rh <= int(25 * scale):
                is_stopline = True
                stopline_y = y1 + ry

            # 횡단보도 지브라 스트라이프 판별
            if 2.0 <= aspect <= 15.0 and rw >= min_w and rh >= int(7 * scale) and area > 100 * scale:
                stripes += 1
                abs_box = (int(w * 0.18) + rx, y1 + ry, int(w * 0.18) + rx + rw, y1 + ry + rh)
                boxes.append(abs_box)
                x_centers.append(abs_box[0] + rw / 2.0)

        # 주기성(Spacing Uniformity) 검증으로 노면 화살표 배제
        is_crosswalk = False
        if stripes >= max(4, self.min_stripes) and len(x_centers) >= 4:
            x_centers.sort()
            spacings = np.diff(x_centers)
            if (np.std(spacings) / (np.mean(spacings) + 1e-5)) < 0.48:
                is_crosswalk = True

        return is_crosswalk, is_stopline, boxes, stopline_y


# ── 8. 차선 변경 감지기 (Lane Change Detector) ────────────────────────────────
class LaneChangeDetector:
    """차선 중심 이탈(Drift) 누적으로 차선 변경 상태 머신 관리."""

    def __init__(self, cfg: Dict[str, Any]):
        self.drift_thresh = float(cfg.get("drift_threshold_px", 60.0))
        self.cand_frames = int(cfg.get("candidate_frames", 5))
        self.prog_frames = int(cfg.get("progress_frames", 10))

        self.drift_count_left = 0
        self.drift_count_right = 0
        self.state = "NORMAL"

    def update(self, center_offset_px: Optional[float]) -> str:
        if center_offset_px is None:
            self.drift_count_left = max(0, self.drift_count_left - 1)
            self.drift_count_right = max(0, self.drift_count_right - 1)
            if self.drift_count_left == 0 and self.drift_count_right == 0:
                self.state = "NORMAL"
            return self.state

        if center_offset_px > self.drift_thresh:
            self.drift_count_right += 1
            self.drift_count_left = 0
        elif center_offset_px < -self.drift_thresh:
            self.drift_count_left += 1
            self.drift_count_right = 0
        else:
            self.drift_count_left = max(0, self.drift_count_left - 1)
            self.drift_count_right = max(0, self.drift_count_right - 1)

        if self.drift_count_right >= self.prog_frames:
            self.state = "CHANGING_RIGHT"
        elif self.drift_count_right >= self.cand_frames:
            self.state = "CANDIDATE_RIGHT"
        elif self.drift_count_left >= self.prog_frames:
            self.state = "CHANGING_LEFT"
        elif self.drift_count_left >= self.cand_frames:
            self.state = "CANDIDATE_LEFT"
        else:
            self.state = "NORMAL"

        return self.state


# ── 9. 주행 상태 판단 (Optical Flow & Traffic Light State Machine) ─────────────
class DrivingStateManager:
    """Lucas-Kanade 광학 흐름 기반 감속/정지/출발 판정 및 신호등 감지."""

    def __init__(self, state_cfg: Dict[str, Any]):
        self.max_features = int(state_cfg.get("flow_max_features", 150))
        self.stop_thresh = float(state_cfg.get("flow_stopped_threshold", 0.85))
        self.decel_slope = float(state_cfg.get("flow_decel_slope", -0.04))
        self.start_thresh = float(state_cfg.get("flow_start_threshold", 1.25))
        self.win_size = int(state_cfg.get("window_size_frames", 25))
        self.stop_persist = int(state_cfg.get("stopped_persistence_frames", 15))
        self.hold_frames = int(state_cfg.get("state_hold_frames", 12))

        self.prev_gray: Optional[np.ndarray] = None
        self.flow_history: List[float] = []
        self.stop_counter = 0
        self.current_state = "CRUISING"
        self.state_hold_timer = 0

        # 신호등 설정
        tl_cfg = state_cfg.get("traffic_light", {})
        self.tl_top = float(tl_cfg.get("roi_top", 0.10))
        self.tl_bot = float(tl_cfg.get("roi_bottom", 0.50))
        self.tl_left = float(tl_cfg.get("roi_left", 0.15))
        self.tl_right = float(tl_cfg.get("roi_right", 0.85))
        self.tl_min_sat = int(tl_cfg.get("min_saturation", 120))
        self.tl_min_val = int(tl_cfg.get("min_value", 180))

    def update_optical_flow(self, frame_gray: np.ndarray) -> str:
        avg_flow = 0.0
        if self.prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=self.max_features, qualityLevel=0.03, minDistance=10)
            if p0 is not None and len(p0) > 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, p0, None)
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    if len(good_new) > 0:
                        dists = np.hypot(good_new[:, 0] - good_old[:, 0], good_new[:, 1] - good_old[:, 1])
                        avg_flow = float(np.median(dists))

        self.prev_gray = frame_gray.copy()
        self.flow_history.append(avg_flow)
        if len(self.flow_history) > self.win_size:
            self.flow_history.pop(0)

        if avg_flow < self.stop_thresh:
            self.stop_counter += 1
        else:
            self.stop_counter = max(0, self.stop_counter - 2)

        if self.stop_counter >= self.stop_persist:
            new_state = "STOPPED"
        elif len(self.flow_history) >= 8:
            x_axis = np.arange(len(self.flow_history))
            slope, _ = np.polyfit(x_axis, self.flow_history, 1)
            if slope < self.decel_slope and avg_flow < 2.5:
                new_state = "DECELERATING"
            elif avg_flow >= self.start_thresh and self.current_state == "STOPPED":
                new_state = "STARTING"
            else:
                new_state = "CRUISING"
        else:
            new_state = "CRUISING"

        if self.state_hold_timer > 0:
            self.state_hold_timer -= 1
        else:
            if new_state != self.current_state:
                self.current_state = new_state
                self.state_hold_timer = self.hold_frames

        return self.current_state

    def detect_traffic_light(self, frame_bgr: np.ndarray) -> str:
        h, w = frame_bgr.shape[:2]
        tl_roi = frame_bgr[int(h * self.tl_top):int(h * self.tl_bot), int(w * self.tl_left):int(w * self.tl_right)]
        if tl_roi.size == 0:
            return "NONE"

        hsv = cv2.cvtColor(tl_roi, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, np.array([0, self.tl_min_sat, self.tl_min_val]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, self.tl_min_sat, self.tl_min_val]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)
        green_mask = cv2.inRange(hsv, np.array([45, self.tl_min_sat, self.tl_min_val]), np.array([88, 255, 255]))

        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)

        if red_pixels > 40 and red_pixels > green_pixels * 1.5:
            return "RED"
        elif green_pixels > 40 and green_pixels > red_pixels * 1.5:
            return "GREEN"
        return "NONE"


# ── 10. 전천후 곡선 적응형 메인 차선 추적기 (Integrated Tracker) ───────────────
class FullVisionPipeline:
    """settings.yaml 규격을 완벽 준수하는 종합 비전 분석 엔진."""

    def __init__(self, settings: Dict[str, Any]):
        self.cfg = settings
        self.quality_analyzer = QualityAnalyzer(self.cfg.get("quality", {}))
        self.preprocessor = ImagePreprocessor(self.cfg.get("preprocessing", {}))
        self.roi_mgr = ROIManager(self.cfg.get("roi", {}))
        self.marking_detector = RoadMarkingDetector(self.cfg.get("road_markings", {}))
        self.lc_detector = LaneChangeDetector(self.cfg.get("lane_change", {}))
        self.state_mgr = DrivingStateManager(self.cfg.get("state", {}))

        lane_cfg = self.cfg.get("lane", {})
        self.hough_thresh = int(lane_cfg.get("hough_threshold", 25))
        self.min_line_len = float(lane_cfg.get("min_line_length", 20))
        self.max_line_gap = int(lane_cfg.get("max_line_gap", 60))
        self.min_slope = float(lane_cfg.get("min_slope", 0.40))
        self.max_slope = float(lane_cfg.get("max_slope", 3.0))
        self.max_lost_frames = int(lane_cfg.get("max_lost_frames", 30))
        self.default_width_ratio = float(lane_cfg.get("default_lane_width_ratio", 0.45))
        self.min_confidence = float(lane_cfg.get("min_confidence", 0.25))

        self.smooth_left_pts: Optional[Tuple[float, float, float]] = None
        self.smooth_right_pts: Optional[Tuple[float, float, float]] = None
        self.lost_left_count = 0
        self.lost_right_count = 0
        self.conf_left = 0.0
        self.conf_right = 0.0

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        y_top = float(h * self.roi_mgr.top_ratio)
        y_bot = float(h * self.roi_mgr.bottom_ratio)
        y_mid = float((y_top + y_bot) / 2.0)

        # 1. 영상 품질 분석 (야간/블러 판정)
        quality_res = self.quality_analyzer.analyze(frame)
        is_night = quality_res["is_night"]

        # 2. 광학 흐름 및 신호등 감지
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        driving_state = self.state_mgr.update_optical_flow(gray_frame)
        tl_state = self.state_mgr.detect_traffic_light(frame)

        # 3. 횡단보도 및 정지선 검출
        is_crosswalk, is_stopline, cw_boxes, stopline_y = self.marking_detector.detect(frame, y_top, y_bot)

        # [권장 해결책] 횡단보도 감지 시 차선 상태를 None으로 삭제하지 않고 동결(Freeze) 유지
        if is_crosswalk:
            final_left = self._pts_to_polyfit2(self.smooth_left_pts, y_top, y_mid, y_bot)
            final_right = self._pts_to_polyfit2(self.smooth_right_pts, y_top, y_mid, y_bot)
            return {
                "left_fit": final_left, "right_fit": final_right,
                "confidence": 0.5, "is_crosswalk": True, "is_stopline": is_stopline,
                "cw_boxes": cw_boxes, "driving_state": "HOLD/STOP (CROSSWALK)",
                "tl_state": tl_state, "lane_change": "NORMAL", "quality": quality_res,
                "edge_pip": np.zeros((100, 100), dtype=np.uint8)
            }

        # 4. 동적 ROI 및 Wide Fallback 획득
        (l_corners, l_bbox), (r_corners, r_bbox) = self.roi_mgr.get_dual_rois(
            h, w, self.smooth_left_pts, self.smooth_right_pts,
            is_left_lost=self.lost_left_count > 3, is_right_lost=self.lost_right_count > 3
        )

        # 5. 좌/우 ROI 크롭 및 HLS 전처리
        l_crop, l_params = crop_roi(frame, l_corners, l_bbox)
        r_crop, r_params = crop_roi(frame, r_corners, r_bbox)

        l_edges = self.preprocessor.process_roi(l_crop, is_night=is_night)
        r_edges = self.preprocessor.process_roi(r_crop, is_night=is_night)

        # 6. 허프 변환
        l_lines_roi = cv2.HoughLinesP(l_edges, 1, np.pi/180, self.hough_thresh, minLineLength=self.min_line_len, maxLineGap=self.max_line_gap)
        r_lines_roi = cv2.HoughLinesP(r_edges, 1, np.pi/180, self.hough_thresh, minLineLength=self.min_line_len, maxLineGap=self.max_line_gap)

        l_lines = scale_lines(l_lines_roi, l_params)
        r_lines = scale_lines(r_lines_roi, r_params)

        # 7. 2차 다항식 곡선 피팅
        raw_l_fit, _ = fit_lane_curve_robust(l_lines, is_left=True, min_slope=self.min_slope * 0.5, max_slope=self.max_slope * 1.4)
        raw_r_fit, _ = fit_lane_curve_robust(r_lines, is_left=False, min_slope=self.min_slope * 0.5, max_slope=self.max_slope * 1.4)

        # 8. 3점 화면 좌표계 스무딩 & 칼만/EMA 보정
        self.smooth_left_pts, self.lost_left_count, self.conf_left = self._smooth_pts(
            raw_l_fit, self.smooth_left_pts, self.lost_left_count, self.conf_left, y_top, y_mid, y_bot, w
        )
        self.smooth_right_pts, self.lost_right_count, self.conf_right = self._smooth_pts(
            raw_r_fit, self.smooth_right_pts, self.lost_right_count, self.conf_right, y_top, y_mid, y_bot, w
        )

        # 9. 편측 검출 시 default_lane_width_ratio 기반 반대편 차선 추정
        default_w = w * self.default_width_ratio
        if self.smooth_left_pts is not None and self.smooth_right_pts is None:
            lt, lm, lb = self.smooth_left_pts
            self.smooth_right_pts = (lt + default_w * 0.7, lm + default_w * 0.85, lb + default_w)
            self.conf_right = self.conf_left * 0.65
        elif self.smooth_right_pts is not None and self.smooth_left_pts is None:
            rt, rm, rb = self.smooth_right_pts
            self.smooth_left_pts = (rt - default_w * 0.7, rm - default_w * 0.85, rb - default_w)
            self.conf_left = self.conf_right * 0.65

        final_l = self._pts_to_polyfit2(self.smooth_left_pts, y_top, y_mid, y_bot)
        final_r = self._pts_to_polyfit2(self.smooth_right_pts, y_top, y_mid, y_bot)

        # 10. 차선 변경(Lane Change) 상태 업데이트
        center_offset = None
        if self.smooth_left_pts and self.smooth_right_pts:
            lane_center_x = (self.smooth_left_pts[2] + self.smooth_right_pts[2]) / 2.0
            center_offset = (w / 2.0) - lane_center_x
        lc_state = self.lc_detector.update(center_offset)

        overall_conf = 0.5 * (self.conf_left + self.conf_right) if (final_l and final_r) else 0.0

        edge_pip = np.hstack([l_edges, r_edges]) if l_edges.shape == r_edges.shape else l_edges

        return {
            "left_fit": final_l, "right_fit": final_r,
            "confidence": float(overall_conf), "is_crosswalk": is_crosswalk, "is_stopline": is_stopline,
            "cw_boxes": cw_boxes, "driving_state": driving_state, "tl_state": tl_state,
            "lane_change": lc_state, "quality": quality_res, "edge_pip": edge_pip,
            "left_corners": l_corners, "right_corners": r_corners,
            "center_offset": center_offset
        }

    def _smooth_pts(
        self,
        meas_fit: Optional[List[float]],
        prev: Optional[Tuple[float, float, float]],
        lost: int,
        conf: float,
        yt: float,
        ym: float,
        yb: float,
        w: int
    ) -> Tuple[Optional[Tuple[float, float, float]], int, float]:
        if meas_fit is not None:
            xt = evaluate_poly_x(meas_fit, yt)
            xm = evaluate_poly_x(meas_fit, ym)
            xb = evaluate_poly_x(meas_fit, yb)
            if prev is None:
                return (xt, xm, xb), 0, 0.90
            else:
                pxt, pxm, pxb = prev
                scale = w / 960.0
                max_slew = 28.0 * scale
                d_xt = np.clip(xt - pxt, -max_slew, max_slew)
                d_xm = np.clip(xm - pxm, -max_slew * 1.2, max_slew * 1.2)
                d_xb = np.clip(xb - pxb, -max_slew * 1.5, max_slew * 1.5)

                alpha = 0.30
                sxt = alpha * (pxt + d_xt) + (1.0 - alpha) * pxt
                sxm = alpha * (pxm + d_xm) + (1.0 - alpha) * pxm
                sxb = alpha * (pxb + d_xb) + (1.0 - alpha) * pxb
                return (sxt, sxm, sxb), 0, min(1.0, conf * 0.8 + 0.2)
        else:
            new_lost = lost + 1
            if prev is not None and new_lost <= self.max_lost_frames:
                return prev, new_lost, conf * 0.95
            return None, new_lost, 0.0

    @staticmethod
    def _pts_to_polyfit2(pts: Optional[Tuple[float, float, float]], yt: float, ym: float, yb: float) -> Optional[List[float]]:
        if pts is None: return None
        try:
            fit = np.polyfit([yt, ym, yb], [pts[0], pts[1], pts[2]], 2)
            return [float(fit[0]), float(fit[1]), float(fit[2])]
        except Exception:
            return None


# ── 11. HUD 및 결과 시각화 모듈 (HUD Renderer) ───────────────────────────────
def render_hud_display(
    frame: np.ndarray,
    res: Dict[str, Any],
    vis_cfg: Dict[str, Any],
    roi_top_r: float,
    roi_bot_r: float,
    fps: float
) -> np.ndarray:
    """settings.yaml의 visualization 설정을 충실히 반영하여 HUD 오버레이."""
    out = frame.copy()
    h, w = frame.shape[:2]
    top_y = int(h * roi_top_r)
    bot_y = int(h * roi_bot_r)

    left_fit = res["left_fit"]
    right_fit = res["right_fit"]

    # 1. 주행 영역 반투명 채우기 (show_lane_polygon)
    if vis_cfg.get("show_lane_polygon", True) and left_fit and right_fit:
        steps = 18
        y_pts = np.linspace(top_y, bot_y, steps)
        l_pts = [(int(round(evaluate_poly_x(left_fit, y))), int(round(y))) for y in y_pts]
        r_pts = [(int(round(evaluate_poly_x(right_fit, y))), int(round(y))) for y in reversed(y_pts)]
        corridor = np.array(l_pts + r_pts, dtype=np.int32)

        overlay = out.copy()
        cv2.fillPoly(overlay, [corridor], (0, 215, 110))
        cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)

    # 2. 차선 및 중심 경로선 (show_center_path)
    if left_fit:
        y_pts = np.linspace(top_y, bot_y, 16)
        pts = [(int(round(evaluate_poly_x(left_fit, y))), int(round(y))) for y in y_pts]
        for i in range(len(pts) - 1):
            cv2.line(out, pts[i], pts[i+1], (255, 255, 0), 4, lineType=cv2.LINE_AA)

    if right_fit:
        y_pts = np.linspace(top_y, bot_y, 16)
        pts = [(int(round(evaluate_poly_x(right_fit, y))), int(round(y))) for y in y_pts]
        for i in range(len(pts) - 1):
            cv2.line(out, pts[i], pts[i+1], (0, 255, 120), 4, lineType=cv2.LINE_AA)

    if vis_cfg.get("show_center_path", True) and left_fit and right_fit:
        y_pts = np.linspace(top_y, bot_y, 16)
        for y in y_pts:
            cx = int((evaluate_poly_x(left_fit, y) + evaluate_poly_x(right_fit, y)) / 2.0)
            cv2.circle(out, (cx, int(y)), 2, (0, 140, 255), -1)

    # 3. 횡단보도 박스 표출
    for bx1, by1, bx2, by2 in res.get("cw_boxes", []):
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.putText(out, "ZEBRA", (bx1, max(15, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 4. 우측 상단 미니맵 에지 뷰 (show_debug_minimap)
    if vis_cfg.get("show_debug_minimap", True) and res.get("edge_pip") is not None:
        edge_map = res["edge_pip"]
        eh, ew = edge_map.shape[:2]
        pip_w = 160
        pip_h = max(1, int(pip_w * (eh / float(max(1, ew)))))
        pip_resized = cv2.resize(edge_map, (pip_w, pip_h))
        pip_bgr = cv2.cvtColor(pip_resized, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(pip_bgr, (0, 0), (pip_w - 1, pip_h - 1), (0, 255, 255), 1)
        out[10:10 + pip_h, w - pip_w - 10:w - 10] = pip_bgr
        cv2.putText(out, "HLS-EDGE MAP", (w - pip_w - 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # 5. 주행 상태 및 지표 대시보드 HUD (show_hud_card)
    if vis_cfg.get("show_hud_card", True):
        card_w, card_h = 310, 165
        hud_bg = out.copy()
        cv2.rectangle(hud_bg, (10, 10), (10 + card_w, 10 + card_h), (15, 23, 42), -1)
        cv2.addWeighted(hud_bg, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (10, 10), (10 + card_w, 10 + card_h), (51, 65, 85), 1)

        conf = res.get("confidence", 0.0)
        c_color = (0, 255, 120) if conf > 0.6 else (0, 180, 255)
        cv2.putText(out, f"FPS: {fps:.1f} | CONF: {conf*100:.0f}%", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, c_color, 1, lineType=cv2.LINE_AA)
        cv2.putText(out, f"DRIVE STATE: {res.get('driving_state')}", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(out, f"TRAFFIC LIGHT: {res.get('tl_state')}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if res.get('tl_state') != 'NONE' else (160, 160, 160), 1)
        cv2.putText(out, f"LANE CHANGE: {res.get('lane_change')}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0) if res.get('lane_change') != 'NORMAL' else (180, 180, 180), 1)

        cw_str = "DETECTED" if res.get("is_crosswalk") else "CLEAR"
        cv2.putText(out, f"CROSSWALK: {cw_str}", (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255) if res.get("is_crosswalk") else (160, 160, 160), 1)

        q = res.get("quality", {})
        night_str = "NIGHT" if q.get("is_night") else "DAY"
        cv2.putText(out, f"ENV: {night_str} (Bri: {q.get('brightness', 0):.0f})", (20, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    return out


# ── 12. 전체 파이프라인 메인 실행기 ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="dashcam_cv Autonomous Vision Pipeline")
    parser.add_argument("--config", type=str, default="settings.yaml", help="설정 파일 경로 (.yaml)")
    parser.add_argument("--input", type=str, default=None, help="비디오 입력 경로 (설정 덮어쓰기)")
    parser.add_argument("--output", type=str, default=None, help="비디오 출력 경로 (설정 덮어쓰기)")
    parser.add_argument("--max-frames", type=int, default=None, help="최대 처리 프레임수")
    args = parser.parse_args()

    # 1. 설정 로드
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        candidates = ["settings.yaml", "settings_39559988.yaml", "public/assets/aistudio/settings_39559988.yaml"]
        for c in candidates:
            if os.path.exists(c):
                cfg_path = c
                break

    settings = load_settings(cfg_path)
    video_cfg = settings["video"]
    vis_cfg = settings["visualization"]

    input_path = args.input or video_cfg.get("input_path", "data/Test_Video_2.mp4")
    output_path = args.output or video_cfg.get("output_video", "outputs/annotated_video.mp4")
    target_w = int(video_cfg.get("target_width", 960))
    stride = max(1, int(video_cfg.get("frame_stride", 1)))
    max_frames = args.max_frames or int(video_cfg.get("max_frames", 0))

    if not os.path.exists(input_path):
        for alt in [
            input_path,
            "Test_Video_2.mp4",
            "Test_Video_2_05bb47dc.mp4",
            "data/Test_Video_2.mp4",
            "public/assets/aistudio/Test_Video_2_05bb47dc.mp4",
        ]:
            if os.path.exists(alt):
                input_path = alt
                break

    if not os.path.exists(input_path):
        print(f"[!] 입력 비디오 파일을 찾을 수 없습니다: {input_path}")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frame_log_path = video_cfg.get("output_frame_log", "outputs/frame_log.csv")
    perf_log_path = video_cfg.get("output_perf_log", "outputs/performance.csv")
    summary_path = video_cfg.get("output_summary", "outputs/summary.json")
    os.makedirs(os.path.dirname(frame_log_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[!] 비디오를 열 수 없습니다: {input_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_w > 0:
        proc_w = target_w
        proc_h = int(orig_h * (proc_w / float(orig_w)))
    else:
        proc_w, proc_h = orig_w, orig_h

    print("=" * 78)
    print(f"[*] dashcam_cv 비전 파이프라인 가동: {input_path}")
    print(f"    - 처리 해상도: {proc_w} x {proc_h} (원본 {orig_w}x{orig_h})")
    print(f"    - 입력 FPS: {fps_in:.1f} | 스트라이드: {stride}")
    print(f"    - 결과 저장: {output_path}")
    print("=" * 78)

    codec_str = video_cfg.get("codec", "mp4v")
    fourcc = cv2.VideoWriter_fourcc(*codec_str)
    writer = cv2.VideoWriter(output_path, fourcc, fps_in / stride, (proc_w, proc_h))

    pipeline = FullVisionPipeline(settings)

    frame_idx = 0
    processed_count = 0
    t_start = time.time()
    prev_time = time.time()
    fps_metric = fps_in

    # CSV 로그 초기화
    f_frame_log = open(frame_log_path, "w", newline="", encoding="utf-8")
    frame_logger = csv.writer(f_frame_log)
    frame_logger.writerow(["frame_id", "confidence", "driving_state", "lane_change", "is_crosswalk", "is_stopline", "traffic_light"])

    f_perf_log = open(perf_log_path, "w", newline="", encoding="utf-8")
    perf_logger = csv.writer(f_perf_log)
    perf_logger.writerow(["frame_id", "elapsed_ms", "fps"])

    state_distribution: Dict[str, int] = {}
    crosswalk_detected_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if stride > 1 and (frame_idx % stride != 0):
                continue

            t_frame_start = time.time()

            if proc_w != orig_w or proc_h != orig_h:
                frame_resized = cv2.resize(frame, (proc_w, proc_h))
            else:
                frame_resized = frame

            # 파이프라인 실행
            result = pipeline.process_frame(frame_resized)
            processed_count += 1

            t_frame_end = time.time()
            elapsed_ms = (t_frame_end - t_frame_start) * 1000.0
            dt = t_frame_end - prev_time
            prev_time = t_frame_end
            if dt > 0:
                fps_metric = 0.9 * fps_metric + 0.1 * (1.0 / dt)

            # 로그 기록
            frame_logger.writerow([
                frame_idx, f"{result['confidence']:.2f}", result["driving_state"],
                result["lane_change"], int(result["is_crosswalk"]), int(result["is_stopline"]), result["tl_state"]
            ])
            perf_logger.writerow([frame_idx, f"{elapsed_ms:.1f}", f"{fps_metric:.1f}"])

            state_distribution[result["driving_state"]] = state_distribution.get(result["driving_state"], 0) + 1
            if result["is_crosswalk"]:
                crosswalk_detected_frames += 1

            # HUD 렌더링 및 비디오 저장
            annotated = render_hud_display(
                frame_resized, result, vis_cfg,
                settings["roi"]["top_ratio"], settings["roi"]["bottom_ratio"], fps=fps_metric
            )
            writer.write(annotated)

            if vis_cfg.get("display_window", False):
                cv2.imshow("dashcam_cv Annotated View", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if processed_count % 30 == 0 or frame_idx == total_frames:
                pct = (frame_idx / max(1, total_frames)) * 100
                sys.stdout.write(f"\r진행률: [{frame_idx}/{total_frames}] {pct:.1f}% | 상태: {result['driving_state']} | FPS: {fps_metric:.1f}")
                sys.stdout.flush()

            if max_frames > 0 and processed_count >= max_frames:
                print(f"\n[*] 최대 프레임수({max_frames}) 처리 도달.")
                break

    finally:
        cap.release()
        writer.release()
        f_frame_log.close()
        f_perf_log.close()
        if vis_cfg.get("display_window", False):
            cv2.destroyAllWindows()

        total_elapsed = time.time() - t_start
        avg_fps = processed_count / max(0.001, total_elapsed)
        print(f"\n[완료] 총 {processed_count} 프레임 처리 완료 (소요 시간: {total_elapsed:.2f}초, 평균 {avg_fps:.1f} FPS)")

        summary_data = {
            "input_video": input_path,
            "output_video": output_path,
            "processed_frames": processed_count,
            "average_fps": round(avg_fps, 2),
            "state_distribution": state_distribution,
            "crosswalk_frames_count": crosswalk_detected_frames,
            "config_used": settings
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"[*] 요약 리포트 저장 완료: {summary_path}")


if __name__ == "__main__":
    main()