"""Lane detection module based on cv1021.py dual-ROI, Canny edge detection, and Hough transform."""
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from ..common.geometry import compute_slope, evaluate_poly_x


def shrink_bounding_box(
    bbox: Tuple[int, int, int, int],
    scale_x: float = 0.72,
    scale_y: float = 0.96,
    w_limit: Optional[int] = None,
    h_limit: Optional[int] = None
) -> Tuple[int, int, int, int]:
    """바운딩 박스가 너무 넓게 잡히는 것을 방지하기 위해 중심 기준으로 폭 축소."""
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    half_w = ((xmax - xmin) * scale_x) / 2.0
    half_h = ((ymax - ymin) * scale_y) / 2.0

    new_xmin = int(round(cx - half_w))
    new_xmax = int(round(cx + half_w))
    new_ymin = int(round(cy - half_h))
    new_ymax = int(round(cy + half_h))

    if w_limit is not None:
        new_xmin = max(0, min(new_xmin, w_limit - 1))
        new_xmax = max(new_xmin + 1, min(new_xmax, w_limit))
    if h_limit is not None:
        new_ymin = max(0, min(new_ymin, h_limit - 1))
        new_ymax = max(new_ymin + 1, min(new_ymax, h_limit))

    return (new_xmin, new_ymin, new_xmax, new_ymax)


def get_dual_roi_geometry(
    h: int,
    w: int,
    top_ratio: float = 0.67,
    bot_ratio: float = 0.86,
    bbox_scale_x: float = 0.72
) -> Tuple[Tuple[np.ndarray, Tuple[int, int, int, int]], Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """축소된 슬림형 듀얼 ROI 및 컴팩트 Bounding Box 산출.

    1. 좌측 차선 ROI (Left Lane):
       - 상단: y=0.67h, x=0.40w ~ 0.46w (폭 0.06w)
       - 하단: y=0.86h, x=0.28w ~ 0.39w (폭 0.11w)
    2. 우측 차선 ROI (Right Lane):
       - 상단: y=0.67h, x=0.54w ~ 0.60w (폭 0.06w)
       - 하단: y=0.86h, x=0.61w ~ 0.72w (폭 0.11w)
    """
    top_y = int(h * top_ratio)
    bot_y = int(h * bot_ratio)

    # 1. 좌측 차선 슬림 사다리꼴
    left_tl = (int(w * 0.40), top_y)
    left_tr = (int(w * 0.46), top_y)
    left_br = (int(w * 0.39), bot_y)
    left_bl = (int(w * 0.28), bot_y)
    left_corners = np.array([[left_tl, left_tr, left_br, left_bl]], dtype=np.int32)

    raw_left_bbox = (int(w * 0.28), top_y, int(w * 0.46), bot_y)
    left_bbox = shrink_bounding_box(raw_left_bbox, scale_x=bbox_scale_x, scale_y=0.98, w_limit=w, h_limit=h)

    # 2. 우측 차선 슬림 사다리꼴
    right_tl = (int(w * 0.54), top_y)
    right_tr = (int(w * 0.60), top_y)
    right_br = (int(w * 0.72), bot_y)
    right_bl = (int(w * 0.61), bot_y)
    right_corners = np.array([[right_tl, right_tr, right_br, right_bl]], dtype=np.int32)

    raw_right_bbox = (int(w * 0.54), top_y, int(w * 0.72), bot_y)
    right_bbox = shrink_bounding_box(raw_right_bbox, scale_x=bbox_scale_x, scale_y=0.98, w_limit=w, h_limit=h)

    return (left_corners, left_bbox), (right_corners, right_bbox)


def crop_and_resize_roi(
    frame: np.ndarray,
    roi_corners: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (320, 360)
) -> Tuple[np.ndarray, Tuple[int, int, float, float]]:
    """축소된 Bounding Box 기준으로 슬라이싱 후 target_size(320x360)로 축소."""
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

    target_w, target_h = target_size
    resized_roi = cv2.resize(masked_crop, target_size, interpolation=cv2.INTER_AREA)

    scale_x = crop_w / float(target_w)
    scale_y = crop_h / float(target_h)

    return resized_roi, (xmin, ymin, scale_x, scale_y)


def preprocess_roi(roi_img: np.ndarray) -> np.ndarray:
    """독립된 ROI 이미지에 Canny 기반 전처리 적용.

    1. Grayscale 변환
    2. 독립적 CLAHE (좌/우 조명 차이에 맞춰 적응)
    3. GaussianBlur (노이즈 완화)
    4. Canny 엣지 검출 (50, 150)
    """
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


# ── lane detect.py 파이프라인 함수들 ─────────────────────────────────────────

def make_roi_mask(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """단일 사다리꼴 ROI 마스크 사전 계산 (lane detect.py 파이프라인용).

    ROI 좌표 비율:
    - 상단: y=0.65h, x=0.44w ~ 0.56w
    - 하단: y=0.88h, x=0.18w ~ 0.82w
    """
    pts = np.array([[
        (int(w * 0.44), int(h * 0.65)),
        (int(w * 0.56), int(h * 0.65)),
        (int(w * 0.82), int(h * 0.88)),
        (int(w * 0.18), int(h * 0.88)),
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask, pts


def extract_fits(
    lines: Optional[np.ndarray],
    h: int,
    w: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """허프 선분에서 좌/우 차선 1차 피팅 계수 추출 (lane detect.py 파이프라인용).

    - 기울기 조건: 0.45 <= |slope| <= 3.0
    - 좌측: slope < 0, 우측 경계 < cx + w*0.05
    - 우측: slope > 0, 좌측 경계 > cx - w*0.05
    - 반환값: polyfit(y, x, 1) → [m, c] (x = m*y + c)
    """
    if lines is None:
        return None, None
    cx = w * 0.5
    lx, ly, rx, ry = [], [], [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.45 or abs(slope) > 3.0:
            continue
        if slope < 0 and max(x1, x2) < cx + w * 0.05:
            lx += [x1, x2]; ly += [y1, y2]
        elif slope > 0 and min(x1, x2) > cx - w * 0.05:
            rx += [x1, x2]; ry += [y1, y2]
    lfit = np.polyfit(ly, lx, 1) if len(lx) >= 4 else None
    rfit = np.polyfit(ry, rx, 1) if len(rx) >= 4 else None
    return lfit, rfit


def scale_fit(
    fit: Optional[np.ndarray],
    scale: float
) -> Optional[np.ndarray]:
    """polyfit 계수를 다른 해상도로 변환 (intercept만 스케일 적용).

    다운스케일(PROCESS_SCALE) 해상도에서 피팅된 계수를
    원본 해상도로 복원할 때 사용.
    - slope(m): 비율이므로 변환 불필요
    - intercept(c): c_orig = c_scaled / scale
    """
    if fit is None:
        return None
    return np.array([fit[0], fit[1] / scale], dtype=np.float32)


def detect_crosswalk(
    frame: np.ndarray,
    top_ratio: float = 0.65,
    bot_ratio: float = 0.88
) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
    """전방 주행 노면 영역에서 횡단보도(Zebra Stripe Pattern) 검출."""
    h, w = frame.shape[:2]
    y1 = int(h * top_ratio)
    y2 = int(h * bot_ratio)
    x1 = int(w * 0.22)
    x2 = int(w * 0.78)

    road = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, thresh = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripes = 0
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > 12000:
            continue
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        aspect = rw / float(max(1, rh))
        if 1.5 <= aspect <= 15.0 and rw > 25:
            stripes += 1
            boxes.append((x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh))

    is_crosswalk = stripes >= 3
    return is_crosswalk, boxes


def scale_lines_to_original(
    lines: Optional[np.ndarray],
    transform_params: Tuple[int, int, float, float]
) -> Optional[np.ndarray]:
    """ROI 로컬 좌표계에서 검출된 선분을 원본 프레임 좌표계로 변환."""
    if lines is None:
        return None

    xmin, ymin, scale_x, scale_y = transform_params
    orig_lines = []

    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        ox1 = int(round(x1 * scale_x + xmin))
        oy1 = int(round(y1 * scale_y + ymin))
        ox2 = int(round(x2 * scale_x + xmin))
        oy2 = int(round(y2 * scale_y + ymin))
        orig_lines.append([[ox1, oy1, ox2, oy2]])

    return np.array(orig_lines)


def fit_lane_line(
    lines: Optional[np.ndarray],
    is_left: bool = True
) -> Optional[List[float]]:
    """해당 ROI에서 검출된 선분들을 바탕으로 기울기 조건을 만족하는 점들을 모아 1차 회귀선(x = m*y + c) 산출."""
    if lines is None or len(lines) == 0:
        return None

    pts_x = []
    pts_y = []

    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # 횡단보도 가로선(|기울기| < 0.40) 및 수직 노이즈(|기울기| > 3.0) 필터링
        if abs(slope) < 0.40 or abs(slope) > 3.0:
            continue

        # 좌측 차선: 음의 기울기
        if is_left and slope < -0.35:
            pts_x.extend([x1, x2])
            pts_y.extend([y1, y2])

        # 우측 차선: 양의 기울기
        elif not is_left and slope > 0.35:
            pts_x.extend([x1, x2])
            pts_y.extend([y1, y2])

    if len(pts_x) >= 4:
        try:
            fit = np.polyfit(pts_y, pts_x, 1)
            return list(fit)
        except Exception:
            return None

    return None


class LaneDetector:
    """Detects left and right lane candidates using the cv1021.py dual-ROI pipeline."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        lane_cfg = self.cfg.get("lane", {})
        roi_cfg = self.cfg.get("roi", {})

        # cv1021.py 파라미터 기본값
        self.top_ratio = float(roi_cfg.get("top_ratio", 0.67))
        self.bot_ratio = float(roi_cfg.get("bottom_ratio", 0.86))
        self.bbox_scale_x = float(roi_cfg.get("bbox_scale_x", 0.72))

        self.hough_threshold = int(lane_cfg.get("hough_threshold", 25))
        self.min_line_length = int(lane_cfg.get("min_line_length", 20))
        self.max_line_gap = int(lane_cfg.get("max_line_gap", 60))
        self.min_slope = float(lane_cfg.get("min_slope", 0.40))
        self.max_slope = float(lane_cfg.get("max_slope", 3.0))

        self.target_size = tuple(lane_cfg.get("roi_target_size", (320, 360)))

        # lane detect.py 파이프라인 파라미터
        self.process_scale = float(lane_cfg.get("process_scale", 0.4))
        self._roi_mask_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._clahe_simple = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _get_roi_mask(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """처리 해상도(sh, sw)에 맞는 ROI 마스크를 캐싱하여 반환."""
        key = (h, w)
        if key not in self._roi_mask_cache:
            self._roi_mask_cache[key] = make_roi_mask(h, w)
        return self._roi_mask_cache[key]

    def detect_crosswalk(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
        """주행 영역 내 횡단보도 패턴 감지."""
        return detect_crosswalk(frame, top_ratio=0.65, bot_ratio=0.88)

    def get_dual_roi_geometry(self, h: int, w: int):
        """좌/우 분리 듀얼 ROI 및 Bounding Box 산출."""
        return get_dual_roi_geometry(
            h, w,
            top_ratio=self.top_ratio,
            bot_ratio=self.bot_ratio,
            bbox_scale_x=self.bbox_scale_x
        )

    def crop_and_resize_roi(self, frame, roi_corners, bbox, target_size=None):
        return crop_and_resize_roi(frame, roi_corners, bbox, target_size or self.target_size)

    def preprocess_roi(self, roi_img: np.ndarray) -> np.ndarray:
        return preprocess_roi(roi_img)

    def scale_lines_to_original(self, lines, transform_params):
        return scale_lines_to_original(lines, transform_params)

    def fit_lane_line(self, lines, is_left: bool = True):
        return fit_lane_line(lines, is_left=is_left)

    def detect(
        self,
        frame: np.ndarray,
        is_crosswalk: Optional[bool] = None
    ) -> Tuple[Optional[List[float]], Optional[List[float]], bool, Dict[str, Any]]:
        """cv1021.py 프로세스 기반 전체 차선 검출 파이프라인 실행.

        Returns:
            curr_left_fit: 좌측 차선 [m, c] 또는 None
            curr_right_fit: 우측 차선 [m, c] 또는 None
            is_crosswalk: 횡단보도 감지 여부
            debug_info: ROI 좌표, BBox, 에지 맵 등 시각화/디버깅 정보 딕셔너리
        """
        h, w = frame.shape[:2]

        # [STEP 1] 횡단보도 실시간 감지 (인자로 전달되지 않은 경우 자체 검출)
        cw_boxes = []
        if is_crosswalk is None:
            is_crosswalk, cw_boxes = self.detect_crosswalk(frame)

        # [STEP 2] 좌/우 분리 듀얼 ROI 및 축소된 Bounding Box 산출
        (left_corners, left_bbox), (right_corners, right_bbox) = self.get_dual_roi_geometry(h, w)

        curr_left_fit = None
        curr_right_fit = None
        left_lines_orig = None
        right_lines_orig = None

        # [STEP 3] 횡단보도 구간이 아닐 때만 차선 슬라이싱 및 검출 수행
        if not is_crosswalk:
            # 좌측 ROI: 슬라이싱 + Canny 전처리 + 허프 변환 (maxLineGap=60)
            left_roi_img, left_params = self.crop_and_resize_roi(
                frame, left_corners, left_bbox, self.target_size
            )
            left_edges = self.preprocess_roi(left_roi_img)
            left_lines_roi = cv2.HoughLinesP(
                left_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=self.hough_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            left_lines_orig = self.scale_lines_to_original(left_lines_roi, left_params)
            curr_left_fit = self.fit_lane_line(left_lines_orig, is_left=True)

            # 우측 ROI: 슬라이싱 + Canny 전처리 + 허프 변환 (maxLineGap=60)
            right_roi_img, right_params = self.crop_and_resize_roi(
                frame, right_corners, right_bbox, self.target_size
            )
            right_edges = self.preprocess_roi(right_roi_img)
            right_lines_roi = cv2.HoughLinesP(
                right_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=self.hough_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            right_lines_orig = self.scale_lines_to_original(right_lines_roi, right_params)
            curr_right_fit = self.fit_lane_line(right_lines_orig, is_left=False)
        else:
            # 횡단보도 구간에서는 가로선 오검출을 방지하기 위해 신규 측정을 배제하고 칼만 예측값 유지
            tw, th = self.target_size
            left_edges = np.zeros((th, tw), dtype=np.uint8)
            right_edges = np.zeros((th, tw), dtype=np.uint8)

        # 디버그용 병합 에지 맵 (320x360 2개 결합 -> 640x360)
        combined_edges = np.hstack([left_edges, right_edges])

        # 신뢰도 계산
        if curr_left_fit is not None and curr_right_fit is not None:
            confidence = 0.90
        elif curr_left_fit is not None or curr_right_fit is not None:
            confidence = 0.60
        else:
            confidence = 0.0

        debug_info = {
            "left_corners": left_corners,
            "right_corners": right_corners,
            "left_bbox": left_bbox,
            "right_bbox": right_bbox,
            "left_edges": left_edges,
            "right_edges": right_edges,
            "combined_edges": combined_edges,
            "left_lines": left_lines_orig,
            "right_lines": right_lines_orig,
            "cw_boxes": cw_boxes,
            "confidence": confidence,
            "is_crosswalk": is_crosswalk
        }

        return curr_left_fit, curr_right_fit, is_crosswalk, debug_info

    def detect_scaled(
        self,
        frame: np.ndarray,
        process_scale: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """lane detect.py 파이프라인 기반 경량 차선 검출.

        기존 detect()의 듀얼 ROI / 원본 해상도 처리와 달리,
        단일 사다리꼴 ROI + PROCESS_SCALE 다운스케일 + π/90 허프 변환으로
        실시간 성능을 최적화한 파이프라인.

        처리 과정:
            1. INTER_NEAREST 로 PROCESS_SCALE 배율 다운스케일
            2. 그레이스케일 변환
            3. CLAHE (clipLimit=2.0) + 3×3 GaussianBlur + Canny (60, 150)
            4. 사전 계산된 단일 ROI 마스크 적용
            5. HoughLinesP (θ=π/90, threshold=25, minLen=18, maxGap=22)
            6. extract_fits() → 기울기 기반 좌/우 분리 피팅
            7. scale_fit() → 원본 해상도 좌표로 복원

        Args:
            frame: 원본 BGR 프레임 (임의 해상도 가능)
            process_scale: 다운스케일 배율 (None이면 self.process_scale 사용)

        Returns:
            left_fit:  좌측 차선 [m, c] 원본 해상도 기준, 없으면 None
            right_fit: 우측 차선 [m, c] 원본 해상도 기준, 없으면 None
            roi_edges: 처리 해상도의 ROI 적용 에지 맵 (디버그용)
        """
        scale = process_scale if process_scale is not None else self.process_scale
        orig_h, orig_w = frame.shape[:2]
        sw = int(orig_w * scale)
        sh = int(orig_h * scale)

        # [STEP 1] 다운스케일 (INTER_NEAREST → 빠른 축소)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_NEAREST)

        # [STEP 2] 그레이스케일
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # [STEP 3] CLAHE (clipLimit=2.0) + 3×3 Blur + Canny (60, 150)
        enhanced = self._clahe_simple.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        edges = cv2.Canny(blurred, 60, 150)

        # [STEP 4] 사전 계산된 단일 ROI 마스크 적용
        roi_mask, _ = self._get_roi_mask(sh, sw)
        roi_edges = cv2.bitwise_and(edges, roi_mask)

        # [STEP 5] 허프 변환 (θ=π/90 → 탐색 각도 절반으로 속도 향상)
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 90,
            threshold=25,
            minLineLength=18,
            maxLineGap=22
        )

        # [STEP 6] 기울기 기반 좌/우 분리 피팅
        cur_l, cur_r = extract_fits(lines, sh, sw)

        # [STEP 7] 원본 해상도로 intercept 복원
        left_fit = scale_fit(cur_l, scale)
        right_fit = scale_fit(cur_r, scale)

        return left_fit, right_fit, roi_edges

    # ── 기존 단위 테스트 호환용 메서드 ──────────────────────────────────────────
    def detect_lines(self, edges: np.ndarray, roi_corners: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """단일 에지 맵에 ROI 마스크를 적용하고 HoughLinesP 실행 (레거시/테스트 호환)."""
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_corners, 255)
        roi_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        return lines, roi_edges

    def filter_and_fit(
        self,
        lines: Optional[np.ndarray],
        w: int,
        h: int
    ) -> Tuple[Optional[List[float]], Optional[List[float]], float]:
        """선분 기울기 및 좌우 위치 기반 1차 다항식 피팅 (레거시/테스트 호환)."""
        if lines is None or len(lines) == 0:
            return None, None, 0.0

        left_x, left_y = [], []
        right_x, right_y = [], []
        center_x = w * 0.5
        margin_x = w * 0.08

        total_left_len = 0.0
        total_right_len = 0.0

        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            slope = compute_slope(x1, y1, x2, y2)
            if slope is None:
                continue

            abs_slope = abs(slope)
            if abs_slope < self.min_slope or abs_slope > self.max_slope:
                continue

            length = float(np.hypot(x2 - x1, y2 - y1))

            if slope < 0 and max(x1, x2) < center_x + margin_x:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
                total_left_len += length
            elif slope > 0 and min(x1, x2) > center_x - margin_x:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])
                total_right_len += length

        left_fit = None
        right_fit = None

        if len(left_x) >= 4:
            try:
                left_fit = list(np.polyfit(left_y, left_x, 1))
            except Exception:
                left_fit = None

        if len(right_x) >= 4:
            try:
                right_fit = list(np.polyfit(right_y, right_x, 1))
            except Exception:
                right_fit = None

        conf = 0.0
        if left_fit is not None and right_fit is not None:
            conf = min(1.0, 0.5 + (total_left_len + total_right_len) / 400.0)
        elif left_fit is not None or right_fit is not None:
            one_len = total_left_len if left_fit is not None else total_right_len
            conf = min(0.65, 0.25 + one_len / 400.0)

        return left_fit, right_fit, float(conf)
