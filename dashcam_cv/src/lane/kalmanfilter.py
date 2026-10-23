import cv2
import numpy as np
import os

class SingleLaneKalmanFilter:
    """
    개별 차선(1차 직선 x = m*y + c) 추적을 위한 4차원 칼만 필터 (Kalman Filter):
    - 상태 벡터 x: [m, c, dm, dc]^T (기울기, 절편, 기울기 변화율, 절편 변화율)
    - 관측 벡터 z: [m, c]^T
    - 등속도(Constant Velocity) 물리 모델을 적용하여 프레임 간 떨림(Jitter)을 완벽히 제거하고,
      보행자나 전방 차량 가림 시에도 직전 속도 벡터를 기반으로 차선 위치를 정밀하게 예측(Predict)
    """
    def __init__(self, dt=1.0, max_lost_frames=30):
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
        self.current_fit = None

    def reset(self):
        self.is_initialized = False
        self.lost_frames = 0
        self.current_fit = None
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

    def update(self, measurement_fit):
        """
        측정값(measurement_fit: [m, c])을 바탕으로 칼만 필터 예측(Predict) 및 갱신(Correct) 수행
        """
        # 1. 등속도 모델 기반 다음 상태 예측
        pred = self.kf.predict()
        pred_m = float(pred[0, 0])
        pred_c = float(pred[1, 0])

        if measurement_fit is not None:
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
            # 측정 결측 시 (가림/교차로 등): 예측 상태 유지
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
    """
    좌/우 차선 각각에 독립된 칼만 필터(Kalman Filter)를 탑재한 시계열 차선 추적기
    """
    def __init__(self, dt=1.0, max_lost_frames=30):
        self.left_kf = SingleLaneKalmanFilter(dt=dt, max_lost_frames=max_lost_frames)
        self.right_kf = SingleLaneKalmanFilter(dt=dt, max_lost_frames=max_lost_frames)
        self.max_lost_frames = max_lost_frames
        self.lost_frames = 0

    def update(self, current_left_fit, current_right_fit):
        left_fit, left_lost = self.left_kf.update(current_left_fit)
        right_fit, right_lost = self.right_kf.update(current_right_fit)

        # 둘 다 미검출된 경우 누적 프레임 카운트
        if current_left_fit is None and current_right_fit is None:
            self.lost_frames += 1
        else:
            self.lost_frames = 0

        return left_fit, right_fit, self.lost_frames


def shrink_bounding_box(bbox, scale_x=0.72, scale_y=0.96, w_limit=None, h_limit=None):
    """
    바운딩 박스가 너무 넓게 잡히는 것을 방지하기 위해 중심(Center) 기준으로
    가로(width) 및 세로(height) 폭을 축소(Shrink)하는 전용 함수
    """
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


def get_dual_roi_geometry(h, w, top_ratio=0.67, bot_ratio=0.86, bbox_scale_x=0.72):
    """
    축소된 슬림형 듀얼 ROI 및 컴팩트 Bounding Box 산출:
    1. 좌측 차선 ROI (Left Lane):
       - 상단: y=0.67h, x=0.40w ~ 0.46w (폭 0.06w)
       - 하단: y=0.86h, x=0.28w ~ 0.39w (폭 0.11w)
    2. 우측 차선 ROI (Right Lane):
       - 상단: y=0.67h, x=0.54w ~ 0.60w (폭 0.06w)
       - 하단: y=0.86h, x=0.61w ~ 0.72w (폭 0.11w)
    3. shrink_bounding_box를 통해 BBox 가로 폭을 28% 추가 압축
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


def crop_and_resize_roi(frame, roi_corners, bbox, target_size=(320, 360)):
    """
    축소된 Bounding Box를 기준으로 슬라이싱(Crop) 후 320x360 축소
    """
    h_img, w_img = frame.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    # 경계 안전 클리핑
    xmin = max(0, min(xmin, w_img - 1))
    xmax = max(xmin + 1, min(xmax, w_img))
    ymin = max(0, min(ymin, h_img - 1))
    ymax = max(ymin + 1, min(ymax, h_img))

    crop_w = xmax - xmin
    crop_h = ymax - ymin

    # 축소된 Bounding Box 슬라이싱 (Crop)
    cropped = frame[ymin:ymax, xmin:xmax]

    # Crop 내부 기준의 상대 사다리꼴 다각형 좌표 (BBox 내부로 안전 클리핑)
    rel_corners = roi_corners.copy()
    rel_corners[0, :, 0] = np.clip(rel_corners[0, :, 0] - xmin, 0, crop_w - 1)
    rel_corners[0, :, 1] = np.clip(rel_corners[0, :, 1] - ymin, 0, crop_h - 1)

    # 사다리꼴 마스크 생성 및 합성
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(mask, rel_corners, 255)
    masked_crop = cv2.bitwise_and(cropped, cropped, mask=mask)

    # 지정 해상도로 축소 (연산 고속화)
    target_w, target_h = target_size
    resized_roi = cv2.resize(masked_crop, target_size, interpolation=cv2.INTER_AREA)

    # 원본 좌표계 복원용 파라미터
    scale_x = crop_w / float(target_w)
    scale_y = crop_h / float(target_h)

    return resized_roi, (xmin, ymin, scale_x, scale_y)


def preprocess_roi(roi_img):
    """
    독립된 ROI 이미지에 Sobel-X 기반 전처리 적용:
    1. Grayscale 변환
    2. 독립적 CLAHE (좌/우 조명 차이에 맞춤 적응)
    3. GaussianBlur (노이즈 완화)
    4. Sobel-X 연산: 수직/대각선 차선 성분을 극대화하고 가로선(횡단보도/정지선) 억제
    5. Sobel-X 기반 Canny 엣지 검출
    """
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # 좌/우 조도 불균형 완화를 위한 CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 고주파 노이즈 제거
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # [핵심] Sobel-X 연산 적용 (가로 방향 그래디언트 추출 -> 수직 차선 성분 극대화)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    max_val = np.max(abs_sobelx)
    scaled_sobel = np.uint8(255.0 * abs_sobelx / (max_val + 1e-5))

    # Sobel-X 맵 기반 엣지 검출
    edges = cv2.Canny(scaled_sobel, 40, 130)
    return edges


def detect_crosswalk(frame, top_ratio=0.65, bot_ratio=0.88):
    """
    전방 주행 노면 영역에서 횡단보도(Zebra Stripe Pattern) 검출:
    - 노면의 밝은 가로 직사각형 줄무늬 군집 탐색
    - 가로 줄무늬가 3개 이상 검출되면 횡단보도(Crosswalk)로 판정
    """
    h, w = frame.shape[:2]
    y1 = int(h * top_ratio)
    y2 = int(h * bot_ratio)
    x1 = int(w * 0.22)
    x2 = int(w * 0.78)

    road = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 밝은 노면 표식 추출
    _, thresh = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)

    # 가로 방향 직사각형 커널로 지브라 스트라이프 강조
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


def scale_lines_to_original(lines, transform_params):
    """
    ROI 로컬 좌표계에서 검출된 선분을 원본 프레임 좌표계로 변환
    """
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


def fit_lane_line(lines, is_left=True):
    """
    해당 ROI(좌측 또는 우측)에서 검출된 선분들을 바탕으로
    기울기 조건을 만족하는 점들을 모아 1차 회귀선(x = m*y + c) 계수 산출
    """
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
            return fit
        except Exception:
            return None

    return None


def draw_dual_lane_overlay(frame, left_fit, right_fit, left_corners, right_corners,
                           left_bbox=None, right_bbox=None, show_bbox=True,
                           is_crosswalk=False, lost_frames=0, top_ratio=0.67, bot_ratio=0.86):
    """
    듀얼 ROI 영역 가이드라인, 축소된 Bounding Box, 검출 차선, 주행 가능 영역을 원본 영상 위에 오버레이:
    - [핵심 기능]: 횡단보도 감지 시(is_crosswalk=True) ROI 가이드라인 및 Bounding Box 표시를 즉시 숨김
    - 횡단보도를 통과하면(is_crosswalk=False) ROI 및 가이드라인이 자동으로 다시 표시됨
    """
    h, w = frame.shape[:2]
    min_y = int(h * top_ratio)
    max_y = int(h * bot_ratio)

    overlay = frame.copy()

    # 횡단보도 구간이 아닐 때만 ROI 가이드라인 및 BBox 사각형 표시
    if not is_crosswalk:
        # 1. 축소된 Bounding Box 사각형 표시
        if show_bbox:
            if left_bbox is not None:
                lx1, ly1, lx2, ly2 = left_bbox
                cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (0, 255, 255), 1)
                cv2.putText(overlay, f"L-BBox ({lx2-lx1}px)", (lx1, ly1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if right_bbox is not None:
                rx1, ry1, rx2, ry2 = right_bbox
                cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)
                cv2.putText(overlay, f"R-BBox ({rx2-rx1}px)", (rx1, ry1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 2. 좌/우 듀얼 ROI 사다리꼴 가이드라인 표시
        cv2.polylines(overlay, [left_corners], isClosed=True, color=(255, 200, 0), thickness=1)
        cv2.polylines(overlay, [right_corners], isClosed=True, color=(0, 200, 255), thickness=1)

    # 3. 주행 차로(반투명 폴리곤) 생성 (차선이 유효할 때)
    if left_fit is not None and right_fit is not None:
        l_x1 = int(left_fit[0] * max_y + left_fit[1])
        l_x2 = int(left_fit[0] * min_y + left_fit[1])
        r_x1 = int(right_fit[0] * max_y + right_fit[1])
        r_x2 = int(right_fit[0] * min_y + right_fit[1])

        if l_x1 < r_x1 and l_x2 < r_x2:
            pts = np.array([[(l_x1, max_y), (l_x2, min_y), (r_x2, min_y), (r_x1, max_y)]], dtype=np.int32)
            cv2.fillPoly(overlay, pts, (0, 210, 50))  # 반투명 안전 주행 영역

            # 좌측 차선 (하늘색) & 우측 차선 (주황색)
            cv2.line(overlay, (l_x1, max_y), (l_x2, min_y), (255, 120, 0), 4)
            cv2.line(overlay, (r_x1, max_y), (r_x2, min_y), (0, 120, 255), 4)

    elif left_fit is not None:
        l_x1 = int(left_fit[0] * max_y + left_fit[1])
        l_x2 = int(left_fit[0] * min_y + left_fit[1])
        cv2.line(overlay, (l_x1, max_y), (l_x2, min_y), (255, 120, 0), 4)

    elif right_fit is not None:
        r_x1 = int(right_fit[0] * max_y + right_fit[1])
        r_x2 = int(right_fit[0] * min_y + right_fit[1])
        cv2.line(overlay, (r_x1, max_y), (r_x2, min_y), (0, 120, 255), 4)

    # 합성
    result = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # 4. 상태 HUD 표시
    has_left = left_fit is not None
    has_right = right_fit is not None

    if is_crosswalk:
        status = "CROSSWALK DETECTED (ROI PAUSED)"
        status_color = (0, 140, 255)  # 횡단보도 경고 주황색
    elif has_left and has_right:
        status = "KALMAN TRACKED (L+R)" if lost_frames == 0 else f"KALMAN PREDICTING (Lost {lost_frames}f)"
        status_color = (0, 255, 0)
    elif has_left:
        status = "LEFT ONLY (Kalman)" if lost_frames == 0 else f"PREDICTING (Left, Lost {lost_frames}f)"
        status_color = (255, 200, 0)
    elif has_right:
        status = "RIGHT ONLY (Kalman)" if lost_frames == 0 else f"PREDICTING (Right, Lost {lost_frames}f)"
        status_color = (0, 200, 255)
    else:
        status = "NO LANE"
        status_color = (0, 0, 255)

    cv2.putText(result, f"Kalman Lane: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(result, "Q: Quit | B: Toggle BBox | D: Toggle Dual Minimap | P: Pause",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    return result


def main():
    video_candidates = [
        "images/Test_Video_2.mp4",
        "c:/vision_work/images/Test_Video_2.mp4",
        "Test_Video_2.mp4",
        "dashcam_cv/data/Test_Video_2.mp4"
    ]

    video_path = None
    for path in video_candidates:
        if os.path.exists(path):
            video_path = path
            break

    if video_path is None:
        print("오류: Test_Video_2.mp4 파일을 찾을 수 없습니다.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일({video_path})을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if fps <= 0 else fps
    delay = int(1000 / fps)

    tracker = LaneTracker(dt=1.0, max_lost_frames=30)
    show_debug = False
    show_bbox = True
    paused = False

    print("=== Sobel-X + maxLineGap 60 + 횡단보도 동적 ROI 제어 차선 추적 파이프라인 ===")
    print("- 'q': 종료")
    print("- 'b': 축소된 Bounding Box 표시 토글")
    print("- 'd': 좌/우 독립 엣지 미니맵 디버그 뷰 토글")
    print("- 'p': 일시정지 / 재생")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("비디오 재생이 완료되었습니다.")
                break

            h, w = frame.shape[:2]

            # [STEP 1] 횡단보도(Zebra Stripe) 실시간 감지
            is_crosswalk, cw_boxes = detect_crosswalk(frame)

            # [STEP 2] 좌/우 분리 듀얼 ROI 및 축소된 Bounding Box 산출
            (left_corners, left_bbox), (right_corners, right_bbox) = get_dual_roi_geometry(
                h, w, top_ratio=0.67, bot_ratio=0.86, bbox_scale_x=0.72
            )

            # [STEP 3] 횡단보도 구간이 아닐 때만 차선 슬라이싱 및 검출 수행
            if not is_crosswalk:
                # 좌측 ROI: 슬라이싱 + Sobel-X 전처리 + 허프 변환 (maxLineGap=60)
                left_roi_img, left_params = crop_and_resize_roi(
                    frame, left_corners, left_bbox, target_size=(320, 360)
                )
                left_edges = preprocess_roi(left_roi_img)
                left_lines_roi = cv2.HoughLinesP(
                    left_edges, rho=1, theta=np.pi / 180, threshold=25, minLineLength=20, maxLineGap=60
                )
                left_lines_orig = scale_lines_to_original(left_lines_roi, left_params)
                curr_left_fit = fit_lane_line(left_lines_orig, is_left=True)

                # 우측 ROI: 슬라이싱 + Sobel-X 전처리 + 허프 변환 (maxLineGap=60)
                right_roi_img, right_params = crop_and_resize_roi(
                    frame, right_corners, right_bbox, target_size=(320, 360)
                )
                right_edges = preprocess_roi(right_roi_img)
                right_lines_roi = cv2.HoughLinesP(
                    right_edges, rho=1, theta=np.pi / 180, threshold=25, minLineLength=20, maxLineGap=60
                )
                right_lines_orig = scale_lines_to_original(right_lines_roi, right_params)
                curr_right_fit = fit_lane_line(right_lines_orig, is_left=False)
            else:
                # 횡단보도 구간에서는 가로선 오검출을 방지하기 위해 신규 측정을 배제하고 칼만 예측값 유지
                curr_left_fit, curr_right_fit = None, None
                left_edges = np.zeros((360, 320), dtype=np.uint8)
                right_edges = np.zeros((360, 320), dtype=np.uint8)

            # [STEP 4] 칼만 필터 시계열 트래커 업데이트
            smooth_left, smooth_right, lost_cnt = tracker.update(curr_left_fit, curr_right_fit)

            # [STEP 5] 원본 영상 오버레이 시각화 (횡단보도 감지 시 ROI 표시 숨김)
            result_frame = draw_dual_lane_overlay(
                frame, smooth_left, smooth_right, left_corners, right_corners,
                left_bbox=left_bbox, right_bbox=right_bbox, show_bbox=show_bbox,
                is_crosswalk=is_crosswalk, lost_frames=lost_cnt
            )

            # [STEP 6] 디버그 미니맵 (우측 상단: 좌/우 독립 엣지 맵 병합 표시)
            if show_debug:
                combined_edges = np.hstack([left_edges, right_edges])
                combined_bgr = cv2.cvtColor(combined_edges, cv2.COLOR_GRAY2BGR)

                cv2.line(combined_bgr, (320, 0), (320, 360), (0, 255, 255), 2)
                cv2.putText(combined_bgr, "LEFT (Sobel-X)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                cv2.putText(combined_bgr, "RIGHT (Sobel-X)", (340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                mini_w, mini_h = 320, 180
                mini_display = cv2.resize(combined_bgr, (mini_w, mini_h))

                cv2.rectangle(result_frame, (w - mini_w - 12, 8), (w - 8, mini_h + 12), (0, 255, 255), 2)
                result_frame[10:10+mini_h, w-mini_w-10:w-10] = mini_display

            cv2.imshow("Lane Detection & Crosswalk Adaptive ROI", result_frame)

        key = cv2.waitKey(1 if paused else delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            show_bbox = not show_bbox
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
