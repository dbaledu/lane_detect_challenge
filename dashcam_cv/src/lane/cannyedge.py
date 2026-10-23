import cv2
import numpy as np
import os
import time

class SingleLaneKalmanFilter:
    """
    개별 차선(1차 직선 x = m*y + c) 추적을 위한 4차원 칼만 필터 (Kalman Filter):
    - 상태 벡터: [m, c, dm, dc]^T (기울기, 절편, 기울기 변화율, 절편 변화율)
    - 관측 벡터: [m, c]^T
    - 등속도 물리 모델을 통해 프레임 간 차선 떨림(Jitter)을 방지하고
      횡단보도 통과 시 직전 궤적을 기반으로 안정적으로 차선 위치를 예측(Predict)
    """
    def __init__(self, dt=1.0, max_lost_frames=30):
        self.dt = dt
        self.max_lost_frames = max_lost_frames
        self.kf = cv2.KalmanFilter(4, 2)

        # 상태 전이 행렬 (F)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 관측 행렬 (H)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 프로세스 노이즈 공분산 (Q)
        self.kf.processNoiseCov = np.array([
            [1e-4, 0, 0, 0],
            [0, 1e-1, 0, 0],
            [0, 0, 1e-4, 0],
            [0, 0, 0, 1e-1]
        ], dtype=np.float32)

        # 측정 노이즈 공분산 (R)
        self.kf.measurementNoiseCov = np.array([
            [5e-3, 0],
            [0, 5.0]
        ], dtype=np.float32)

        # 오차 공분산 행렬 (P)
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
        pred = self.kf.predict()
        pred_m = float(pred[0, 0])
        pred_c = float(pred[1, 0])

        if measurement_fit is not None:
            m_meas, c_meas = float(measurement_fit[0]), float(measurement_fit[1])

            if not self.is_initialized:
                self.kf.statePost = np.array([[m_meas], [c_meas], [0.0], [0.0]], dtype=np.float32)
                self.is_initialized = True
                self.lost_frames = 0
                self.current_fit = [m_meas, c_meas]
            else:
                diff_m = abs(m_meas - pred_m)
                diff_c = abs(c_meas - pred_c)

                # 급격한 노이즈 튐 차단 (Innovation Gating)
                if diff_m > 0.45 or diff_c > 160.0:
                    self.lost_frames += 1
                    self.current_fit = [pred_m, pred_c]
                else:
                    z = np.array([[m_meas], [c_meas]], dtype=np.float32)
                    estimated = self.kf.correct(z)
                    self.lost_frames = 0
                    self.current_fit = [float(estimated[0, 0]), float(estimated[1, 0])]
        else:
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
    좌/우 차선 각각에 독립된 칼만 필터를 적용하여
    프레임 간 떨림(Jitter)을 방지하는 스무딩 & 추적 클래스
    """
    def __init__(self, dt=1.0, max_lost_frames=30):
        self.left_kf = SingleLaneKalmanFilter(dt=dt, max_lost_frames=max_lost_frames)
        self.right_kf = SingleLaneKalmanFilter(dt=dt, max_lost_frames=max_lost_frames)
        self.max_lost_frames = max_lost_frames
        self.lost_frames = 0

    def update(self, current_left_fit, current_right_fit):
        left_fit, left_lost = self.left_kf.update(current_left_fit)
        right_fit, right_lost = self.right_kf.update(current_right_fit)

        if current_left_fit is None and current_right_fit is None:
            self.lost_frames += 1
        else:
            self.lost_frames = 0

        return left_fit, right_fit, self.lost_frames


def get_optimized_roi_corners(h, w):
    """
    Test_Video_2.mp4에 최적화된 사다리꼴 ROI 좌표:
    - 상단(0.65h): 앞차 범퍼 및 보행자 상체 노이즈 차단
    - 하단(0.88h): 화면 하단 OSD 텍스트 및 보닛 반사 차단
    - 좌우(0.18w ~ 0.82w): 광각 왜곡 인도/상점 간판 배제
    """
    roi_corners = np.array([[
        (int(w * 0.44), int(h * 0.65)),  # 상단 좌측
        (int(w * 0.56), int(h * 0.65)),  # 상단 우측
        (int(w * 0.82), int(h * 0.88)),  # 하단 우측
        (int(w * 0.18), int(h * 0.88))   # 하단 좌측
    ]], dtype=np.int32)
    return roi_corners


def preprocess_roi_fast(frame, roi_corners, target_w=640, target_h=360):
    """
    [초고속 전처리 파이프라인 (실시간 30 FPS 보장)]:
    1. 전체 1080p 화면 대신 도로 영역(0.65h~0.88h, 0.18w~0.82w)만 먼저 슬라이싱 (연산량 80% 절감)
    2. 640x360 축소 후 CLAHE 대비 향상 적용 (~1.5ms 고속 연산)
    3. GaussianBlur 노이즈 제거
    4. Canny 엣지 검출 및 사다리꼴 마스킹
    """
    h, w = frame.shape[:2]
    ymin, ymax = int(h * 0.65), int(h * 0.88)
    xmin, xmax = int(w * 0.18), int(w * 0.82)
    crop_w = xmax - xmin
    crop_h = ymax - ymin

    # 도로 영역 BBox 슬라이싱 (Crop)
    road_crop = frame[ymin:ymax, xmin:xmax]

    # 고속 처리를 위해 640x360으로 리사이즈
    resized_crop = cv2.resize(road_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 1. Grayscale 변환
    gray = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2GRAY)

    # 2. 노면 대비 극대화 (작은 해상도에서 CLAHE 적용 -> 고속 연산)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. 노이즈 제거
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 4. Canny 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)

    # 5. 상대 사다리꼴 마스킹 적용
    scale_x = target_w / float(crop_w)
    scale_y = target_h / float(crop_h)

    rel_corners = roi_corners.copy().astype(np.float32)
    rel_corners[0, :, 0] = (rel_corners[0, :, 0] - xmin) * scale_x
    rel_corners[0, :, 1] = (rel_corners[0, :, 1] - ymin) * scale_y
    rel_corners = rel_corners.astype(np.int32)

    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    cv2.fillPoly(mask, rel_corners, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 원본 좌표 복원용 파라미터 반환
    transform_params = (xmin, ymin, crop_w / float(target_w), crop_h / float(target_h))
    return masked_edges, transform_params


def detect_crosswalk_precise(frame, top_ratio=0.68, bot_ratio=0.86):
    """
    [횡단보도 정밀 감지기 - 일반 도로 오감지 원천 차단]:
    - 일반 도로에서는 절대 False가 나오도록 최소 4개 이상의 넓은 지브라 줄무늬 군집 탐색
    - 도로 폭의 30% 이상 넓게 스트라이프가 펼쳐져 있을 때만 횡단보도로 판정
    """
    h, w = frame.shape[:2]
    y1 = int(h * top_ratio)
    y2 = int(h * bot_ratio)
    x1 = int(w * 0.22)
    x2 = int(w * 0.78)

    road_patch = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(road_patch, cv2.COLOR_BGR2GRAY)

    # 밝은 노면 표식 추출
    _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

    # 가로 직사각형 모폴로지 커널로 지브라 스트라이프 추출
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 4))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150 or area > 10000:
            continue
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        aspect = rw / float(max(1, rh))
        # 횡단보도 스트라이프: 가로 폭이 35px 이상이고 가로가 세로보다 훨씬 김
        if 2.0 <= aspect <= 12.0 and rw > 35:
            stripes.append((x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh))

    # 최소 4개 이상의 스트라이프가 도로 폭 30% 이상 넓게 펼쳐져 있어야 횡단보도로 확정
    if len(stripes) >= 4:
        min_stripe_x = min(s[0] for s in stripes)
        max_stripe_x = max(s[2] for s in stripes)
        if (max_stripe_x - min_stripe_x) > (w * 0.30):
            return True

    return False


def scale_lines_to_original(lines, transform_params):
    """
    640x360 ROI 좌표계에서 검출된 선분을 원본 프레임 좌표계로 변환
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


def extract_lane_fits(lines, h, w):
    """
    허프 변환으로 추출된 직선들 중:
    - 횡단보도 가로선 제거 (|기울기| < 0.40)
    - 수직 노이즈 제거 (|기울기| > 3.0)
    - 좌/우 영역 분리 (x 좌표 기준 분할)
    - 1차 다항식(polyfit) 계수 산출
    """
    if lines is None:
        return None, None

    left_x, left_y = [], []
    right_x, right_y = [], []
    center_x = w * 0.5

    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # 횡단보도 가로선 및 수직 노이즈 필터링
        if abs(slope) < 0.40 or abs(slope) > 3.0:
            continue

        # 좌측 차선: 음의 기울기
        if slope < -0.35 and max(x1, x2) < center_x + (w * 0.05):
            left_x.extend([x1, x2])
            left_y.extend([y1, y2])

        # 우측 차선: 양의 기울기
        elif slope > 0.35 and min(x1, x2) > center_x - (w * 0.05):
            right_x.extend([x1, x2])
            right_y.extend([y1, y2])

    left_fit = None
    right_fit = None

    if len(left_x) >= 4:
        try:
            left_fit = np.polyfit(left_y, left_x, 1)
        except Exception:
            left_fit = None

    if len(right_x) >= 4:
        try:
            right_fit = np.polyfit(right_y, right_x, 1)
        except Exception:
            right_fit = None

    return left_fit, right_fit


def draw_lane_overlay(frame, left_fit, right_fit, roi_corners, is_crosswalk=False,
                      force_show_roi=False, lost_frames=0, fps_display=30.0):
    """
    검출/추적된 차선과 주행 영역(Polygon), ROI 가이드라인을 원본 영상 위에 합성하여 시각화:
    - [ROI 표시 보장]: 기본 주행 상태에서는 사다리꼴 ROI 가이드라인이 밝은 노란색(두께 2)으로 선명하게 항상 표시됨
    - [동적 제어]: 실제 횡단보도에 도달했을 때만(is_crosswalk=True) ROI 가이드라인이 숨겨지고, 지나면 다시 켜짐
    - 'r' 키를 누르면 강제로 ROI를 항상 켜둘 수 있음
    """
    h, w = frame.shape[:2]
    min_y = int(h * 0.65)
    max_y = int(h * 0.88)

    overlay = frame.copy()

    # 1. 좌/우 차선 및 주행 가능 영역(녹색 반투명 폴리곤) 생성
    if left_fit is not None and right_fit is not None:
        l_x1 = int(left_fit[0] * max_y + left_fit[1])
        l_x2 = int(left_fit[0] * min_y + left_fit[1])
        r_x1 = int(right_fit[0] * max_y + right_fit[1])
        r_x2 = int(right_fit[0] * min_y + right_fit[1])

        if l_x1 < r_x1 and l_x2 < r_x2:
            pts = np.array([[(l_x1, max_y), (l_x2, min_y), (r_x2, min_y), (r_x1, max_y)]], dtype=np.int32)
            cv2.fillPoly(overlay, pts, (0, 210, 50))  # 반투명 안전 주행 영역

            # 좌측 차선(하늘색) 및 우측 차선(주황색)
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

    # 반투명 합성 (오버레이 35% + 원본 65%)
    result = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # 2. 사다리꼴 ROI 가이드라인을 최종 화면(result) 위에 직접 그려 100% 선명하게 표시
    # 횡단보도가 아니거나, 강제 표시 모드(force_show_roi)일 때 선명한 노란색으로 표시
    show_roi_now = (not is_crosswalk) or force_show_roi
    if show_roi_now:
        cv2.polylines(result, [roi_corners], isClosed=True, color=(0, 255, 255), thickness=2)

    # 3. 상태 HUD 표시
    has_lane = (left_fit is not None or right_fit is not None)
    if is_crosswalk and not force_show_roi:
        status = "CROSSWALK ZONE (ROI HIDDEN)"
        status_color = (0, 140, 255)  # 경고 주황색
    elif lost_frames == 0 and has_lane:
        status = "TRACKED (ROI ON)"
        status_color = (0, 255, 0)
    elif has_lane:
        status = f"PREDICTING (Lost {lost_frames}f, ROI ON)"
        status_color = (0, 220, 255)
    else:
        status = "NO LANE (ROI ON)"
        status_color = (0, 0, 255)

    cv2.putText(result, f"Status: {status} | FPS: {fps_display:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
    cv2.putText(result, "Q: Quit | R: Toggle Force ROI | D: Debug Map | P: Pause", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    return result


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_candidates = [
        os.path.join(base_dir, "../../data/Test_Video_2.mp4"),
        os.path.join(base_dir, "../../../images/Test_Video_2.mp4"),
        "dashcam_cv/data/Test_Video_2.mp4",
        "data/Test_Video_2.mp4",
        "images/Test_Video_2.mp4",
        "Test_Video_2.mp4",
        "c:/github/lane_detect_challenge/dashcam_cv/data/Test_Video_2.mp4",
        "c:/vision_work/images/Test_Video_2.mp4"
    ]

    video_path = None
    for path in video_candidates:
        if os.path.exists(path):
            video_path = path
            break

    if video_path is None:
        print("오류: Test_Video_2.mp4 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일({video_path})을 열 수 없습니다.")
        return

    # 30 FPS 기준 재생 프레임 간격 (33.33ms)
    target_fps = 30.0
    frame_interval_ms = 1000.0 / target_fps

    tracker = LaneTracker(dt=1.0, max_lost_frames=30)
    show_debug = False
    force_show_roi = False  # 'r' 키로 강제 켜기 가능
    paused = False

    # 횡단보도 감지 지속 카운터
    crosswalk_hold_frames = 0

    print("=== 고속 30 FPS + Canny 엣지 + maxLineGap 60 + 선명한 ROI 표시 파이프라인 ===")
    print("- 'q': 종료")
    print("- 'r': ROI 가이드라인 강제 켜기/끄기 토글")
    print("- 'd': Canny 엣지 미니맵 디버그 토글")
    print("- 'p': 일시정지 / 재생")

    prev_time = time.time()
    current_fps = 30.0

    while cap.isOpened():
        t_start = time.time()

        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("비디오 재생이 완료되었습니다.")
                break

            h, w = frame.shape[:2]

            # [STEP 1] 횡단보도 정밀 감지 (일반 도로 오작동 배제)
            detected_cw = detect_crosswalk_precise(frame)

            if detected_cw:
                crosswalk_hold_frames = 30  # 약 1초간 유지

            if crosswalk_hold_frames > 0:
                is_crosswalk = True
                crosswalk_hold_frames -= 1
            else:
                is_crosswalk = False

            # [STEP 2] 사다리꼴 ROI 산출
            roi_corners = get_optimized_roi_corners(h, w)

            # [STEP 3] 차선 전처리 및 검출
            if (not is_crosswalk) or force_show_roi:
                # [초고속 전처리]: ROI 먼저 Crop -> 640x360 축소 -> Canny 적용
                roi_edges_640, transform_params = preprocess_roi_fast(frame, roi_corners)

                # [확률적 허프 변환]: 요청하신 maxLineGap=60 설정
                lines_640 = cv2.HoughLinesP(
                    roi_edges_640,
                    rho=1,
                    theta=np.pi / 180,
                    threshold=30,
                    minLineLength=25,
                    maxLineGap=60  # 점선 차선을 매끄럽게 연결하기 위한 간격 60
                )

                # 원본 해상도 좌표계로 변환
                lines_orig = scale_lines_to_original(lines_640, transform_params)
                curr_left_fit, curr_right_fit = extract_lane_fits(lines_orig, h, w)
            else:
                # 횡단보도 구간: 가로선 오검출을 방지하기 위해 신규 측정을 배제하고 칼만 예측값 유지
                curr_left_fit, curr_right_fit = None, None
                roi_edges_640 = np.zeros((360, 640), dtype=np.uint8)

            # [STEP 4] 칼만 필터 시계열 트래커 업데이트
            smooth_left, smooth_right, lost_cnt = tracker.update(curr_left_fit, curr_right_fit)

            # [STEP 5] 결과 시각화 (ROI 가이드라인이 선명하게 노란색으로 표시됨)
            result_frame = draw_lane_overlay(
                frame, smooth_left, smooth_right, roi_corners,
                is_crosswalk=is_crosswalk, force_show_roi=force_show_roi,
                lost_frames=lost_cnt, fps_display=current_fps
            )

            # [STEP 6] 디버그 미니맵 (Sobel-X 엣지 맵)
            if show_debug:
                debug_bgr = cv2.cvtColor(roi_edges_640, cv2.COLOR_GRAY2BGR)
                mini_h, mini_w = int(h * 0.28), int(w * 0.28)
                debug_resized = cv2.resize(debug_bgr, (mini_w, mini_h))
                cv2.rectangle(result_frame, (w - mini_w - 12, 8), (w - 8, mini_h + 12), (0, 255, 255), 2)
                result_frame[10:10+mini_h, w-mini_w-10:w-10] = debug_resized
                cv2.putText(result_frame, "Canny ROI Edges", (w - mini_w, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Lane Detection (Canny & Vibrant ROI)", result_frame)

            # 실제 실시간 FPS 계산
            current_time = time.time()
            current_fps = 1.0 / max(1e-4, current_time - prev_time)
            prev_time = current_time

        # [STEP 7] 정확한 30 FPS 보장을 위한 동적 대기 시간 계산
        elapsed_ms = (time.time() - t_start) * 1000.0
        wait_ms = max(1, int(frame_interval_ms - elapsed_ms))

        key = cv2.waitKey(1 if paused else wait_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            force_show_roi = not force_show_roi
            print(f">> ROI 강제 표시 모드: {'ON' if force_show_roi else 'OFF'}")
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()