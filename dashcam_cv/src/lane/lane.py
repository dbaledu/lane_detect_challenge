import cv2
import numpy as np
import os

class LaneTracker:
    """
    프레임 간 차선 연속성을 유지하고, 교차로나 보행자/차량 가림 현상 시
    차선 깜빡임(Jitter)을 방지하는 스무딩 & 추적 클래스
    """
    def __init__(self, alpha=0.2, max_lost_frames=30):
        self.alpha = alpha  # 지수이동평균(EMA) 가중치 (0~1)
        self.max_lost_frames = max_lost_frames
        self.left_fit = None
        self.right_fit = None
        self.lost_frames = 0

    def update(self, current_left_fit, current_right_fit):
        # 좌측 차선 업데이트
        if current_left_fit is not None:
            if self.left_fit is None:
                self.left_fit = current_left_fit
            else:
                self.left_fit = self.alpha * np.array(current_left_fit) + (1 - self.alpha) * np.array(self.left_fit)

        # 우측 차선 업데이트
        if current_right_fit is not None:
            if self.right_fit is None:
                self.right_fit = current_right_fit
            else:
                self.right_fit = self.alpha * np.array(current_right_fit) + (1 - self.alpha) * np.array(self.right_fit)

        # 둘 다 검출되지 않은 경우 (보행자 가림 또는 교차로 통과 시)
        if current_left_fit is None and current_right_fit is None:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.left_fit = None
                self.right_fit = None
        else:
            self.lost_frames = 0

        return self.left_fit, self.right_fit, self.lost_frames


def preprocess_image(frame):
    """
    타임스퀘어 야간 조명(전광판 과노출 + 어두운 노면)에 대응하기 위한 전처리:
    1. Grayscale 변환
    2. CLAHE (대비 제한 적응형 히스토그램 평활화)로 노면 차선의 대비를 극대화
    3. GaussianBlur로 고주파 노이즈 제거
    4. Canny Edge 검출
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 조도 불균형 완화를 위한 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 노이즈 제거
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 엣지 검출
    edges = cv2.Canny(blurred, 60, 150)
    return edges


def get_optimized_roi(edges):
    """
    Test_Video_2.mp4에 최적화된 사다리꼴 ROI:
    1. 상단(0.65h): 앞차 범퍼/번호판, 횡단보도 보행자 상체 노이즈 차단
    2. 하단(0.88h): 화면 하단 OSD 텍스트(속도계, 가속도계 센서) 및 보닛 반사 차단
    3. 좌우(0.18w ~ 0.82w): 광각 렌즈 왜곡으로 휘어지는 인도/상점 간판 배제
    """
    h, w = edges.shape[:2]
    roi_corners = np.array([[
        (int(w * 0.44), int(h * 0.65)),  # 상단 좌측
        (int(w * 0.56), int(h * 0.65)),  # 상단 우측
        (int(w * 0.82), int(h * 0.88)),  # 하단 우측
        (int(w * 0.18), int(h * 0.88))   # 하단 좌측
    ]], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_corners, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges, roi_corners


def extract_lane_fits(lines, h, w):
    """
    허프 변환으로 추출된 직선들 중:
    - 횡단보도 가로선 제거 (|기울기| < 0.45)
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
        
        # 1. 횡단보도 지브라 패턴(수평선) 및 가파른 노이즈 필터링
        if abs(slope) < 0.45 or abs(slope) > 3.0:
            continue
            
        # 2. 좌측 차선: 음의 기울기 & 화면 좌측 절반에 위치
        if slope < 0 and max(x1, x2) < center_x + (w * 0.05):
            left_x.extend([x1, x2])
            left_y.extend([y1, y2])
            
        # 3. 우측 차선: 양의 기울기 & 화면 우측 절반에 위치
        elif slope > 0 and min(x1, x2) > center_x - (w * 0.05):
            right_x.extend([x1, x2])
            right_y.extend([y1, y2])

    left_fit = None
    right_fit = None

    if len(left_x) >= 4:
        left_fit = np.polyfit(left_y, left_x, 1)

    if len(right_x) >= 4:
        right_fit = np.polyfit(right_y, right_x, 1)

    return left_fit, right_fit


def draw_lane_overlay(frame, left_fit, right_fit, roi_corners, lost_frames=0):
    """
    검출/추적된 차선과 주행 영역(Polygon)을 원본 영상 위에 합성하여 시각화
    """
    h, w = frame.shape[:2]
    min_y = int(h * 0.65)
    max_y = int(h * 0.88)
    
    overlay = frame.copy()
    
    # 1. ROI 가이드라인 표시
    cv2.polylines(overlay, [roi_corners], isClosed=True, color=(100, 100, 100), thickness=1)

    # 2. 좌/우 차선 모두 유효할 때: 주행 영역 채우기
    if left_fit is not None and right_fit is not None:
        l_x1 = int(left_fit[0] * max_y + left_fit[1])
        l_x2 = int(left_fit[0] * min_y + left_fit[1])
        r_x1 = int(right_fit[0] * max_y + right_fit[1])
        r_x2 = int(right_fit[0] * min_y + right_fit[1])

        if l_x1 < r_x1 and l_x2 < r_x2:
            pts = np.array([[(l_x1, max_y), (l_x2, min_y), (r_x2, min_y), (r_x1, max_y)]], dtype=np.int32)
            cv2.fillPoly(overlay, pts, (0, 220, 0))  # 반투명 주행 영역
            
            cv2.line(overlay, (l_x1, max_y), (l_x2, min_y), (255, 100, 0), 4)
            cv2.line(overlay, (r_x1, max_y), (r_x2, min_y), (0, 100, 255), 4)

    # 3. 한쪽 차선만 유효할 때
    elif left_fit is not None:
        l_x1 = int(left_fit[0] * max_y + left_fit[1])
        l_x2 = int(left_fit[0] * min_y + left_fit[1])
        cv2.line(overlay, (l_x1, max_y), (l_x2, min_y), (255, 100, 0), 4)

    elif right_fit is not None:
        r_x1 = int(right_fit[0] * max_y + right_fit[1])
        r_x2 = int(right_fit[0] * min_y + right_fit[1])
        cv2.line(overlay, (r_x1, max_y), (r_x2, min_y), (0, 100, 255), 4)

    result = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    
    # 상태 HUD 표시
    status = "DETECTED" if lost_frames == 0 and (left_fit is not None or right_fit is not None) else \
             f"TRACKING (Lost {lost_frames}f)" if (left_fit is not None or right_fit is not None) else "NO LANE"
    status_color = (0, 255, 0) if "DETECTED" in status else (0, 200, 255) if "TRACKING" in status else (0, 0, 255)
    
    cv2.putText(result, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(result, "Q: Quit | D: Toggle Debug View | P: Pause", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    return result


def main():
    video_candidates = [
        "images/Test_Video_2.mp4",
        "c:/vision_work/images/Test_Video_2.mp4",
        "Test_Video_2.mp4"
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
    delay = int(1000 / fps) if fps > 0 else 30
    
    tracker = LaneTracker(alpha=0.18, max_lost_frames=25)
    show_debug = False
    paused = False

    print("=== 차선 검출 실행 ===")
    print("- 'q': 종료")
    print("- 'd': ROI 및 엣지 미니맵 디버그 화면 토글")
    print("- 'p': 일시정지 / 재생")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("비디오 재생이 완료되었습니다.")
                break

            h, w = frame.shape[:2]

            # 1. 전처리 (CLAHE + Gaussian Blur + Canny)
            edges = preprocess_image(frame)

            # 2. 최적화된 ROI 마스킹
            roi_edges, roi_corners = get_optimized_roi(edges)

            # 3. 확률적 허프 변환
            lines = cv2.HoughLinesP(
                roi_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=35,
                minLineLength=30,
                maxLineGap=25
            )

            # 4. 차선 직선 필터링 및 1차 회귀 피팅
            curr_left_fit, curr_right_fit = extract_lane_fits(lines, h, w)

            # 5. 시계열 트래커 업데이트
            smooth_left, smooth_right, lost_cnt = tracker.update(curr_left_fit, curr_right_fit)

            # 6. 결과 시각화
            result_frame = draw_lane_overlay(frame, smooth_left, smooth_right, roi_corners, lost_cnt)

            if show_debug:
                # 우측 상단에 ROI 엣지 미니맵 표시
                debug_bgr = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
                mini_h, mini_w = int(h * 0.3), int(w * 0.3)
                debug_resized = cv2.resize(debug_bgr, (mini_w, mini_h))
                cv2.rectangle(result_frame, (w - mini_w - 12, 8), (w - 8, mini_h + 12), (0, 255, 255), 2)
                result_frame[10:10+mini_h, w-mini_w-10:w-10] = debug_resized
                cv2.putText(result_frame, "ROI Edge Map", (w - mini_w, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Test_Video_2 Lane Detection", result_frame)

        key = cv2.waitKey(1 if paused else delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

