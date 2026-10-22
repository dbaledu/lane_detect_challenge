import cv2
import numpy as np
import os
import threading
import queue
import time

# ── 성능 튜닝 상수 ───────────────────────────────────────────────
PROCESS_SCALE  = 0.4    # 검출용 다운스케일 (낮을수록 빠름, 권장 0.35~0.5)
DISPLAY_SCALE  = 0.75   # 출력 창 해상도 비율 (1.0 = 원본)
PLAYBACK_SPEED = 0.31    # 재생 속도 배율 (1.0=원본, 0.5=2배 느림, 0.2=5배 느림)


# ── 스레드 기반 프레임 리더 ──────────────────────────────────────
class FrameReader(threading.Thread):
    """별도 스레드에서 프레임을 미리 읽어 Queue 에 적재. I/O 대기 비용 제거."""
    def __init__(self, cap, maxsize=4):
        super().__init__(daemon=True)
        self.cap   = cap
        self.q     = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.q.put((False, None))
                return
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((True, frame))

    def read(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def stop(self):
        self._stop.set()


# ── EMA 차선 트래커 ──────────────────────────────────────────────
class LaneTracker:
    def __init__(self, alpha=0.2, max_lost_frames=30):
        self.alpha           = alpha
        self.max_lost_frames = max_lost_frames
        self.left_fit        = None
        self.right_fit       = None
        self.lost_frames     = 0

    def update(self, cur_left, cur_right):
        def ema(prev, cur):
            if cur is None:
                return prev
            cur = np.asarray(cur, dtype=np.float32)
            return cur if prev is None else (
                self.alpha * cur + (1.0 - self.alpha) * np.asarray(prev, dtype=np.float32)
            )
        self.left_fit  = ema(self.left_fit,  cur_left)
        self.right_fit = ema(self.right_fit, cur_right)

        if cur_left is None and cur_right is None:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.left_fit = self.right_fit = None
        else:
            self.lost_frames = 0

        return self.left_fit, self.right_fit, self.lost_frames


# ── ROI 마스크 사전 계산 (1회만 호출) ───────────────────────────
def make_roi_mask(h, w):
    pts = np.array([[
        (int(w * 0.44), int(h * 0.65)),
        (int(w * 0.56), int(h * 0.65)),
        (int(w * 0.82), int(h * 0.88)),
        (int(w * 0.18), int(h * 0.88)),
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask, pts


def extract_fits(lines, h, w):
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


def scale_fit(fit, scale):
    """polyfit 계수를 다른 해상도로 변환 (intercept만 스케일)."""
    return None if fit is None else np.array([fit[0], fit[1] / scale], dtype=np.float32)


def draw_overlay(frame, left_fit, right_fit, lost_frames):
    """BGR 컬러 프레임에 차선 오버레이 합성."""
    h, w   = frame.shape[:2]
    y_far  = int(h * 0.65)
    y_near = int(h * 0.88)
    overlay = frame.copy()

    if left_fit is not None and right_fit is not None:
        lx_near = int(np.polyval(left_fit,  y_near))
        lx_far  = int(np.polyval(left_fit,  y_far))
        rx_near = int(np.polyval(right_fit, y_near))
        rx_far  = int(np.polyval(right_fit, y_far))
        if lx_near < rx_near and lx_far < rx_far:
            poly_pts = np.array(
                [[(lx_near, y_near), (lx_far, y_far), (rx_far, y_far), (rx_near, y_near)]],
                dtype=np.int32
            )
            cv2.fillPoly(overlay, poly_pts, (0, 200, 0))
        cv2.line(overlay, (lx_near, y_near), (lx_far, y_far), (255, 120,   0), 3)
        cv2.line(overlay, (rx_near, y_near), (rx_far, y_far), (  0, 120, 255), 3)
    elif left_fit is not None:
        lx_near = int(np.polyval(left_fit, y_near))
        lx_far  = int(np.polyval(left_fit, y_far))
        cv2.line(overlay, (lx_near, y_near), (lx_far, y_far), (255, 120, 0), 3)
    elif right_fit is not None:
        rx_near = int(np.polyval(right_fit, y_near))
        rx_far  = int(np.polyval(right_fit, y_far))
        cv2.line(overlay, (rx_near, y_near), (rx_far, y_far), (0, 120, 255), 3)

    result = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    has_lane = left_fit is not None or right_fit is not None
    if has_lane and lost_frames == 0:
        status, color = "DETECTED",                     (0, 255,   0)
    elif has_lane:
        status, color = f"TRACKING (lost {lost_frames}f)", (0, 200, 255)
    else:
        status, color = "NO LANE",                      (0,   0, 255)

    cv2.putText(result, f"Status: {status}", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(result, "Q:Quit  D:Debug  P:Pause", (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return result


def main():
    video_candidates = [
        "images/Test_Video_2.mp4",
        "c:/vision_work/images/Test_Video_2.mp4",
        "Test_Video_2.mp4",
    ]
    video_path = next((p for p in video_candidates if os.path.exists(p)), None)
    if video_path is None:
        print("오류: Test_Video_2.mp4 를 찾을 수 없습니다.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: {video_path} 를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 목표 프레임 간격: 원본 FPS × PLAYBACK_SPEED 로 정밀 제어
    fps             = cap.get(cv2.CAP_PROP_FPS) or 30
    target_interval = 1.0 / (fps * PLAYBACK_SPEED)  # 초 단위

    # 원본 해상도 파악
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sw     = int(orig_w * PROCESS_SCALE)
    sh     = int(orig_h * PROCESS_SCALE)
    dw     = int(orig_w * DISPLAY_SCALE)
    dh     = int(orig_h * DISPLAY_SCALE)

    # ROI 마스크 사전 계산 (처리 해상도)
    roi_mask_s, _ = make_roi_mask(sh, sw)

    # CLAHE 1회 생성
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    tracker    = LaneTracker(alpha=0.18, max_lost_frames=25)
    show_debug = False
    paused     = False
    cached_edges = None

    reader = FrameReader(cap)
    reader.start()

    print(f"=== 차선 검출 (전체 프레임 모드  scale={PROCESS_SCALE}  disp={DISPLAY_SCALE}) ===")
    print("  Q: 종료 | D: 엣지 디버그 토글 | P: 일시정지")

    while True:
        if not paused:
            ret, frame = reader.read()
            if not ret:
                print("재생 완료.")
                break

            # ── 프레임 처리 시작 시각 기록 (속도 제어용) ─────
            frame_start = time.time()

            # ── 경량 처리 파이프라인 (매 프레임) ──────────────
            # 1. INTER_NEAREST 로 빠른 축소
            small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_NEAREST)

            # 2. 그레이스케일 변환
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # 3. CLAHE + 3×3 Blur (5×5 대비 ~4배 빠름) + Canny
            enhanced = clahe.apply(gray)
            blurred  = cv2.GaussianBlur(enhanced, (3, 3), 0)
            edges    = cv2.Canny(blurred, 60, 150)

            # 4. 사전 계산된 ROI 마스크 적용 (fillPoly 호출 없음)
            roi_edges = cv2.bitwise_and(edges, roi_mask_s)

            # 5. 허프 변환 (π/90 각도 해상도 → 탐색 각도 절반)
            lines = cv2.HoughLinesP(
                roi_edges, 1, np.pi / 90,
                threshold=25, minLineLength=18, maxLineGap=22
            )

            # 6. 차선 피팅 + 원본 해상도 변환
            cur_l, cur_r = extract_fits(lines, sh, sw)
            cur_l = scale_fit(cur_l, PROCESS_SCALE)
            cur_r = scale_fit(cur_r, PROCESS_SCALE)

            # 7. EMA 트래커 업데이트
            smooth_l, smooth_r, lost = tracker.update(cur_l, cur_r)

            # ── 표시 해상도에서 오버레이 합성 ─────────────────
            disp = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_NEAREST)

            # fit 계수를 표시 해상도로 변환
            dl = scale_fit(smooth_l, 1.0 / DISPLAY_SCALE) if smooth_l is not None else None
            dr = scale_fit(smooth_r, 1.0 / DISPLAY_SCALE) if smooth_r is not None else None

            result = draw_overlay(disp, dl, dr, lost)

            # 디버그 미니맵
            if show_debug:
                cached_edges = roi_edges
                mini_h, mini_w = sh // 3, sw // 3
                mini = cv2.resize(
                    cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR),
                    (mini_w, mini_h), interpolation=cv2.INTER_NEAREST
                )
                y0, x0 = 10, dw - mini_w - 10
                if y0 + mini_h <= dh and x0 + mini_w <= dw and x0 >= 0:
                    result[y0:y0+mini_h, x0:x0+mini_w] = mini
                    cv2.putText(result, "ROI Edge", (x0, y0+mini_h+12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Lane Detection", result)

            # ── 정밀 속도 제어 (time.sleep 기반) ─────────────────
            # 처리 시간을 제외한 나머지 시간만 sleep → 목표 간격 유지
            elapsed   = time.time() - frame_start
            sleep_sec = target_interval - elapsed
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('d'): show_debug = not show_debug
        elif key == ord('p'): paused = not paused

    reader.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()