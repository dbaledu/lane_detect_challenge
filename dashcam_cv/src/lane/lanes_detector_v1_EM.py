# ==============================================================================
# [코드 개요] 야간 도심 차선 검출 파이프라인 (카메라 뷰 직접 처리, BEV 미사용)
# ==============================================================================
#
# ■ 전체 파이프라인 (프레임 1장 기준, process_frame)
#
#   입력 프레임(640x360 리사이즈)
#     1) Grayscale 변환
#     2) ROI 세로 슬라이스 추출 (y: 0.65h ~ 0.88h)
#     3) CLAHE        : 야간 저조도 대비 향상
#     4) Gaussian Blur(5x5) + Canny(45/135) : 엣지 검출
#     5) 사다리꼴 마스크 : 주행 영역 밖 엣지 제거
#     6) 횡단보도/모션 판별 : 차선 표시 ON/OFF 결정 (CrosswalkAndMotionDetector)
#     7) 세로 방향 팽창(1x5) : 끊어진 점선/마모 차선 연결
#                              (횡단보도 감지 시 중앙 영역 엣지 제거)
#     8) HoughLinesP  : 직선 선분 후보 추출
#     9) 기울기/위치 필터 + 좌/우 분류 : 중앙선을 넘는 후보 사전 차단
#    10) 대표 직선 산출 : IQR 이상치 제거 + 길이 가중 평균 (fit_representative_line)
#    11) 시계열 추적   : 기하 검증 + EMA 스무딩 + 손실 시 유지 (ADASLaneTrackerV3)
#    12) 렌더링        : 안전 마진 가드 -> 네온 폴리곤/경계선/중앙 점선 오버레이
#   출력 프레임 -> 화면 미리보기 + 영상 파일 저장 (run_video_pipeline)
#
# ■ 구성 요소
#
#   [클래스]
#   - CrosswalkAndMotionDetector
#       횡단보도(엣지 교번 횟수)와 차량 정지/주행(프레임 차분)을 판별하여
#       차선 표시를 숨길지(display_lanes) 결정하는 상태 머신.
#   - ADASLaneTrackerV3
#       좌/우 차선의 시계열 추적기. EMA 스무딩, 한쪽만 검출될 때의 평행 보정,
#       검출 실패 시 최대 35프레임 유지(HOLD), 좌/우 교차(꼬임) 기하 검증 담당.
#
#   [함수]
#   - line_to_points(y_bottom, y_top, line_params)
#       (기울기, 절편) -> [x_bottom, y_bottom, x_top, y_top] 좌표 변환.
#   - fit_representative_line(line_candidates, y_bottom, y_top)
#       후보 선분들에서 대표 직선 1개를 산출 (IQR 필터 + 길이 가중 평균).
#   - process_frame(frame, tracker, crosswalk_detector, clahe, prev_gray_roi)
#       위 파이프라인 1~12단계를 수행. (결과 프레임, 추적 상태, ROI 그레이) 반환.
#   - draw_dashed_line(img, pt1, pt2, ...)
#       차로 중앙 유도 점선 그리기 유틸리티.
#   - run_video_pipeline(...)
#       비디오 읽기 -> process_frame -> 저장/미리보기 메인 루프, 진행률/FPS 출력.
#
#   [실행]
#   - __main__ : argparse 옵션 (--input, --output, --start_frame,
#                --max_frames, --no_preview) 처리 후 run_video_pipeline 호출.
#
# ■ 추적기 상태값(status)
#   TRACKING / PARTIAL_LEFT / PARTIAL_RIGHT / RECOVER_LEFT / RECOVER_RIGHT /
#   HOLD (n/35) / LOST / DETECTING / CROSSWALK_PAUSE
# ==============================================================================

import os
import sys
import time
import glob
import argparse
from collections import deque
import cv2
import numpy as np


class CheckerboardCalibrator:
    """
    체커보드(Checkerboard / Chessboard) 기반 카메라 캘리브레이션 및 렌즈 왜곡 보정기
    - 차량용 광각 렌즈(Dashcam)의 왜곡(배럴/어안 왜곡)을 보정하여 곡선으로 왜곡된 차선을 실제 물리적 직선으로 복원
    - cv2.findChessboardCorners + cv2.cornerSubPix 서브픽셀 정밀 코너 추출
    - cv2.calibrateCamera를 통한 카메라 매트릭스(K) 및 왜곡 계수(Distortion Coeffs: k1, k2, p1, p2, k3) 계산
    - 실시간 비디오 처리를 위해 cv2.initUndistortRectifyMap + cv2.remap 최적화 적용 (120+ FPS 고속 렌더링)
    - .npz 파일 저장 및 자동 로드 지원, 체커보드 이미지 부재 시 내장 대시캠 광각 왜곡 모델 즉시 적용 지원
    """
    def __init__(self, calib_file=None, pattern_size=(9, 6), square_size=25.0):
        self.pattern_size = pattern_size  # (columns, rows) 내부 코너 수 (예: 9x6 체커보드)
        self.square_size = square_size    # 격자 한 칸의 실제 크기 (mm 단위)
        self.camera_matrix = None         # K: 3x3 내부 파라미터 행렬 (fx, fy, cx, cy)
        self.dist_coeffs = None          # dist: [k1, k2, p1, p2, k3]
        self.reproj_error = None         # 재투영 오차 (픽셀)
        self.img_size = None             # (w, h)
        self.is_calibrated = False
        self.map1 = None                 # cv2.remap 고속 처리를 위한 사전 계산 룩업 테이블
        self.map2 = None

        # 서브픽셀 코너 탐색 반복 종료 조건
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if calib_file and os.path.exists(calib_file):
            self.load_calibration(calib_file)

    def calibrate_from_images(self, image_source, output_npz=None, show_corners=False):
        """
        체커보드 이미지 폴더 또는 이미지 파일 리스트로부터 카메라 캘리브레이션 수행
        :param image_source: 체커보드 이미지가 위치한 폴더 경로 또는 파일 경로 리스트
        :param output_npz: 계산된 파라미터를 저장할 .npz 파일 경로 (선택 사항)
        :param show_corners: 코너 검출 결과를 화면에 시각화할지 여부
        :return: (성공 여부, 처리된 이미지 수, 평균 재투영 오차)
        """
        if isinstance(image_source, str):
            if os.path.isdir(image_source):
                image_files = []
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
                    image_files.extend(glob.glob(os.path.join(image_source, ext)))
                    image_files.extend(glob.glob(os.path.join(image_source, ext.upper())))
                image_files = sorted(list(set(image_files)))
            elif os.path.isfile(image_source):
                image_files = [image_source]
            else:
                image_files = sorted(glob.glob(image_source))
        else:
            image_files = list(image_source)

        if not image_files:
            print(f"[경고] 체커보드 캘리브레이션 이미지를 찾을 수 없습니다: {image_source}")
            return False, 0, None

        cols, rows = self.pattern_size
        # 3D 실제 세계 좌표점 생성 (z = 0 평면 기준)
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_size

        objpoints = []  # 3D 공간 상의 실제 점
        imgpoints = []  # 2D 이미지 평면 상의 픽셀 점
        detected_count = 0
        img_shape = None

        print(f"\n[체커보드 캘리브레이션 시작] 패턴: {cols}x{rows} 코너 | 격자: {self.square_size}mm | 대상 이미지: {len(image_files)}장")

        for idx, fname in enumerate(image_files):
            img = cv2.imread(fname)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_shape is None:
                img_shape = gray.shape[::-1]  # (w, h)

            # 체커보드 코너 탐색
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)

            if ret:
                # 서브픽셀 정밀화
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.subpix_criteria)
                objpoints.append(objp)
                imgpoints.append(corners_sub)
                detected_count += 1

                if show_corners:
                    vis = cv2.drawChessboardCorners(img.copy(), (cols, rows), corners_sub, ret)
                    cv2.imshow("Checkerboard Corners", vis)
                    cv2.waitKey(250)

                print(f"  [{idx+1:2d}/{len(image_files):2d}] 코너 검출 성공: {os.path.basename(fname)}")
            else:
                print(f"  [{idx+1:2d}/{len(image_files):2d}] 코너 검출 실패: {os.path.basename(fname)}")

        if show_corners:
            cv2.destroyWindow("Checkerboard Corners")

        if detected_count < 3:
            print(f"[오류] 캘리브레이션을 위해서는 최소 3장 이상의 유효한 체커보드 이미지가 필요합니다. (검출: {detected_count}장)")
            return False, detected_count, None

        print(f"\n[최적화 연산] {detected_count}장의 이미지 데이터로 cv2.calibrateCamera 수행 중...")
        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )

        # 재투영 오차(Re-projection Error) 계산
        total_error = 0.0
        for i in range(len(objpoints)):
            projected_pts, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            err = cv2.norm(imgpoints[i], projected_pts, cv2.NORM_L2) / len(projected_pts)
            total_error += err
        mean_error = total_error / len(objpoints)

        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.reproj_error = mean_error
        self.img_size = img_shape
        self.is_calibrated = True

        # 고속 remap 테이블 초기화
        self.init_remap(img_shape)

        print(f"==================================================")
        print(f" [체커보드 카메라 캘리브레이션 완료]")
        print(f"  - 유효 이미지 수 : {detected_count} / {len(image_files)}")
        print(f"  - 평균 재투영 오차 : {mean_error:.4f} px (1.0 미만 권장)")
        print(f"  - 카메라 초점 거리 : fx={mtx[0, 0]:.2f}, fy={mtx[1, 1]:.2f}")
        print(f"  - 광학 중심점     : cx={mtx[0, 2]:.2f}, cy={mtx[1, 2]:.2f}")
        print(f"  - 렌즈 왜곡 계수   : k1={dist[0, 0]:.4f}, k2={dist[0, 1]:.4f}, p1={dist[0, 2]:.4f}, p2={dist[0, 3]:.4f}")
        print(f"==================================================")

        if output_npz:
            self.save_calibration(output_npz)

        return True, detected_count, mean_error

    def setup_default_dashcam_calibration(self, img_shape=(640, 360), fov_deg=115.0, k1=-0.14, k2=0.025, p1=0.0, p2=0.0):
        """
        체커보드 이미지 촬영 전이라도 즉시 테스트 가능한 차량용 대시캠 광각 렌즈 기본 프로파일
        :param img_shape: (width, height)
        :param fov_deg: 대각/수평 시야각 추정치 (대시캠 표준 115도)
        :param k1, k2: 방사 왜곡 계수 (음수 k1은 배럴 왜곡 보정)
        """
        w, h = img_shape
        fov_rad = np.radians(fov_deg)
        fx = (w / 2.0) / np.tan(fov_rad / 2.0)
        fy = fx
        cx = w / 2.0
        cy = h / 2.0

        self.camera_matrix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.dist_coeffs = np.array([[k1, k2, p1, p2, 0.0]], dtype=np.float64)
        self.reproj_error = 0.0
        self.img_size = (w, h)
        self.is_calibrated = True
        self.init_remap((w, h))

        print(f"[안내] 차량용 대시캠 표준 왜곡 보정 프로파일 적용 완료 ({w}x{h}, FOV={fov_deg}°, k1={k1}, k2={k2})")

    def init_remap(self, img_size, alpha=0.0):
        """
        실시간 비디오 프레임 처리 속도 극대화를 위해 왜곡 보정 맵(remap) 사전 연산
        - cv2.undistort 대비 3~5배 빠른 속도로 60~120+ FPS 유지
        :param img_size: (width, height)
        :param alpha: 0.0(보정 후 유효 픽셀만 남김) ~ 1.0(모든 원본 픽셀 보존)
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return
        w, h = img_size
        self.img_size = (w, h)
        new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha, (w, h)
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, new_camera_mtx, (w, h), cv2.CV_32FC1
        )

    def undistort(self, frame):
        """
        비디오 프레임 렌즈 왜곡 보정 (Undistortion)
        - 체커보드로 보정된 왜곡 계수를 사용하여 직선 왜곡을 복원
        """
        if not self.is_calibrated or frame is None:
            return frame

        h, w = frame.shape[:2]
        if self.map1 is None or self.img_size != (w, h):
            self.init_remap((w, h))

        # remap을 통한 초고속 렌즈 왜곡 보정
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def save_calibration(self, npz_path):
        """캘리브레이션 파라미터를 .npz 파일로 저장"""
        if not self.is_calibrated:
            print("[경고] 캘리브레이션 데이터가 없어 저장할 수 없습니다.")
            return False
        os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
        np.savez(
            npz_path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            reproj_error=self.reproj_error if self.reproj_error is not None else 0.0,
            pattern_size=np.array(self.pattern_size),
            square_size=self.square_size
        )
        print(f"[저장 완료] 체커보드 캘리브레이션 파라미터 저장됨: {npz_path}")
        return True

    def load_calibration(self, npz_path):
        """저장된 .npz 캘리브레이션 파일 로드"""
        if not os.path.exists(npz_path):
            return False
        try:
            data = np.load(npz_path)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.reproj_error = float(data['reproj_error']) if 'reproj_error' in data else 0.0
            if 'pattern_size' in data:
                self.pattern_size = tuple(data['pattern_size'])
            if 'square_size' in data:
                self.square_size = float(data['square_size'])
            self.is_calibrated = True
            print(f"[로드 완료] 체커보드 캘리브레이션 데이터 로드: {npz_path} (재투영 오차: {self.reproj_error:.4f}px)")
            return True
        except Exception as e:
            print(f"[오류] 캘리브레이션 파일 로드 실패 ({npz_path}): {e}")
            return False


class CrosswalkAndMotionDetector:
    """
    횡단보도 감지 및 차량 모션 판별기
    - 횡단보도 진입 또는 정지 대기 시: 차선 표시 일시 정지 (모든 표시 숨김)
    - 차량 재주행 및 횡단보도 통과 시: 차선 표시 복구
    """
    def __init__(self, w=640, h=360):
        self.w = w
        self.h = h
        self.is_crosswalk = False
        self.is_stopped = False
        self.display_lanes = True
        self.stop_duration_frames = 0
        self.motion_history = deque(maxlen=15)

    def update(self, current_gray_roi, prev_gray_roi, edge_roi):
        # 1. 횡단보도 엣지 교번 빈도 검출 (ROI 영역 내부에서 분석)
        roi_h = edge_roi.shape[0]
        scan_y1 = int(roi_h * 0.35)
        scan_y2 = int(roi_h * 0.65)

        zebra_score = 0
        for y_pos in [scan_y1, scan_y2]:
            row = edge_roi[y_pos, int(self.w * 0.28):int(self.w * 0.72)]
            transitions = np.count_nonzero(np.diff(row > 128))
            if transitions >= 14:
                zebra_score += 1

        self.is_crosswalk = (zebra_score >= 1)

        # 2. 차량 모션(주행/정지) 판별 (ROI 영역 차이값)
        motion_val = 0.0
        if prev_gray_roi is not None:
            diff = cv2.absdiff(
                current_gray_roi[:, int(self.w * 0.25):int(self.w * 0.75)],
                prev_gray_roi[:, int(self.w * 0.25):int(self.w * 0.75)]
            )
            motion_val = float(np.mean(diff))
            self.motion_history.append(motion_val)

        avg_motion = np.mean(self.motion_history) if self.motion_history else 5.0
        self.is_stopped = (avg_motion < 1.6)

        # 3. 횡단보도 등장 시 표시 일시정지, 재주행 시 복구
        if self.is_crosswalk:
            self.display_lanes = False
            self.stop_duration_frames += 1
        elif self.is_stopped and self.stop_duration_frames > 0:
            self.display_lanes = False
            self.stop_duration_frames += 1
        else:
            if avg_motion >= 1.5:
                self.display_lanes = True
                self.stop_duration_frames = 0

        return self.display_lanes, self.is_crosswalk


class ADASLaneTrackerV3:
    """
    시계열 지수이동평균(EMA) 및 기하학적 형상 보존 기반 차선 추적기
    - [수정점 2] 꼬임(Cross-over) 방지 기하학 가드 적용
    - 양측 차선 안정적 검출 시 EMA 스무딩 및 동적 차로 폭 학습
    - 편측 차선 검출 또는 일시적 가림 시 이전 차선 상태 유지(Hold) 및 안전 복구
    """
    def __init__(self, max_hold_frames=35, alpha=0.22):
        self.max_hold_frames = max_hold_frames
        self.alpha = alpha

        self.left_line = None
        self.right_line = None
        self.loss_count = 0
        self.status = "DETECTING"

        # 학습된 기준 차로 폭 (640 해상도 기준)
        self.est_w_bottom = 255.0
        self.est_w_top = 58.0

    @staticmethod
    def _get_slope(line):
        if line is None:
            return None
        x1, y1, x2, y2 = line
        if x2 == x1:
            return None
        return (y2 - y1) / (x2 - x1)

    def _is_valid_geometry(self, left, right, y_top):
        """
        [수정점 2 핵심] 차선 꼬임(Cross-over) 방지 기하학 검증
        - 좌/우 차선이 바닥 및 상단에서 절대 교차하지 않아야 함
        - 두 차선의 교점(소실점)이 가시 주행 영역(y >= y_top) 내에 있으면 꼬임으로 판정하여 즉시 거부
        """
        if left is None or right is None:
            return False

        lx_b, ly_b, lx_t, ly_t = left
        rx_b, ry_b, rx_t, ry_t = right

        # 1. 수평 순서 검증 (좌측선은 항상 우측선보다 왼쪽에 위치)
        if lx_b >= rx_b - 120:  # 바닥 폭 최소 120px 보장
            return False
        if lx_t >= rx_t - 20:   # 상단 폭 최소 20px 보장 (0 이하 교차 절대 불허)
            return False

        w_b = rx_b - lx_b
        if not (130 <= w_b <= 380):
            return False

        w_t = rx_t - lx_t
        if not (20 <= w_t <= 140):
            return False

        # 2. 교점(소실점) 위치 검사: 주행 영역 내부에서 교차(X자 꼬임) 방지
        ml = self._get_slope(left)
        mr = self._get_slope(right)
        if ml is not None and mr is not None and abs(ml - mr) > 1e-4:
            bl = ly_b - ml * lx_b
            br = ry_b - mr * rx_b
            # x = (y - b)/m => y = (br*ml - bl*mr) / (ml - mr)
            y_intersect = (br * ml - bl * mr) / (ml - mr)
            # 교점이 y_top보다 아래에 있으면 주행 화면 내에서 두 선이 교차함(꼬임 발생)
            if y_intersect >= y_top - 5:
                return False

        return True

    def update(self, det_left, det_right, y_top, is_paused=False):
        if is_paused:
            self.status = "CROSSWALK_PAUSE"
            return self._export()

        # 1. 양측 차선 모두 검출 + 기하학 꼬임 방지 유효성 통과
        if det_left is not None and det_right is not None and self._is_valid_geometry(det_left, det_right, y_top):
            if self.left_line is None or self.right_line is None:
                self.left_line = np.array(det_left, dtype=float)
                self.right_line = np.array(det_right, dtype=float)
            else:
                self.left_line = (1 - self.alpha) * self.left_line + self.alpha * np.array(det_left, dtype=float)
                self.right_line = (1 - self.alpha) * self.right_line + self.alpha * np.array(det_right, dtype=float)

            # 차로 폭 동적 학습
            w_b = float(self.right_line[0] - self.left_line[0])
            w_t = float(self.right_line[2] - self.left_line[2])
            if 140 <= w_b <= 380:
                self.est_w_bottom = 0.9 * self.est_w_bottom + 0.1 * w_b
            if 20 <= w_t <= 140:
                self.est_w_top = 0.9 * self.est_w_top + 0.1 * w_t

            self.loss_count = 0
            self.status = "TRACKING"
            return self._export()

        # 2. 좌측 차선만 검출 (기존 추적 상태 유지하면서 좌측선 갱신)
        elif det_left is not None and self.left_line is not None and self.right_line is not None:
            self.left_line = (1 - self.alpha) * self.left_line + self.alpha * np.array(det_left, dtype=float)
            self.loss_count = min(self.loss_count + 1, self.max_hold_frames)
            self.status = f"PARTIAL_LEFT (HOLD {self.loss_count}/{self.max_hold_frames})"
            return self._export()

        # 3. 우측 차선만 검출 (기존 추적 상태 유지하면서 우측선 갱신)
        elif det_right is not None and self.left_line is not None and self.right_line is not None:
            self.right_line = (1 - self.alpha) * self.right_line + self.alpha * np.array(det_right, dtype=float)
            self.loss_count = min(self.loss_count + 1, self.max_hold_frames)
            self.status = f"PARTIAL_RIGHT (HOLD {self.loss_count}/{self.max_hold_frames})"
            return self._export()

        # 4. 차선 미검출 또는 기하 검증 미통과 시 이전 차선 상태 홀드
        else:
            self._handle_loss()
            return self._export()

    def _handle_loss(self):
        if self.left_line is not None and self.right_line is not None:
            self.loss_count += 1
            if self.loss_count <= self.max_hold_frames:
                self.status = f"HOLD ({self.loss_count}/{self.max_hold_frames})"
            else:
                self.left_line = None
                self.right_line = None
                self.status = "LOST"
        else:
            self.status = "DETECTING"

    def _export(self):
        out_l = self.left_line.astype(int).tolist() if self.left_line is not None else None
        out_r = self.right_line.astype(int).tolist() if self.right_line is not None else None
        return out_l, out_r, self.status


def line_to_points(y_bottom, y_top, line_params):
    if line_params is None:
        return None
    slope, intercept = line_params
    if abs(slope) < 1e-4:
        return None
    try:
        x_bottom = int((y_bottom - intercept) / slope)
        x_top = int((y_top - intercept) / slope)
        if abs(x_bottom) > 3000 or abs(x_top) > 3000:
            return None
        return [x_bottom, y_bottom, x_top, y_top]
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


def fit_representative_line(line_candidates, y_bottom, y_top):
    if not line_candidates:
        return None

    slopes = np.array([s for s, _, _ in line_candidates])
    intercepts = np.array([i for _, i, _ in line_candidates])
    weights = np.array([l for _, _, l in line_candidates])

    if len(slopes) >= 4:
        q1, q3 = np.percentile(slopes, [25, 75])
        iqr = q3 - q1
        inlier_mask = (slopes >= q1 - 1.5 * iqr) & (slopes <= q3 + 1.5 * iqr)
        if inlier_mask.sum() >= 2:
            slopes = slopes[inlier_mask]
            intercepts = intercepts[inlier_mask]
            weights = weights[inlier_mask]

    if weights.sum() == 0:
        return None

    avg_slope = np.average(slopes, weights=weights)
    avg_intercept = np.average(intercepts, weights=weights)

    return line_to_points(y_bottom, y_top, (avg_slope, avg_intercept))


def process_frame(frame, tracker, crosswalk_detector, clahe, prev_gray_roi=None, calibrator=None):
    """
    단일 프레임 처리 함수
    - [체커보드 캘리브레이션] 렌즈 왜곡 보정(Undistortion) 선적용
    - [수정점 1] 버드아이 뷰(BEV) 삭제
    - [수정점 2] 지정 사다리꼴 ROI 적용 및 꼬임(Cross-over) 방지
    - [수정점 3] 모든 HUD 삭제 & 고시인성 네온 차로 & ROI-CLAHE
    """
    # 0. 체커보드 캘리브레이션 기반 광각 렌즈 왜곡 보정 (Undistortion)
    if calibrator is not None and getattr(calibrator, "is_calibrated", False):
        frame = calibrator.undistort(frame)

    h, w = frame.shape[:2]
    mid_x = w / 2

    # [수정점 2] 사용자 지정 ROI 영역 설정:
    # 상단 좌측: (int(w * 0.44), int(h * 0.65))
    # 상단 우측: (int(w * 0.56), int(h * 0.65))
    # 하단 우측: (int(w * 0.82), int(h * 0.88))
    # 하단 좌측: (int(w * 0.18), int(h * 0.88))
    y_top = int(h * 0.65)
    y_bottom = int(h * 0.88)
    roi_h = y_bottom - y_top

    # 1. 흑백 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. [수정점 3] ROI 세로 슬라이스에만 CLAHE 및 Canny 집중 적용 (상단 65% 배제)
    gray_roi = gray[y_top:y_bottom, :]

    # ROI 전용 CLAHE
    clahe_roi = clahe.apply(gray_roi)

    # ROI 전용 Canny 윤곽선 검출
    blurred_roi = cv2.GaussianBlur(clahe_roi, (5, 5), 1.2)
    canny_roi = cv2.Canny(blurred_roi, threshold1=45, threshold2=135)

    # 3. [수정점 2] 사용자 지정 사다리꼴 ROI 마스크 생성 (ROI 로컬 좌표계)
    roi_mask = np.zeros((roi_h, w), dtype=np.uint8)
    roi_pts = np.array([
        [int(w * 0.18), roi_h],  # 하단 좌측
        [int(w * 0.44), 0],      # 상단 좌측
        [int(w * 0.56), 0],      # 상단 우측
        [int(w * 0.82), roi_h]   # 하단 우측
    ], np.int32)
    cv2.fillPoly(roi_mask, [roi_pts], 255)
    masked_canny_roi = cv2.bitwise_and(canny_roi, roi_mask)

    # 4. 횡단보도 감지 및 모션 분석 (표시 중단 / 재개 판별)
    display_lanes, is_crosswalk = crosswalk_detector.update(
        gray_roi, prev_gray_roi, masked_canny_roi
    )

    # 5. 세로 모폴로지 팽창 (마모 차선 조각 결합)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    morphed_roi = cv2.dilate(masked_canny_roi, vert_kernel, iterations=1)

    # 횡단보도 감지 시 중앙 잡선 억제
    if is_crosswalk:
        morphed_roi[int(roi_h * 0.3):, int(w * 0.30):int(w * 0.70)] = 0

    # 6. 허프 변환 (HoughLinesP)
    lines = cv2.HoughLinesP(
        morphed_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=22,
        maxLineGap=45
    )

    left_candidates = []
    right_candidates = []

    # 7. 기하학적 정제 및 [꼬임 방지 좌우 분류]
    if lines is not None:
        flat_lines = np.array(lines).reshape(-1, 4)
        for x1, y1_rel, x2, y2_rel in flat_lines:
            if x1 == x2:
                continue
            # ROI 로컬 y좌표 -> 전체 화면 절대 y좌표 변환
            y1 = y1_rel + y_top
            y2 = y2_rel + y_top

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if abs(slope) < 0.42 or abs(slope) > 3.2:
                continue

            x_top_pred = (y_top - intercept) / slope
            x_bot_pred = (y_bottom - intercept) / slope

            # [수정점 2: 좌/우 분류 시 교차 및 꼬임 원천 방지]
            if slope < -0.42 and x_bot_pred < mid_x and x1 < mid_x + 20:
                if int(w * 0.34) <= x_top_pred <= mid_x - 8:
                    left_candidates.append((slope, intercept, length))

            elif slope > 0.42 and x_bot_pred > mid_x and x1 > mid_x - 20:
                if mid_x + 8 <= x_top_pred <= int(w * 0.66):
                    right_candidates.append((slope, intercept, length))

    # 8. 대표 직선 산출 (IQR 기반 가중 평균)
    raw_left = fit_representative_line(left_candidates, y_bottom, y_top)
    raw_right = fit_representative_line(right_candidates, y_bottom, y_top)

    # 9. 시계열 추적기 갱신 (꼬임 방지 유효성 검증)
    smooth_left, smooth_right, status = tracker.update(
        raw_left, raw_right,
        y_top=y_top,
        is_paused=(not display_lanes)
    )

    # 10. [수정점 1 & 2] 고시인성 차선 렌더링 + 꼬임 방지 가드 (HUD 미출력)
    result = frame.copy()

    if display_lanes and smooth_left is not None and smooth_right is not None:
        lx1, ly1, lx2, ly2 = smooth_left
        rx1, ry1, rx2, ry2 = smooth_right

        # [수정점 2 핵심: 렌더링 직전 꼬임 방지 안전 마진 강제 적용 (Anti-Twisting Guard)]
        min_gap_top = 26
        if rx2 - lx2 < min_gap_top:
            center_top = (lx2 + rx2) / 2
            lx2 = int(center_top - min_gap_top / 2)
            rx2 = int(center_top + min_gap_top / 2)

        min_gap_bottom = 160
        if rx1 - lx1 < min_gap_bottom:
            center_bot = (lx1 + rx1) / 2
            lx1 = int(center_bot - min_gap_bottom / 2)
            rx1 = int(center_bot + min_gap_bottom / 2)

        overlay = frame.copy()

        # (1) 고시인성 네온 에메랄드 주행 차로 폴리곤
        lane_poly = np.array([[lx1, ly1], [lx2, ly2], [rx2, ry2], [rx1, ry1]], np.int32)
        cv2.fillPoly(overlay, [lane_poly], (50, 255, 120))
        result = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        # (2) 좌측 차선: 일렉트릭 시안(8px 외곽선) + 화이트 코어(3px 발광)
        cv2.line(result, (lx1, ly1), (lx2, ly2), (255, 210, 0), 8)
        cv2.line(result, (lx1, ly1), (lx2, ly2), (255, 255, 255), 3)

        # (3) 우측 차선: 네온 코랄/레드(8px 외곽선) + 화이트 코어(3px 발광)
        cv2.line(result, (rx1, ry1), (rx2, ry2), (30, 70, 255), 8)
        cv2.line(result, (rx1, ry1), (rx2, ry2), (255, 255, 255), 3)

        # (4) 차로 중앙 유도 궤적 점선
        cx1, cy1 = int((lx1 + rx1) / 2), ly1
        cx2, cy2 = int((lx2 + rx2) / 2), ly2
        draw_dashed_line(result, (cx1, cy1), (cx2, cy2), color=(255, 255, 220), thickness=3, dash_length=14)

    return result, status, gray_roi


def draw_dashed_line(img, pt1, pt2, color=(255, 255, 255), thickness=2, dash_length=12):
    """중앙 궤적 유도용 고시인성 점선 그리기 유틸리티"""
    dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist < 1:
        return
    num_dashes = int(dist / dash_length)
    for i in range(0, num_dashes, 2):
        r1 = i / num_dashes
        r2 = min((i + 1) / num_dashes, 1.0)
        p1 = (int(pt1[0] + r1 * (pt2[0] - pt1[0])), int(pt1[1] + r1 * (pt2[1] - pt1[1])))
        p2 = (int(pt1[0] + r2 * (pt2[0] - pt1[0])), int(pt1[1] + r2 * (pt2[1] - pt1[1])))
        cv2.line(img, p1, p2, color, thickness)


def run_video_pipeline(
    video_path=r"c:\vision_work\video\Test_Video_2.mp4",
    output_path=r"c:\vision_work\output_Test_Video_2_v3_clean_lane.mp4",
    start_frame=0,
    max_frames=0,
    show_preview=True,
    calib_file="camera_calib.npz",
    calibrate_dir=None,
    use_undistort=True,
    checkerboard_size=(9, 6),
    square_size=25.0
):
    """
    비디오 입력에서 체커보드 캘리브레이션 왜곡 보정 및 클린 뷰 차선 검출 파이프라인을 실행하는 메인 루프
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[오류] 비디오 파일을 찾을 수 없습니다: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    proc_w, proc_h = 640, 360

    # 1. 체커보드 캘리브레이터(CheckerboardCalibrator) 준비
    calibrator = CheckerboardCalibrator(
        calib_file=calib_file,
        pattern_size=checkerboard_size,
        square_size=square_size
    )

    if calibrate_dir and os.path.exists(calibrate_dir):
        print(f"[체커보드 캘리브레이션] '{calibrate_dir}' 디렉토리 이미지들로부터 카메라 캘리브레이션 수행...")
        calibrator.calibrate_from_images(calibrate_dir, output_npz=calib_file)
    elif use_undistort and not calibrator.is_calibrated:
        print("[체커보드 캘리브레이션] 기존 캘리브레이션 파일(.npz) 부재로 내장 대시캠 광각 왜곡 모델(k1=-0.14)을 자동 구성합니다.")
        calibrator.setup_default_dashcam_calibration((proc_w, proc_h))

    calib_status_str = "적용됨 (체커보드 왜곡 보정 활성화)" if (use_undistort and calibrator.is_calibrated) else "미적용 (RAW 입력)"

    # 2. 횡단보도 및 모션 감지기
    crosswalk_detector = CrosswalkAndMotionDetector(proc_w, proc_h)

    # 3. ROI 전용 CLAHE 객체 생성
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    # 4. 시계열 지수이동평균(EMA) 및 꼬임 방지 기능 탑재 차선 추적기
    tracker = ADASLaneTrackerV3(max_hold_frames=35, alpha=0.22)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (proc_w, proc_h))

    remaining_frames = total_frames - start_frame
    frames_to_process = remaining_frames if max_frames <= 0 else min(remaining_frames, max_frames)

    print("\n" + "=" * 78)
    print(" [ADAS v3 야간 도심 차선 검출 비디오 파이프라인 - 체커보드 보정 & 클린 뷰]")
    print(f" - 입력 영상: {video_path}")
    print(f" - 처리 제원: {proc_w}x{proc_h} @ {fps:.1f} FPS | 대상 프레임: {frames_to_process}")
    print(f" - [카메라 캘리브레이션] 렌즈 왜곡 보정: {calib_status_str}")
    print(f" - [수정점 1] 버드아이 뷰(BEV) 삭제 완료")
    print(f" - [수정점 2] 지정 사다리꼴 ROI 적용 (y: 0.65~0.88, 상단 0.44~0.56, 하단 0.18~0.82) & 꼬임 방지")
    print(f" - [수정점 3] 모든 HUD 삭제 & 고시인성 네온 에메랄드 차로 & ROI-CLAHE")
    print(f" - 저장 경로: {output_path}")
    if show_preview:
        print(" - 실시간 모니터링: 활성화 ('q' 또는 ESC 키로 중단 가능)")
    print("=" * 78 + "\n")

    window_name = "ADAS Clean Lane Detection (Calibrated Anti-Twisting ROI)"
    if show_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

    frame_idx = 0
    t_start = time.time()
    prev_gray_roi = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (0 < max_frames <= frame_idx):
                break

            frame_idx += 1
            resized = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            # 파이프라인 처리 (체커보드 렌즈 왜곡 보정 -> 사다리꼴 ROI -> CLAHE -> Canny -> Hough -> EMA Smoothing)
            res_frame, status, cur_gray_roi = process_frame(
                resized,
                tracker,
                crosswalk_detector,
                clahe,
                prev_gray_roi,
                calibrator=calibrator if use_undistort else None
            )
            prev_gray_roi = cur_gray_roi

            writer.write(res_frame)

            if show_preview:
                cv2.imshow(window_name, res_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print(f"\n[안내] 사용자에 의해 안전하게 중단되었습니다 (프레임 {frame_idx}).")
                    break

            if frame_idx % 120 == 0 or frame_idx == frames_to_process:
                elapsed = time.time() - t_start
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0
                pct = (frame_idx / frames_to_process) * 100
                lane_disp_str = "ON" if crosswalk_detector.display_lanes else "PAUSED (CROSSWALK)"
                print(f"진행률: [{frame_idx:4d}/{frames_to_process}] ({pct:5.1f}%) | "
                      f"속도: {proc_fps:4.1f} FPS | 상태: {status} | 차선표시: {lane_disp_str}")

    finally:
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    total_time = time.time() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    print("\n" + "=" * 78)
    print(f"[처리 완료!] 총 {frame_idx} 프레임 처리 완료 (소요 시간: {total_time:.2f}초, 평균: {avg_fps:.1f} FPS)")
    print(f"[저장 파일] {output_path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAS Clean Lane Detection Pipeline with Checkerboard Calibration")
    parser.add_argument("--mode", type=str, default="run", choices=["run", "calibrate"],
                        help="실행 모드: run(비디오 차선 검출 파이프라인), calibrate(체커보드 캘리브레이션 전용)")
    parser.add_argument("--input", type=str, default=r"c:\vision_work\video\Test_Video_2.mp4",
                        help="입력 비디오 파일 경로")
    parser.add_argument("--output", type=str, default=r"c:\vision_work\output_Test_Video_2_v3_clean_lane.mp4",
                        help="출력 비디오 파일 경로")
    parser.add_argument("--start_frame", type=int, default=0, help="시작 프레임 번호")
    parser.add_argument("--max_frames", type=int, default=0, help="처리할 최대 프레임 수 (0은 전체 처리)")
    parser.add_argument("--no_preview", action="store_true", help="실시간 미리보기 창 비활성화 (인코딩 속도 향상)")

    # 체커보드 캘리브레이션 옵션
    parser.add_argument("--calib_dir", type=str, default=None,
                        help="체커보드 캘리브레이션 이미지가 있는 폴더 경로 (지정 시 자동 캘리브레이션 수행)")
    parser.add_argument("--calib_file", type=str, default="camera_calib.npz",
                        help="캘리브레이션 파라미터 파일 경로 (.npz)")
    parser.add_argument("--undistort", action="store_true", default=True,
                        help="체커보드 왜곡 보정 적용 (기본값: True)")
    parser.add_argument("--no_undistort", action="store_false", dest="undistort",
                        help="렌즈 왜곡 보정 끄기 (RAW 영상 직접 처리)")
    parser.add_argument("--cb_cols", type=int, default=9,
                        help="체커보드 가로 내부 코너 수 (기본: 9)")
    parser.add_argument("--cb_rows", type=int, default=6,
                        help="체커보드 세로 내부 코너 수 (기본: 6)")
    parser.add_argument("--square_size", type=float, default=25.0,
                        help="체커 격자 한 변의 실제 크기 (mm 단위, 기본: 25.0)")

    args = parser.parse_args()

    if args.mode == "calibrate":
        # 체커보드 캘리브레이션 단독 실행 모드
        if not args.calib_dir:
            print("[오류] --calib_dir 인자로 체커보드 이미지 폴더 경로를 지정해야 합니다.")
            print("예시: python lane_detection_v3_video_pipeline.py --mode calibrate --calib_dir ./calib_images --calib_file camera_calib.npz")
            sys.exit(1)
        calibrator = CheckerboardCalibrator(pattern_size=(args.cb_cols, args.cb_rows), square_size=args.square_size)
        success, count, err = calibrator.calibrate_from_images(args.calib_dir, output_npz=args.calib_file, show_corners=not args.no_preview)
        if success:
            print(f"[성공] 캘리브레이션 파일 저장 완료: {args.calib_file}")
            sys.exit(0)
        else:
            print("[실패] 캘리브레이션을 완료하지 못했습니다.")
            sys.exit(1)

    # 비디오 파이프라인 실행
    run_video_pipeline(
        video_path=args.input,
        output_path=args.output,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        show_preview=not args.no_preview,
        calib_file=args.calib_file,
        calibrate_dir=args.calib_dir,
        use_undistort=args.undistort,
        checkerboard_size=(args.cb_cols, args.cb_rows),
        square_size=args.square_size
    )
