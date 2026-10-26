"""Lane perception package for dashcam_cv system.

듀얼 ROI 파이프라인 (cv1021.py 기반):
  - detect(): 좌/우 분리 듀얼 ROI + 4D Kalman 트래커
  - fit_lane_line(), scale_lines_to_original() 등

경량 단일 ROI 파이프라인 (lane detect.py 기반):
  - detect_scaled(): PROCESS_SCALE 다운스케일 + 단일 ROI + π/90 허프
  - make_roi_mask(), extract_fits(), scale_fit() 등
"""
from .lane_detector import (
    LaneDetector,
    get_dual_roi_geometry,
    shrink_bounding_box,
    crop_and_resize_roi,
    preprocess_roi,
    detect_crosswalk,
    scale_lines_to_original,
    fit_lane_line,
    make_roi_mask,
    extract_fits,
    scale_fit,
)
from .lane_tracker import LaneTracker, SingleLaneKalmanFilter
from .lane_change import LaneChangeDetector

__all__ = [
    # 클래스
    "LaneDetector",
    "LaneTracker",
    "SingleLaneKalmanFilter",
    "LaneChangeDetector",
    # 듀얼 ROI 파이프라인 함수 (cv1021.py 기반)
    "get_dual_roi_geometry",
    "shrink_bounding_box",
    "crop_and_resize_roi",
    "preprocess_roi",
    "detect_crosswalk",
    "scale_lines_to_original",
    "fit_lane_line",
    # 경량 단일 ROI 파이프라인 함수 (lane detect.py 기반)
    "make_roi_mask",
    "extract_fits",
    "scale_fit",
]
