"""Lightweight Object Detector module with YOLO backend and OpenCV fallback."""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ..common.data_models import ObjectBox

COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


class ObjectDetector:
    """Detects vehicles and pedestrians with interval-based lightweight execution."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        det_cfg = self.cfg.get("detector", {})

        self.enabled = det_cfg.get("enabled", True)
        self.backend = det_cfg.get("backend", "auto")
        self.model_name = det_cfg.get("model_name", "yolov8n.pt")
        self.conf_th = float(det_cfg.get("confidence_threshold", 0.35))
        self.person_conf_th = float(det_cfg.get("person_confidence_threshold", 0.22))
        self.iou_th = float(det_cfg.get("iou_threshold", 0.45))
        self.interval = max(1, int(det_cfg.get("detect_interval", 5)))
        self.target_classes = set(det_cfg.get("target_classes", [0, 1, 2, 3, 5, 7]))

        self.yolo_model = None
        self.cached_boxes: List[ObjectBox] = []
        self._init_backend()

    def _init_backend(self):
        """Initializes YOLO or OpenCV fallback backend."""
        if not self.enabled or self.backend == "none":
            print("[ObjectDetector] Detector disabled.")
            return

        if self.backend in ("auto", "ultralytics"):
            try:
                from ultralytics import YOLO
                print(f"[ObjectDetector] Loading Ultralytics YOLO model: {self.model_name}...")
                self.yolo_model = YOLO(self.model_name)
                print("[ObjectDetector] Ultralytics YOLO initialized successfully.")
                self.backend = "ultralytics"
                return
            except Exception as e:
                print(f"[ObjectDetector] Ultralytics initialization notice: {e}")
                print("[ObjectDetector] Switching to built-in OpenCV roadway contour detector fallback.")
                self.backend = "opencv_fallback"
        else:
            self.backend = "opencv_fallback"

    def detect(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[ObjectBox], bool]:
        """Runs detection if frame_idx matches interval; otherwise returns cached detections.

        Returns:
            boxes: list of ObjectBox instances
            updated: True if inference was executed on this frame, False if skipped
        """
        if not self.enabled or self.backend == "none":
            return [], False

        should_run = ((frame_idx - 1) % self.interval == 0) or (len(self.cached_boxes) == 0 and frame_idx == 1)

        if not should_run:
            return self.cached_boxes, False

        boxes: List[ObjectBox] = []

        # if self.backend == "ultralytics" and self.yolo_model is not None:
        #     boxes = self._detect_yolo(frame)
        # else:
        #     boxes = self._detect_opencv_fallback(frame)
        # 수정일자 2026/10/20 yolo 모델을 항상 사용하도록 변경
        boxes = self._detect_yolo(frame)

        self.cached_boxes = boxes
        return boxes, True

    def _detect_yolo(self, frame: np.ndarray) -> List[ObjectBox]:
        """Runs Ultralytics YOLO inference with adaptive sensitivity for pedestrians."""
        run_conf = min(self.conf_th, self.person_conf_th)
        results = self.yolo_model(
            frame,
            conf=run_conf,
            iou=self.iou_th,
            classes=list(self.target_classes),
            verbose=False
        )

        boxes: List[ObjectBox] = []
        if not results:
            return boxes

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return boxes

        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            cls_id = int(classes[i])
            score = float(confs[i])
            # Apply lower threshold for pedestrians on crosswalk/road
            min_score = self.person_conf_th if cls_id == 0 else self.conf_th
            if score < min_score:
                continue

            label = COCO_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            boxes.append(ObjectBox(
                track_id=-1,
                bbox=(x1, y1, x2, y2),
                class_id=cls_id,
                label=label,
                confidence=score,
                center=(cx, cy)
            ))
        return boxes

    def _detect_opencv_fallback(self, frame: np.ndarray) -> List[ObjectBox]:
        """Lightweight OpenCV contour-based vehicle/pedestrian detection in road region."""
        h, w = frame.shape[:2]
        # Restrict search to middle road horizon (where crosswalk and vehicles appear)
        road_y1 = int(h * 0.48)
        road_y2 = int(h * 0.88)
        road_roi = frame[road_y1:road_y2, :]

        gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[ObjectBox] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            aspect = rw / float(rh)

            x1 = rx
            y1 = road_y1 + ry
            x2 = rx + rw
            y2 = road_y1 + ry + rh

            # 1. Pedestrian on crosswalk/road: tall silhouette (aspect <= 0.65), area >= 300
            if aspect <= 0.65 and 300 <= area <= (w * h * 0.15):
                boxes.append(ObjectBox(
                    track_id=-1,
                    bbox=(x1, y1, x2, y2),
                    class_id=0,
                    label="person",
                    confidence=0.65,
                    center=((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                ))

            # 2. Vehicles: wider aspect (0.7 <= aspect <= 3.2), area >= 1200
            elif 0.7 <= aspect <= 3.2 and 1200 <= area <= (w * h * 0.25):
                boxes.append(ObjectBox(
                    track_id=-1,
                    bbox=(x1, y1, x2, y2),
                    class_id=2,
                    label="car",
                    confidence=0.70,
                    center=((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                ))

        return boxes
