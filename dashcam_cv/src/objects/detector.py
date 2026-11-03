"""Lightweight Object Detector module with YOLO backend, night enhancement, and 2-path traffic light verification (객체검출1027.py 기반)."""
import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional

try:
    import torch
except (ImportError, OSError, Exception):
    torch = None

from ..common.data_models import ObjectBox

COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}


class _DropHalfWarning(logging.Filter):
    """구버전 폴백 시 남을 수 있는 'half' deprecated 경고 도배 방지."""

    def filter(self, record):
        return "'half' is deprecated" not in record.getMessage()


def pick_device() -> str:
    """RTX 5060(sm_120) 등 아키텍처 호환성 및 CUDA 가용성을 검사하여 디바이스 반환 (객체검출1027.py)."""
    if torch is None or not torch.cuda.is_available():
        return "cpu"

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        print(f"[ObjectDetector] 현재 PyTorch 빌드는 {arch}를 지원하지 않아 CPU로 실행합니다.")
        return "cpu"

    try:
        _ = torch.zeros(1, device="cuda") + 1
    except Exception as e:
        print(f"[ObjectDetector] CUDA 테스트 실패({e}) -> CPU로 실행합니다.")
        return "cpu"

    print(f"[ObjectDetector] GPU 사용: {torch.cuda.get_device_name(0)} ({arch})")
    return "cuda"


class NightEnhancer:
    """야간 전처리: L 채널 1개만 처리하여 색상 왜곡 방지 및 연산 최소화 (감마 LUT + CLAHE)."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8), gamma: float = 0.8):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        inv = 1.0 / max(gamma, 1e-4)
        self.lut = (((np.arange(256) / 255.0) ** inv) * 255).astype(np.uint8)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(cv2.LUT(l, self.lut))
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


class SubDetector:
    """단일 YOLO 모델을 감싸며 워밍업 및 FP16 추론을 수행하는 서브 검출기 (일반 객체 및 신호등 공유)."""

    def __init__(
        self,
        weight: str,
        device: str,
        imgsz: int = 640,
        classes: Optional[List[int]] = None,
        conf: float = 0.22,
        id_map: Optional[Dict[int, int]] = None,
    ):
        from ultralytics import YOLO

        self.weight = weight
        self.device = device
        self.imgsz = imgsz
        self.classes = classes
        self.conf = conf
        self.id_map = id_map or {}
        self.half = False

        logging.getLogger("ultralytics").addFilter(_DropHalfWarning())
        self.model = YOLO(weight)
        self.names = self.model.names

    def warmup(
        self,
        warm_img: np.ndarray,
        imgsz: int = 640,
        tl_imgsz: int = 640,
        tl_roi: Optional[np.ndarray] = None,
    ):
        """첫 번째 프레임 추론 시 GPU 메모리 할당 및 FP16 초기화 지연 방지 (객체검출1027.py)."""
        if self.device == "cuda":
            self.half = True
            try:
                self.predict(warm_img, imgsz=imgsz, classes=[0, 1, 2, 3, 5, 7], conf=0.22)
            except Exception:
                self.half = False

        for _ in range(2):
            self.predict(warm_img, imgsz=imgsz, classes=[0, 1, 2, 3, 5, 7], conf=0.22)
            if tl_roi is not None and tl_roi.size > 0:
                self.predict(tl_roi, imgsz=tl_imgsz, classes=[9], conf=0.10)

    def predict(
        self,
        img: np.ndarray,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        conf: Optional[float] = None,
        iou: float = 0.45,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """단일 이미지에 대해 YOLO 검출 수행 (FP16 최적화 지원)."""
        use_imgsz = imgsz or self.imgsz
        use_classes = classes if classes is not None else self.classes
        use_conf = conf if conf is not None else self.conf

        r = self.model.predict(
            source=img,
            imgsz=use_imgsz,
            conf=use_conf,
            iou=iou,
            classes=use_classes,
            device=self.device,
            max_det=100,
            verbose=False,
            **({"half": True} if self.half else {}),
        )[0]

        b = r.boxes
        if b is None or len(b) == 0:
            return np.empty((0, 4), np.float32), np.empty(0, int), np.empty(0, np.float32)

        cls_ids = b.cls.cpu().numpy().astype(int)
        if self.id_map:
            cls_ids = np.array([self.id_map.get(int(c), int(c)) for c in cls_ids], dtype=int)

        return b.xyxy.cpu().numpy(), cls_ids, b.conf.cpu().numpy()


class ObjectDetector:
    """차량, 보행자, 신호등을 검출하는 고속 경량 2-Path 객체 검출기 (객체검출1027.py 기반)."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        det_cfg = self.cfg.get("detector", {})

        self.enabled = det_cfg.get("enabled", True)
        self.backend = det_cfg.get("backend", "auto")
        self.model_name = det_cfg.get("model_name", "yolo26n.pt")
        self.conf_th = float(det_cfg.get("confidence_threshold", 0.22))
        self.person_conf_th = float(det_cfg.get("person_confidence_threshold", 0.22))
        self.iou_th = float(det_cfg.get("iou_threshold", 0.45))
        self.interval = max(1, int(det_cfg.get("detect_interval", 5)))
        self.target_classes = list(det_cfg.get("target_classes", [0, 1, 2, 3, 5, 7, 9]))

        # 일반 객체 클래스와 신호등 클래스 분리 (객체검출1027.py)
        self.general_classes = [c for c in self.target_classes if c != 9]
        if not self.general_classes:
            self.general_classes = [0, 1, 2, 3, 5, 7]
        self.has_tl = 9 in self.target_classes
        self.tl_class = 9

        # 본네트 및 야간 전처리 (감마 LUT + CLAHE)
        self.hood_crop_ratio = float(det_cfg.get("hood_crop_ratio", 0.15))
        self.enable_enhance = det_cfg.get("enable_enhance", True)
        self.imgsz_gpu = int(det_cfg.get("imgsz_gpu", 640))
        self.imgsz_cpu = int(det_cfg.get("imgsz_cpu", 480))

        # 신호등 전용 설정 (객체검출1027.py)
        self.tl_roi_ratio = float(det_cfg.get("tl_roi_ratio", 0.6))
        self.tl_imgsz_gpu = int(det_cfg.get("tl_imgsz_gpu", 1280))
        self.tl_imgsz_cpu = int(det_cfg.get("tl_imgsz_cpu", 640))
        self.tl_conf = float(det_cfg.get("tl_conf", 0.10))
        self.tl_max_aspect = float(det_cfg.get("tl_max_aspect", 5.0))
        self.tl_min_sat = int(det_cfg.get("tl_min_sat", 80))
        self.tl_min_val = int(det_cfg.get("tl_min_val", 140))
        self.tl_min_lamp_ratio = float(det_cfg.get("tl_min_lamp_ratio", 0.03))

        self.sub_detector: Optional[SubDetector] = None
        self.detectors: List[SubDetector] = []
        self.enhancer: Optional[NightEnhancer] = None
        self.device = "cpu"
        self.imgsz = self.imgsz_cpu
        self.tl_imgsz = self.tl_imgsz_cpu
        self._warmed_up = False
        self.cached_boxes: List[ObjectBox] = []

        self._init_backend()

    def _init_backend(self):
        """YOLO 모델, 전처리 모듈 및 디바이스 초기화."""
        if not self.enabled or self.backend == "none":
            print("[ObjectDetector] Detector disabled.")
            return

        if self.backend in ("auto", "ultralytics"):
            try:
                self.device = pick_device()
                if self.device == "cuda" and torch is not None:
                    torch.backends.cudnn.benchmark = True
                    self.imgsz = self.imgsz_gpu
                    self.tl_imgsz = self.tl_imgsz_gpu
                else:
                    if torch is not None:
                        torch.set_num_threads(os.cpu_count() or 4)
                    self.imgsz = self.imgsz_cpu
                    self.tl_imgsz = self.tl_imgsz_cpu

                # 단일 YOLO 모델 인스턴스 (두 경로 공유)
                self.sub_detector = SubDetector(
                    weight=self.model_name,
                    device=self.device,
                )
                self.detectors = [self.sub_detector]
                print(
                    f"[ObjectDetector] 객체검출1027 듀얼경로 YOLO 로드 성공: {self.model_name} | "
                    f"device={self.device.upper()} | imgsz={self.imgsz} | tl_imgsz={self.tl_imgsz}"
                )

                # 2. 야간 전처리기 생성 (L채널 감마 + CLAHE)
                if self.enable_enhance:
                    self.enhancer = NightEnhancer(clip_limit=2.0, tile_grid_size=(8, 8), gamma=0.8)

                self.backend = "ultralytics"
                return
            except Exception as e:
                print(f"[ObjectDetector] Ultralytics 초기화 실패({e}) -> OpenCV 폴백 모드로 전환.")
                self.backend = "opencv_fallback"
        else:
            self.backend = "opencv_fallback"

    def detect(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[ObjectBox], bool]:
        """간격(interval)에 따라 객체 검출을 수행하고 ObjectBox 목록을 반환."""
        if not self.enabled or self.backend == "none":
            return [], False

        should_run = ((frame_idx - 1) % self.interval == 0) or (len(self.cached_boxes) == 0 and frame_idx == 1)
        if not should_run:
            return self.cached_boxes, False

        if self.backend == "ultralytics" and self.sub_detector is not None:
            boxes = self._detect_yolo(frame)
        else:
            boxes = self._detect_opencv_fallback(frame)

        self.cached_boxes = boxes
        return boxes, True

    def _verify_traffic_lights(
        self,
        roi: np.ndarray,
        xyxy: np.ndarray,
        ids: np.ndarray,
        confs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """건물 조명 오검출 제거: 박스 종횡비 + 박스 내 적/황/녹색 램프 픽셀 비율 검증 (객체검출1027.py)."""
        if len(ids) == 0:
            return np.empty((0, 4), np.float32), np.empty(0, int), np.empty(0, np.float32)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lit = (
            cv2.inRange(hsv, (0, self.tl_min_sat, self.tl_min_val), (12, 255, 255))        # 적색 1
            | cv2.inRange(hsv, (18, self.tl_min_sat, self.tl_min_val), (95, 255, 255))     # 황색~녹색
            | cv2.inRange(hsv, (165, self.tl_min_sat, self.tl_min_val), (180, 255, 255))   # 적색 2
        )

        keep = []
        for i, (x1, y1, x2, y2) in enumerate(xyxy.astype(int)):
            w_box, h_box = x2 - x1, y2 - y1
            if min(w_box, h_box) < 1 or max(w_box, h_box) > self.tl_max_aspect * min(w_box, h_box):
                continue
            y1_c = max(0, min(roi.shape[0], y1))
            y2_c = max(0, min(roi.shape[0], y2))
            x1_c = max(0, min(roi.shape[1], x1))
            x2_c = max(0, min(roi.shape[1], x2))
            if y2_c <= y1_c or x2_c <= x1_c:
                continue
            # 유효 램프 픽셀 수 검사
            lamp_pixels = cv2.countNonZero(lit[y1_c:y2_c, x1_c:x2_c])
            if lamp_pixels >= max(2, self.tl_min_lamp_ratio * w_box * h_box):
                keep.append(i)

        if not keep:
            return np.empty((0, 4), np.float32), np.empty(0, int), np.empty(0, np.float32)

        keep = np.array(keep, dtype=int)
        return xyxy[keep], ids[keep], confs[keep]

    def _detect_yolo(self, frame: np.ndarray) -> List[ObjectBox]:
        """객체검출1027.py 기반 듀얼 경로(일반 객체 + 신호등) 고속 추론 및 오검출 검증."""
        h, w = frame.shape[:2]

        if self.sub_detector is None:
            return []

        # 1. 본네트 하단 영역 제외 (오검출 방지 및 연산 절약)
        crop_h = int(h * (1.0 - self.hood_crop_ratio))
        crop = frame[:crop_h]

        # 일반 객체용 축소 스케일 및 크기 계산
        g_scale = self.imgsz / max(w, crop_h)
        g_size = (max(1, round(w * g_scale)), max(1, round(crop_h * g_scale)))
        g_k = np.array([w / g_size[0], crop_h / g_size[1]] * 2, np.float32)

        # 신호등용 상단 ROI 및 크기 계산
        roi_h = max(1, int(crop_h * self.tl_roi_ratio))
        t_scale = min(1.0, self.tl_imgsz / max(w, roi_h))
        t_size = (max(1, round(w * t_scale)), max(1, round(roi_h * t_scale)))
        t_k = np.array([w / t_size[0], roi_h / t_size[1]] * 2, np.float32)
        tl_target_sz = int(np.ceil(max(t_size) / 32) * 32)

        # 2. 첫 프레임 예열 (Warmup)
        if not self._warmed_up:
            small_warm = cv2.resize(crop, g_size, interpolation=cv2.INTER_AREA)
            if self.enhancer is not None:
                small_warm = self.enhancer(small_warm)
            roi_warm = cv2.resize(crop[:roi_h], t_size, interpolation=cv2.INTER_AREA)
            self.sub_detector.warmup(small_warm, imgsz=self.imgsz, tl_imgsz=tl_target_sz, tl_roi=roi_warm)
            self._warmed_up = True

        all_xyxy = []
        all_ids = []
        all_confs = []

        # 3. 경로 1) 일반 객체: 축소 + 야간 전처리 (CLAHE + 감마 LUT)
        small = cv2.resize(crop, g_size, interpolation=cv2.INTER_AREA)
        if self.enhancer is not None:
            small = self.enhancer(small)

        g_xyxy, g_ids, g_confs = self.sub_detector.predict(
            small,
            imgsz=self.imgsz,
            classes=self.general_classes,
            conf=self.conf_th,
            iou=self.iou_th
        )
        if len(g_ids):
            all_xyxy.append(g_xyxy * g_k)
            all_ids.append(g_ids)
            all_confs.append(g_confs)

        # 4. 경로 2) 신호등: 상단 ROI, 전처리 없이 고해상도 검출 후 검증
        if self.has_tl:
            roi = cv2.resize(crop[:roi_h], t_size, interpolation=cv2.INTER_AREA)
            t_xyxy, t_ids, t_confs = self.sub_detector.predict(
                roi,
                imgsz=tl_target_sz,
                classes=[self.tl_class],
                conf=self.tl_conf,
                iou=self.iou_th
            )
            t_xyxy, t_ids, t_confs = self._verify_traffic_lights(roi, t_xyxy, t_ids, t_confs)
            if len(t_ids):
                all_xyxy.append(t_xyxy * t_k)
                all_ids.append(t_ids)
                all_confs.append(t_confs)

        if not all_xyxy:
            return []

        cat_xyxy = np.concatenate(all_xyxy)
        cat_ids = np.concatenate(all_ids)
        cat_confs = np.concatenate(all_confs)

        # 5. ObjectBox 생성
        boxes: List[ObjectBox] = []
        for i in range(len(cat_xyxy)):
            cls_id = int(cat_ids[i])
            score = float(cat_confs[i])

            min_score = self.person_conf_th if cls_id == 0 else (self.tl_conf if cls_id == 9 else self.conf_th)
            if score < min_score:
                continue

            x1, y1, x2, y2 = cat_xyxy[i]
            x1 = max(0, min(w - 1, int(round(x1))))
            y1 = max(0, min(crop_h - 1, int(round(y1))))
            x2 = max(0, min(w - 1, int(round(x2))))
            y2 = max(0, min(crop_h - 1, int(round(y2))))

            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            label = COCO_CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            boxes.append(
                ObjectBox(
                    track_id=-1,
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    label=label,
                    confidence=score,
                    center=(cx, cy),
                )
            )

        return boxes

    def _detect_opencv_fallback(self, frame: np.ndarray) -> List[ObjectBox]:
        """OpenCV 기반 윤곽선 검출 폴백."""
        h, w = frame.shape[:2]
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

            if aspect <= 0.65 and 300 <= area <= (w * h * 0.15):
                boxes.append(
                    ObjectBox(
                        track_id=-1,
                        bbox=(x1, y1, x2, y2),
                        class_id=0,
                        label="person",
                        confidence=0.65,
                        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    )
                )
            elif 0.7 <= aspect <= 3.2 and 1200 <= area <= (w * h * 0.25):
                boxes.append(
                    ObjectBox(
                        track_id=-1,
                        bbox=(x1, y1, x2, y2),
                        class_id=2,
                        label="car",
                        confidence=0.70,
                        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    )
                )

        return boxes
