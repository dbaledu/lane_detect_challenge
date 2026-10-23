"""Lightweight Object Detector module with YOLO backend, night enhancement, and hood crop."""
import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional

try:
    import torch
except ImportError:
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
    100: "crosswalk",
}

CROSSWALK_ID = 100


class _DropHalfWarning(logging.Filter):
    """구버전 폴백 시 남을 수 있는 'half' deprecated 경고 도배 방지."""

    def filter(self, record):
        return "'half' is deprecated" not in record.getMessage()


def pick_device() -> str:
    """RTX 5060(sm_120) 등 아키텍처 호환성 및 CUDA 가용성을 검사하여 디바이스 반환."""
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
    """단일 YOLO 모델을 감싸며 워밍업 및 FP16 추론을 수행하는 서브 검출기."""

    def __init__(self, weight: str, device: str, imgsz: int, classes: Optional[List[int]], conf: float, id_map: Optional[Dict[int, int]] = None):
        from ultralytics import YOLO

        self.weight = weight
        self.device = device
        self.imgsz = imgsz
        self.classes = classes
        self.conf = conf
        self.id_map = id_map or {}
        self.precision_kwargs = {}

        logging.getLogger("ultralytics").addFilter(_DropHalfWarning())
        self.model = YOLO(weight)
        self.names = self.model.names

    def warmup(self, img: np.ndarray, n: int = 2):
        """첫 번째 프레임 추론 시 GPU 메모리 할당 및 FP16 초기화 지연 방지."""
        if self.device == "cuda":
            try:
                self.precision_kwargs = {"half": True}
                self.predict(img)
            except Exception:
                self.precision_kwargs = {}
        for _ in range(n):
            self.predict(img)

    def predict(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """단일 이미지에 대해 YOLO 검출 수행."""
        r = self.model.predict(
            source=img,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            max_det=100,
            verbose=False,
            **(self.precision_kwargs or {}),
        )[0]

        b = r.boxes
        if b is None or len(b) == 0:
            return np.empty((0, 4), np.float32), np.empty(0, int), np.empty(0, np.float32)

        cls_ids = b.cls.cpu().numpy().astype(int)
        if self.id_map:
            cls_ids = np.array([self.id_map.get(int(c), int(c)) for c in cls_ids], dtype=int)

        return b.xyxy.cpu().numpy(), cls_ids, b.conf.cpu().numpy()


class ObjectDetector:
    """차량, 보행자, 신호등 및 횡단보도를 검출하는 고속 경량 객체 검출기."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        det_cfg = self.cfg.get("detector", {})

        self.enabled = det_cfg.get("enabled", True)
        self.backend = det_cfg.get("backend", "auto")
        self.model_name = det_cfg.get("model_name", "yolo26n.pt")
        self.conf_th = float(det_cfg.get("confidence_threshold", 0.35))
        self.person_conf_th = float(det_cfg.get("person_confidence_threshold", 0.22))
        self.iou_th = float(det_cfg.get("iou_threshold", 0.45))
        self.interval = max(1, int(det_cfg.get("detect_interval", 5)))
        self.target_classes = list(det_cfg.get("target_classes", [0, 1, 2, 3, 5, 7, 9]))

        # object_th_1022.py 고속/야간 최적화 설정
        self.hood_crop_ratio = float(det_cfg.get("hood_crop_ratio", 0.15))
        self.enable_enhance = det_cfg.get("enable_enhance", True)
        self.imgsz_gpu = int(det_cfg.get("imgsz_gpu", 640))
        self.imgsz_cpu = int(det_cfg.get("imgsz_cpu", 480))

        # 횡단보도 전용 모델 설정 (선택)
        self.enable_crosswalk = det_cfg.get("enable_crosswalk", False)
        self.crosswalk_weight = det_cfg.get("crosswalk_weight", "hf://saifyxpro/Revpass")
        self.crosswalk_class_name = det_cfg.get("crosswalk_class_name", "crosswalk").lower()
        self.crosswalk_conf = float(det_cfg.get("crosswalk_conf", 0.25))

        self.detectors: List[SubDetector] = []
        self.enhancer: Optional[NightEnhancer] = None
        self.device = "cpu"
        self.imgsz = self.imgsz_cpu
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
                else:
                    if torch is not None:
                        torch.set_num_threads(os.cpu_count() or 4)
                    self.imgsz = self.imgsz_cpu

                # 1. 메인 COCO 검출기 생성
                main_det = SubDetector(
                    weight=self.model_name,
                    device=self.device,
                    imgsz=self.imgsz,
                    classes=self.target_classes,
                    conf=min(self.conf_th, self.person_conf_th),
                )
                self.detectors.append(main_det)
                print(f"[ObjectDetector] 메인 YOLO 모델 로드 성공: {self.model_name} (device={self.device}, imgsz={self.imgsz})")

                # 2. 횡단보도 모델 (선택)
                if self.enable_crosswalk:
                    try:
                        cw_det = SubDetector(
                            weight=self.crosswalk_weight,
                            device=self.device,
                            imgsz=self.imgsz,
                            classes=None,
                            conf=self.crosswalk_conf,
                        )
                        cw_ids = [i for i, n in cw_det.names.items() if str(n).lower() == self.crosswalk_class_name]
                        if cw_ids:
                            cw_det.classes = cw_ids
                            cw_det.id_map = {i: CROSSWALK_ID for i in cw_ids}
                            self.detectors.append(cw_det)
                            print(f"[ObjectDetector] 횡단보도 모델 활성화: 클래스 {cw_ids} -> 내부 ID {CROSSWALK_ID}")
                    except Exception as e:
                        print(f"[ObjectDetector] 횡단보도 모델 로드 건너뜀: {e}")

                # 3. 야간 전처리기 생성
                if self.enable_enhance:
                    self.enhancer = NightEnhancer()

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

        if self.backend == "ultralytics" and self.detectors:
            boxes = self._detect_yolo(frame)
        else:
            boxes = self._detect_opencv_fallback(frame)

        self.cached_boxes = boxes
        return boxes, True

    def _detect_yolo(self, frame: np.ndarray) -> List[ObjectBox]:
        """본네트 제거, 해상도 축소, 야간 보정을 거친 고속 다중 YOLO 추론."""
        h, w = frame.shape[:2]

        # 1. 본네트 하단 영역 제외 (오검출 방지 및 연산 절약)
        crop_h = int(h * (1.0 - self.hood_crop_ratio))
        cropped = frame[:crop_h]

        # 2. 추론 해상도로 축소
        scale = self.imgsz / max(w, crop_h)
        small_w = max(1, round(w * scale))
        small_h = max(1, round(crop_h * scale))
        small = cv2.resize(cropped, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # 3. 첫 프레임 예열 (Warmup)
        if not self._warmed_up:
            warm_img = self.enhancer(small) if self.enhancer is not None else small
            for det in self.detectors:
                det.warmup(warm_img)
            self._warmed_up = True

        # 4. 야간 전처리
        if self.enhancer is not None:
            small = self.enhancer(small)

        # 5. 서브 검출기 추론 수행
        all_xyxy, all_ids, all_confs = [], [], []
        for det in self.detectors:
            xyxy, cls_ids, confs = det.predict(small)
            if len(cls_ids):
                all_xyxy.append(xyxy)
                all_ids.append(cls_ids)
                all_confs.append(confs)

        if not all_xyxy:
            return []

        cat_xyxy = np.concatenate(all_xyxy)
        cat_ids = np.concatenate(all_ids)
        cat_confs = np.concatenate(all_confs)

        # 6. 원본 좌표계로 스케일 복원
        inv_scale = 1.0 / scale
        boxes: List[ObjectBox] = []

        for i in range(len(cat_xyxy)):
            cls_id = int(cat_ids[i])
            score = float(cat_confs[i])

            # 보행자 저조도 민감도 분기
            min_score = self.person_conf_th if cls_id == 0 else self.conf_th
            if score < min_score:
                continue

            x1, y1, x2, y2 = cat_xyxy[i] * inv_scale
            x1 = max(0, min(w - 1, int(round(x1))))
            y1 = max(0, min(crop_h - 1, int(round(y1))))
            x2 = max(0, min(w - 1, int(round(x2))))
            y2 = max(0, min(crop_h - 1, int(round(y2))))

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
