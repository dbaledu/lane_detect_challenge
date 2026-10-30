"""
[야간 주행 영상 실시간 객체 검출 - 경량/고속 버전, 횡단보도 지원]
- 모델 1: YOLO (COCO) -> 보행자, 자전거, 승용차, 오토바이, 버스, 트럭, 신호등
- 모델 2: 횡단보도(crosswalk) 클래스가 있는 모델 -> 횡단보도 전용 (ENABLE_CROSSWALK=True일 때)
- 조작: q / ESC = 종료, 스페이스바 = 일시정지/재생
- 최적화: 재생/검출 스레드 분리, 본네트 크롭, 축소 후 야간 전처리, FP16 반정밀도 연산, 프레임 스킵
- 시작 지연 방지: 검출 스레드 안에서 실제 첫 프레임으로 예열(Warmup)이 끝난 뒤에 재생을 시작한다.
"""

import os
import time
import queue
import logging
import threading
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

try:
    import torch
except (ImportError, OSError, Exception):
    torch = None

from ultralytics import YOLO

from ..common.data_models import ObjectBox

# OpenCV 내부 C/C++ 최적화(SIMD 등) 활성화하여 연산 속도 향상
cv2.setUseOptimized(True)

# ==========================================
# 설정값 (객체검출.py 기준 기본값)
# ==========================================
INPUT_VIDEO = "images/Test_Video_2.mp4"   # 입력 비디오 파일 경로
OUTPUT_VIDEO = "output_Test_Video_2.mp4" # 출력 비디오 저장 경로
SAVE_OUTPUT = False                       # True로 설정 시 추론 결과를 영상 파일로 저장

# --- 모델 1: 일반 객체 (COCO 데이터셋 기준) ---
MODEL_WEIGHT = "yolo26n.pt"
# COCO 데이터셋 클래스 ID: 0(보행자), 1(자전거), 2(승용차), 3(오토바이), 5(버스), 7(트럭), 9(신호등)
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 9]

# --- 모델 2: 횡단보도 전용 모델 ---
CROSSWALK_WEIGHT = "hf://saifyxpro/Revpass" # 허깅페이스 또는 로컬 (.pt) 파일 경로
CROSSWALK_CLASS_NAME = "crosswalk"          # 해당 모델에서 횡단보도를 지칭하는 클래스명
CROSSWALK_CONF = 0.25                       # 횡단보도 모델 신뢰도 임계값
ENABLE_CROSSWALK = False                    # True일 때만 횡단보도 모델 추가 로드
CROSSWALK_ID = 100                          # 일반 COCO ID와 충돌을 방지하기 위한 내부 고유 ID

IMGSZ_GPU = 640            # GPU 사용 시 입력 이미지 해상도
IMGSZ_CPU = 480            # CPU 사용 시 입력 이미지 해상도 (연산량 감소)
HOOD_CROP_RATIO = 0.15     # 하단 차량 본네트 영역 제거 비율 (15% 잘라냄)
ENABLE_ENHANCE = True      # 야간 영상 조도/명암 개선 전처리 적용 여부
DISPLAY_MAX_WIDTH = 1280   # 모니터 출력을 위한 최대 너비
SHOW_FPS = True            # 화면에 FPS 표기 여부

CONF = 0.22                # 검출 신뢰도(Confidence) 임계값
IOU = 0.45                 # NMS(Non-Maximum Suppression) IOU 임계값

# 클래스 ID별 표시 라벨 이름 및 BGR 색상 정의
CLASS_META = {
    0: ("Person", (0, 255, 255)),
    1: ("Bicycle", (0, 255, 255)),
    2: ("Car", (0, 255, 255)),
    3: ("Motorcycle", (0, 255, 255)),
    5: ("Bus", (0, 255, 255)),
    7: ("Truck", (0, 255, 255)),
    9: ("traffic light", (0, 255, 0)),
}

# 하위 호환 및 파이프라인 매핑용 소문자 라벨 딕셔너리
COCO_CLASS_NAMES = {cid: meta[0].lower() for cid, meta in CLASS_META.items()}
COCO_CLASS_NAMES[CROSSWALK_ID] = "crosswalk"

# 객체 감지가 안 되었을 때 반환할 빈 NumPy 배열 정의 (메모리 재사용)
_EMPTY = (np.empty((0, 4), np.float32), np.empty(0, int), np.empty(0, np.float32))


# ==========================================
# 1. 하드웨어 디바이스 검증 및 선택
# ==========================================
def pick_device() -> str:
    # 현재 시스템에서 CUDA GPU 사용이 가능한지 단계별로 점검하고
    # 사용 가능한 경우 'cuda', 불가능한 경우 'cpu'를 반환합니다.
    if torch is None:
        print("[경고] PyTorch가 임포트되지 않아 CPU로 실행합니다.")
        return "cpu"

    cuda_build = torch.version.cuda
    print(f"[PyTorch] 버전 {torch.__version__} | CUDA 빌드: {cuda_build or '없음(CPU 전용)'}")

    # 1) PyTorch의 CUDA 지원 여부 체크
    if not torch.cuda.is_available():
        if cuda_build is None:
            print("[경고] CPU 전용 PyTorch가 설치되어 있어 GPU를 쓸 수 없습니다.")
        else:
            print("[경고] CUDA를 사용할 수 없습니다. 드라이버/그래픽카드 인식을 확인하세요 (nvidia-smi).")
        _print_reinstall_help()
        return "cpu"

    # 2) GPU 아키텍처(Compute Capability)와 PyTorch 빌드 호환성 체크
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        print(f"[경고] 현재 PyTorch 빌드는 {arch}를 지원하지 않아 CPU로 실행합니다.")
        _print_reinstall_help()
        return "cpu"

    # 3) 실제 CUDA 메모리 할당 및 텐서 연산 테스트
    try:
        _ = torch.zeros(1, device="cuda") + 1
    except Exception as e:
        print(f"[경고] CUDA 테스트 실패({e}) -> CPU로 실행합니다.")
        _print_reinstall_help()
        return "cpu"

    print(f"[GPU] {torch.cuda.get_device_name(0)} ({arch}) 사용")
    return "cuda"


def _print_reinstall_help():
    # GPU 지원 버전 PyTorch 재설치 안내 메시지 출력
    print("       [해결 방법] 같은 파이썬 환경에서 순서대로 실행:")
    print("         python -m pip uninstall -y torch torchvision torchaudio")
    print("         python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")


# ==========================================
# 2. 야간 전처리 (L 채널 집중 처리)
# ==========================================
class NightEnhancer:
    # 야간 영상의 시가성을 높이기 위한 알고리즘.
    # BGR을 LAB 색상 공간으로 변환 후, 밝기(L 채널)에만 Gamma 보정 및 CLAHE를 적용하여
    # 색상 왜곡을 최소화하면서 야간 대조비를 극대화합니다.
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8), gamma: float = 0.8):
        # CLAHE (적응형 히스토그램 균일화): 적절한 대비 연산 수행
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        # 감마 보정용 Look-Up Table(LUT) 사전 계산 (속도 최적화)
        inv = 1.0 / max(gamma, 1e-4)
        self.lut = (((np.arange(256) / 255.0) ** inv) * 255).astype(np.uint8)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)  # BGR -> LAB 변환
        l, a, b = cv2.split(lab)                    # L(밝기), A/B(색상) 분리
        l = self.clahe.apply(cv2.LUT(l, self.lut))  # Brightness에 감마 적용 후 CLAHE 수행
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)  # BGR 재합성


# ==========================================
# 3. 객체 검출 모듈
# ==========================================
class _DropHalfWarning(logging.Filter):
    # Ultralytics 내부 'half' 옵션 사용 시 발생하는 경고 메시지 필터링
    def filter(self, record):
        return "'half' is deprecated" not in record.getMessage()


class Detector:
    # 단일 YOLO 모델을 래핑하여 추론 및 전/후처리를 담당하는 클래스.
    def __init__(
        self,
        weight: str,
        device: str,
        imgsz: int,
        classes: Optional[List[int]],
        conf: float,
        id_map: Optional[Dict[int, int]] = None,
    ):
        self.weight = weight
        self.device = device
        self.imgsz = imgsz
        self.classes = classes
        self.conf = conf
        self.id_map = id_map or {}
        self.precision_kwargs = None  # 추론 정밀도 (FP16/FP32 - Warmup 시 자동 설정)

        logging.getLogger("ultralytics").addFilter(_DropHalfWarning())
        self.model = YOLO(weight)
        self.names = self.model.names

    def warmup(self, img: np.ndarray, n: int = 3):
        # CUDA 엔진, 메모리 할당 및 cuDNN 벤치마크를 사전 실행(Warmup)하여
        # 실제 비디오 재생 초기의 렉(병목 현상)을 방지합니다.
        if self.precision_kwargs is None:
            self.precision_kwargs = {}
            if self.device == "cuda":
                try:
                    # GPU 사용 시 FP16(half precision) 자동 적용 시도하여 속도 2배 향상
                    self.precision_kwargs = {"half": True}
                    self.predict(img)
                except Exception:
                    self.precision_kwargs = {}
            print(
                f"[Detector] {self.weight} | device={self.device.upper()} | "
                f"precision={self.precision_kwargs or 'FP32'} | imgsz={self.imgsz} | classes={self.classes}"
            )
        out = _EMPTY
        for _ in range(n):
            out = self.predict(img)
        return out

    def predict(
        self,
        img: np.ndarray,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        conf: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 이미지를 입력받아 바운딩 박스 좌표, 클래스 ID, 신호등/객체 점수를 반환합니다.
        run_imgsz = imgsz if imgsz is not None else self.imgsz
        run_classes = classes if classes is not None else self.classes
        run_conf = conf if conf is not None else self.conf

        r = self.model.predict(
            source=img,
            imgsz=run_imgsz,
            conf=run_conf,
            iou=IOU,
            classes=run_classes,
            device=self.device,
            max_det=100,
            verbose=False,
            **(self.precision_kwargs or {}),
        )[0]

        b = r.boxes
        if b is None or len(b) == 0:
            return _EMPTY

        # PyTorch Tensor -> CPU NumPy 배열로 변환
        cls_ids = b.cls.cpu().numpy().astype(int)

        # 내부 통합 관리를 위해 클래스 ID 매핑이 설정된 경우 적용
        if self.id_map:
            cls_ids = np.array([self.id_map.get(int(c), int(c)) for c in cls_ids], dtype=int)

        return b.xyxy.cpu().numpy(), cls_ids, b.conf.cpu().numpy()


def verify_traffic_lights(
    roi: np.ndarray,
    xyxy: np.ndarray,
    ids: np.ndarray,
    confs: np.ndarray,
    max_aspect: float = 5.0,
    min_sat: int = 80,
    min_val: int = 140,
    min_lamp_ratio: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """건물 조명 오검출 제거: 박스 모양 + 박스 안에 적/황/녹색 램프 픽셀 비율 검증 (objects_detector_th.py 호환)."""
    if not len(ids):
        return np.empty((0, 4), np.float32), np.empty(0, int), np.empty(0, np.float32)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lit = (
        cv2.inRange(hsv, (0, min_sat, min_val), (12, 255, 255))
        | cv2.inRange(hsv, (18, min_sat, min_val), (95, 255, 255))
        | cv2.inRange(hsv, (165, min_sat, min_val), (180, 255, 255))
    )

    keep = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy.astype(int)):
        w, h = x2 - x1, y2 - y1
        # 검증 1) 모양: 너무 가늘고 긴 박스는 신호등이 아니라 판단
        if min(w, h) < 1 or max(w, h) > max_aspect * min(w, h):
            continue
        # 검증 2) 색: 박스 안의 램프색 픽셀 수가 (최소 2개, 박스 면적의 일정 비율) 이상이어야 통과
        if cv2.countNonZero(lit[y1:y2, x1:x2]) >= max(2, min_lamp_ratio * w * h):
            keep.append(i)

    keep = np.array(keep, dtype=int)
    if len(keep) == 0:
        return np.empty((0, 4), np.float32), np.empty(0, int), np.empty(0, np.float32)
    return xyxy[keep], ids[keep], confs[keep]


# 구버전 호환용 별칭
SubDetector = Detector


def build_detectors(
    device: str,
    model_weight: str = MODEL_WEIGHT,
    conf: float = CONF,
    target_classes: Optional[List[int]] = None,
    enable_crosswalk: bool = ENABLE_CROSSWALK,
    crosswalk_weight: str = CROSSWALK_WEIGHT,
    crosswalk_class_name: str = CROSSWALK_CLASS_NAME,
    crosswalk_conf: float = CROSSWALK_CONF,
    imgsz_gpu: int = IMGSZ_GPU,
    imgsz_cpu: int = IMGSZ_CPU,
) -> Tuple[List[Detector], int]:
    # 설정값에 따라 COCO 검출기 및 횡단보도 검출기 객체를 생성합니다.
    imgsz = imgsz_gpu if device == "cuda" else imgsz_cpu
    if device == "cuda" and torch is not None:
        torch.backends.cudnn.benchmark = True  # GPU 커널 최적화
    elif torch is not None:
        torch.set_num_threads(os.cpu_count() or 4)  # CPU 코어 병렬 연산 최적화

    classes = target_classes if target_classes is not None else TARGET_CLASSES
    detectors = [Detector(model_weight, device, imgsz, classes, conf)]

    # 횡단보도 모델 로드 설정이 True인 경우 추가
    if enable_crosswalk:
        try:
            cw = Detector(crosswalk_weight, device, imgsz, None, crosswalk_conf)
            cw_ids = [i for i, n in cw.names.items() if str(n).lower() == crosswalk_class_name.lower()]
            if not cw_ids:
                print(f"[경고] 횡단보도 모델에 '{crosswalk_class_name}' 클래스가 없습니다.")
            else:
                cw.classes = cw_ids
                cw.id_map = {i: CROSSWALK_ID for i in cw_ids}
                detectors.append(cw)
        except Exception as e:
            print(f"[경고] 횡단보도 모델 로드 실패({e!r}) -> 일반 검출만 실행합니다.")

    return detectors, imgsz


class AsyncDetector:
    # 비동기 검출 스레드 (재생과 검출 분리, 듀얼 경로 지원)
    def __init__(
        self,
        detectors: List[Detector],
        enhancer: Optional[NightEnhancer],
        W: int,
        crop_h: int,
        imgsz: int,
        tl_imgsz: int = 1280,
        warm_crop: Optional[np.ndarray] = None,
        tl_cfg: Optional[Dict[str, Any]] = None,
        # 하위 호환용 인자
        inv_scale: Optional[float] = None,
        warm_img: Optional[np.ndarray] = None,
    ):
        self.dets = detectors
        self.enh = enhancer
        self.imgsz = imgsz
        self.tl_imgsz = tl_imgsz
        self.W = W
        self.crop_h = crop_h
        self.tl_cfg = tl_cfg or {}

        self.enable_traffic_light = self.tl_cfg.get("enabled", True)
        self.tl_class_id = int(self.tl_cfg.get("class_id", 9))
        self.tl_roi_ratio = float(self.tl_cfg.get("roi_ratio", 0.6))
        self.tl_conf = float(self.tl_cfg.get("confidence_threshold", 0.10))
        self.tl_max_aspect = float(self.tl_cfg.get("max_aspect_ratio", 5.0))
        self.tl_min_sat = int(self.tl_cfg.get("min_saturation", 80))
        self.tl_min_val = int(self.tl_cfg.get("min_value", 140))
        self.tl_min_lamp_ratio = float(self.tl_cfg.get("min_lamp_ratio", 0.03))

        # 일반 객체 경로 크기
        g = imgsz / max(W, crop_h)
        self.g_size = (max(1, round(W * g)), max(1, round(crop_h * g)))
        self.g_k = np.array([W / self.g_size[0], crop_h / self.g_size[1]] * 2, np.float32)

        # 신호등 경로 크기
        self.roi_h = max(1, int(crop_h * self.tl_roi_ratio))
        t = min(1.0, tl_imgsz / max(W, self.roi_h))
        self.t_size = (max(1, round(W * t)), max(1, round(self.roi_h * t)))
        self.t_k = np.array([W / self.t_size[0], self.roi_h / self.t_size[1]] * 2, np.float32)
        self.model_tl_imgsz = int(np.ceil(max(self.t_size) / 32) * 32)

        # 예열 이미지 (warm_crop 우선, 없으면 warm_img)
        self._warm = warm_crop if warm_crop is not None else warm_img

        self._cv = threading.Condition()
        self._pending = None
        self._result = _EMPTY
        self._stop = False
        self.infer_fps = 0.0
        self.error = None
        self.ready = threading.Event()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, crop: np.ndarray):
        with self._cv:
            self._pending = crop
            self._cv.notify()

    def latest(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._cv:
            return self._result

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=2.0)

    def _infer(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 경로 1: 일반 객체 (축소 + 야간 전처리)
        small = cv2.resize(crop, self.g_size, interpolation=cv2.INTER_AREA)
        if self.enh is not None:
            small = self.enh(small)

        all_xyxy, all_ids, all_confs = [], [], []
        main_det = self.dets[0]

        # 메인 일반 객체 추론 (신호등 분리)
        g_classes = [c for c in (main_det.classes or TARGET_CLASSES) if c != self.tl_class_id] if self.enable_traffic_light else main_det.classes
        g_xyxy, g_ids, g_confs = main_det.predict(small, imgsz=self.imgsz, classes=g_classes)
        if len(g_ids):
            all_xyxy.append(g_xyxy * self.g_k)
            all_ids.append(g_ids)
            all_confs.append(g_confs)

        # 추가 검출기(횡단보도 등)
        for extra_det in self.dets[1:]:
            e_xyxy, e_ids, e_confs = extra_det.predict(small, imgsz=self.imgsz)
            if len(e_ids):
                all_xyxy.append(e_xyxy * self.g_k)
                all_ids.append(e_ids)
                all_confs.append(e_confs)

        # 경로 2: 신호등 전용 경로 (상단 ROI 고해상도 + 램프 색상/형상 검증)
        if self.enable_traffic_light:
            roi = cv2.resize(crop[:self.roi_h], self.t_size, interpolation=cv2.INTER_AREA)
            t_xyxy, t_ids, t_confs = main_det.predict(roi, imgsz=self.model_tl_imgsz, classes=[self.tl_class_id], conf=self.tl_conf)
            t_xyxy, t_ids, t_confs = verify_traffic_lights(
                roi, t_xyxy, t_ids, t_confs,
                max_aspect=self.tl_max_aspect,
                min_sat=self.tl_min_sat,
                min_val=self.tl_min_val,
                min_lamp_ratio=self.tl_min_lamp_ratio,
            )
            if len(t_ids):
                all_xyxy.append(t_xyxy * self.t_k)
                all_ids.append(t_ids)
                all_confs.append(t_confs)

        if not all_xyxy:
            return _EMPTY

        return np.concatenate(all_xyxy), np.concatenate(all_ids), np.concatenate(all_confs)

    def _warmup(self):
        if self.dets and self.dets[0].device == "cuda":
            self.dets[0].precision_kwargs = {"half": True}
            try:
                self._infer(self._warm)
            except Exception:
                self.dets[0].precision_kwargs = {}
        res = _EMPTY
        for _ in range(3):
            res = self._infer(self._warm)
        return res

    def _run(self):
        try:
            res = self._warmup()
            with self._cv:
                self._result = res
            self.ready.set()

            while True:
                with self._cv:
                    while self._pending is None and not self._stop:
                        self._cv.wait()
                    if self._stop:
                        return
                    img, self._pending = self._pending, None

                t = time.perf_counter()
                res = self._infer(img)
                dt = max(time.perf_counter() - t, 1e-6)

                with self._cv:
                    self._result = res
                    self.infer_fps = (1.0 / dt) if self.infer_fps == 0 else 0.8 * self.infer_fps + 0.2 / dt
        except Exception as e:
            self.error = e
            self.ready.set()


def draw_boxes(frame: np.ndarray, xyxy: np.ndarray, cls_ids: np.ndarray, scores: np.ndarray):
    # 영상 프레임 상에 객체 검출 바운딩 박스를 그리는 함수
    for (x1, y1, x2, y2), c, s in zip(xyxy.astype(int), cls_ids, scores):
        label, color = CLASS_META.get(int(c), ("Object", (200, 200, 200)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if c in (0, CROSSWALK_ID) else 1)


# ==========================================
# 4. dashcam_cv 파이프라인 통합 클래스
# ==========================================
class ObjectDetector:
    """objects_detector_th.py 기반의 듀얼 경로(일반 객체 + 신호등 ROI 고해상도 검증) 경량 객체 검출기."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        det_cfg = self.cfg.get("detector", {})

        self.enabled = det_cfg.get("enabled", True)
        self.backend = det_cfg.get("backend", "auto")
        self.model_name = det_cfg.get("model_name", MODEL_WEIGHT)
        self.conf = float(det_cfg.get("confidence_threshold", CONF))
        self.person_conf = float(det_cfg.get("person_confidence_threshold", self.conf))
        self.iou = float(det_cfg.get("iou_threshold", IOU))
        self.interval = max(1, int(det_cfg.get("detect_interval", 5)))
        raw_target_classes = list(det_cfg.get("target_classes", TARGET_CLASSES))

        # 야간 최적화 및 크롭 설정
        self.hood_crop_ratio = float(det_cfg.get("hood_crop_ratio", HOOD_CROP_RATIO))
        self.enable_enhance = det_cfg.get("enable_enhance", ENABLE_ENHANCE)
        self.imgsz_gpu = int(det_cfg.get("imgsz_gpu", IMGSZ_GPU))
        self.imgsz_cpu = int(det_cfg.get("imgsz_cpu", IMGSZ_CPU))
        self.ne_cfg = det_cfg.get("night_enhancer", {})

        # 신호등 전용 듀얼 경로 설정 (objects_detector_th.py 호환)
        tl_cfg = det_cfg.get("traffic_light", {})
        self.enable_traffic_light = tl_cfg.get("enabled", True)
        self.tl_class_id = int(tl_cfg.get("class_id", 9))
        self.tl_roi_ratio = float(tl_cfg.get("roi_ratio", 0.6))
        self.tl_imgsz_gpu = int(tl_cfg.get("imgsz_gpu", 1280))
        self.tl_imgsz_cpu = int(tl_cfg.get("imgsz_cpu", 640))
        self.tl_conf = float(tl_cfg.get("confidence_threshold", 0.10))
        self.tl_max_aspect = float(tl_cfg.get("max_aspect_ratio", 5.0))
        self.tl_min_sat = int(tl_cfg.get("min_saturation", 80))
        self.tl_min_val = int(tl_cfg.get("min_value", 140))
        self.tl_min_lamp_ratio = float(tl_cfg.get("min_lamp_ratio", 0.03))

        # 신호등 전용 경로 활성화 시 일반 객체 대상에서 신호등 클래스 분리
        if self.enable_traffic_light and self.tl_class_id in raw_target_classes:
            self.general_target_classes = [c for c in raw_target_classes if c != self.tl_class_id]
        else:
            self.general_target_classes = raw_target_classes

        # 횡단보도 전용 모델 설정
        self.enable_crosswalk = det_cfg.get("enable_crosswalk", ENABLE_CROSSWALK)
        self.crosswalk_weight = det_cfg.get("crosswalk_weight", CROSSWALK_WEIGHT)
        self.crosswalk_class_name = det_cfg.get("crosswalk_class_name", CROSSWALK_CLASS_NAME).lower()
        self.crosswalk_conf = float(det_cfg.get("crosswalk_conf", CROSSWALK_CONF))

        self.detectors: List[Detector] = []
        self.enhancer: Optional[NightEnhancer] = None
        self.device = "cpu"
        self.imgsz = self.imgsz_cpu
        self.tl_imgsz = self.tl_imgsz_cpu
        self._warmed_up = False
        self.cached_boxes: List[ObjectBox] = []

        self._init_backend()

    def _init_backend(self):
        """디바이스 점검 및 모델 로드."""
        if not self.enabled or self.backend == "none":
            print("[ObjectDetector] Detector disabled.")
            return

        self.device = pick_device()
        self.imgsz = self.imgsz_gpu if self.device == "cuda" else self.imgsz_cpu
        self.tl_imgsz = self.tl_imgsz_gpu if self.device == "cuda" else self.tl_imgsz_cpu

        if self.device == "cuda" and torch is not None:
            torch.backends.cudnn.benchmark = True
        elif torch is not None:
            torch.set_num_threads(os.cpu_count() or 4)

        # 1. 메인 COCO 검출기 (일반 객체 클래스 대상)
        main_det = Detector(
            self.model_name,
            self.device,
            self.imgsz,
            self.general_target_classes,
            min(self.conf, self.person_conf),
        )
        self.detectors = [main_det]

        # 2. 횡단보도 모델 (선택)
        if self.enable_crosswalk:
            try:
                cw = Detector(self.crosswalk_weight, self.device, self.imgsz, None, self.crosswalk_conf)
                cw_ids = [i for i, n in cw.names.items() if str(n).lower() == self.crosswalk_class_name]
                if cw_ids:
                    cw.classes = cw_ids
                    cw.id_map = {i: CROSSWALK_ID for i in cw_ids}
                    self.detectors.append(cw)
            except Exception as e:
                print(f"[경고] 횡단보도 모델 로드 실패({e!r}) -> 일반 검출만 실행합니다.")

        # 3. 야간 전처리기
        if self.enable_enhance:
            self.enhancer = NightEnhancer(
                clip_limit=float(self.ne_cfg.get("clip_limit", 2.0)),
                tile_grid_size=tuple(self.ne_cfg.get("tile_grid_size", [8, 8])),
                gamma=float(self.ne_cfg.get("gamma", 0.8)),
            )

        self.backend = "ultralytics"

    def detect(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[ObjectBox], bool]:
        """간격(interval)에 따라 객체 검출을 수행하고 ObjectBox 목록을 반환."""
        if not self.enabled or not self.detectors:
            return [], False

        should_run = ((frame_idx - 1) % self.interval == 0) or (len(self.cached_boxes) == 0 and frame_idx == 1)
        if not should_run:
            return self.cached_boxes, False

        boxes = self._detect_yolo(frame)
        self.cached_boxes = boxes
        return boxes, True

    def _detect_yolo(self, frame: np.ndarray) -> List[ObjectBox]:
        """본네트 제거 + 일반 객체(전처리+축소) & 신호등(상단 고해상도 ROI+램프검증) 듀얼 추론."""
        h, w = frame.shape[:2]

        # 1. 본네트 하단 영역 제외
        crop_h = int(h * (1.0 - self.hood_crop_ratio))
        crop = frame[:crop_h]

        # 일반 객체 경로 크기 계산
        g_scale = self.imgsz / max(w, crop_h)
        g_size = (max(1, round(w * g_scale)), max(1, round(crop_h * g_scale)))
        g_k = np.array([w / g_size[0], crop_h / g_size[1]] * 2, np.float32)

        # 2. 첫 프레임 예열 (Warmup)
        if not self._warmed_up:
            warm_small = cv2.resize(crop, g_size, interpolation=cv2.INTER_AREA)
            if self.enhancer is not None:
                warm_small = self.enhancer(warm_small)
            for det in self.detectors:
                det.warmup(warm_small)
            self._warmed_up = True

        # ── 경로 1: 일반 객체 (축소 + 야간 전처리 + YOLO) ──
        small = cv2.resize(crop, g_size, interpolation=cv2.INTER_AREA)
        if self.enhancer is not None:
            small = self.enhancer(small)

        all_xyxy, all_ids, all_confs = [], [], []
        main_det = self.detectors[0]

        g_xyxy, g_ids, g_confs = main_det.predict(
            small,
            imgsz=self.imgsz,
            classes=self.general_target_classes,
            conf=min(self.conf, self.person_conf),
        )
        if len(g_ids):
            all_xyxy.append(g_xyxy * g_k)
            all_ids.append(g_ids)
            all_confs.append(g_confs)

        # 추가 검출기(횡단보도 등)
        for extra_det in self.detectors[1:]:
            e_xyxy, e_ids, e_confs = extra_det.predict(small, imgsz=self.imgsz)
            if len(e_ids):
                all_xyxy.append(e_xyxy * g_k)
                all_ids.append(e_ids)
                all_confs.append(e_confs)

        # ── 경로 2: 신호등 전용 경로 (상단 ROI 고해상도 + 램프 색/형상 검증) ──
        if self.enable_traffic_light:
            roi_h = max(1, int(crop_h * self.tl_roi_ratio))
            t_scale = min(1.0, self.tl_imgsz / max(w, roi_h))
            t_size = (max(1, round(w * t_scale)), max(1, round(roi_h * t_scale)))
            t_k = np.array([w / t_size[0], roi_h / t_size[1]] * 2, np.float32)
            model_tl_imgsz = int(np.ceil(max(t_size) / 32) * 32)

            roi_small = cv2.resize(crop[:roi_h], t_size, interpolation=cv2.INTER_AREA)
            t_xyxy, t_ids, t_confs = main_det.predict(
                roi_small,
                imgsz=model_tl_imgsz,
                classes=[self.tl_class_id],
                conf=self.tl_conf,
            )
            t_xyxy, t_ids, t_confs = verify_traffic_lights(
                roi_small,
                t_xyxy,
                t_ids,
                t_confs,
                max_aspect=self.tl_max_aspect,
                min_sat=self.tl_min_sat,
                min_val=self.tl_min_val,
                min_lamp_ratio=self.tl_min_lamp_ratio,
            )
            if len(t_ids):
                all_xyxy.append(t_xyxy * t_k)
                all_ids.append(t_ids)
                all_confs.append(t_confs)

        if not all_xyxy:
            return []

        cat_xyxy = np.concatenate(all_xyxy)
        cat_ids = np.concatenate(all_ids)
        cat_confs = np.concatenate(all_confs)

        boxes: List[ObjectBox] = []
        for i in range(len(cat_xyxy)):
            cls_id = int(cat_ids[i])
            score = float(cat_confs[i])

            if cls_id == self.tl_class_id and self.enable_traffic_light:
                min_score = self.tl_conf
            elif cls_id == 0:
                min_score = self.person_conf
            else:
                min_score = self.conf

            if score < min_score:
                continue

            x1, y1, x2, y2 = cat_xyxy[i]
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
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    class_id=cls_id,
                    label=label,
                    confidence=score,
                    center=(cx, cy),
                )
            )

        return boxes


# ==========================================
# 5. 비디오 재생 및 메인 제어 루프 (단독 실행용)
# ==========================================
_END = object()
WINDOW = "Night Drive Detection"


def run(input_path: str = INPUT_VIDEO, output_path: str = OUTPUT_VIDEO):
    # 비디오 파일 존재 여부 확인
    if not os.path.exists(input_path):
        alt = os.path.join("images", input_path)
        if os.path.exists(alt):
            input_path = alt
        else:
            print(f"[오류] 입력 파일을 찾을 수 없습니다: {input_path}")
            return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[오류] 비디오를 열 수 없습니다: {input_path}")
        return

    # 영상 속성 추출
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_dt = 1.0 / fps

    # 동영상 저장 객체 설정
    writer = None
    if SAVE_OUTPUT:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # 하드웨어 디바이스 및 검출기 초기화
    device = pick_device()
    detectors, imgsz = build_detectors(device)
    enhancer = NightEnhancer() if ENABLE_ENHANCE else None

    # --- 연산 최적화를 위한 자르기 및 축소 스케일링 설정 ---
    crop_h = int(H * (1.0 - HOOD_CROP_RATIO))  # 차량 본네트 부분 자르기
    scale = imgsz / max(W, crop_h)              # 모델 입력 해상도에 맞춘 비율 계산
    small_w, small_h = max(1, round(W * scale)), max(1, round(crop_h * scale))

    disp_scale = min(1.0, DISPLAY_MAX_WIDTH / W)
    disp_size = (max(1, int(W * disp_scale)), max(1, int(H * disp_scale)))
    font_scale = max(0.5, W / 1800)

    # 비디오의 첫 프레임 읽기 (Warmup에 사용)
    ok, first_frame = cap.read()
    if not ok:
        print("[오류] 첫 프레임을 읽을 수 없습니다.")
        cap.release()
        return
    warm_small = cv2.resize(first_frame[:crop_h], (small_w, small_h), interpolation=cv2.INTER_AREA)

    print("[정보] 모델 예열 중... (끝나면 재생이 시작됩니다)")

    # 비동기 검출 스레드 생성 및 예열 시작
    worker = AsyncDetector(
        detectors,
        enhancer,
        W,
        crop_h,
        imgsz,
        tl_imgsz=1280 if device == "cuda" else 640,
        warm_crop=first_frame[:crop_h].copy(),
    )
    worker.ready.wait()  # 예열 완료까지 메인 스레드 대기

    if worker.error is not None:
        print(f"[오류] 모델 예열 실패: {worker.error!r}")
        worker.stop()
        cap.release()
        return

    # 영상 저장을 위한 별도 인코딩 스레드 구성
    out_q, t_write = None, None
    if writer is not None:
        out_q = queue.Queue(maxsize=32)

        def write_worker():
            while True:
                item = out_q.get()
                if item is _END:
                    break
                writer.write(item)

        t_write = threading.Thread(target=write_worker, daemon=True)
        t_write.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    idx = 0
    shown = 0
    disp_fps = 0.0
    t_prev = None
    t0 = time.perf_counter()  # 실제 비디오 재생 타이머 시작

    try:
        while True:
            if worker.error is not None:
                break

            # 첫 프레임은 미리 읽어둔 프레임을 쓰고, 이후에는 카메라/파일에서 읽음
            if first_frame is not None:
                ok, frame, first_frame = True, first_frame, None
            else:
                ok, frame = cap.read()
            if not ok:
                break
            idx += 1

            # 검출 스레드로 본네트 잘라낸 프레임 전송 (축소 및 듀얼 경로는 검출 스레드에서 수행)
            worker.submit(frame[:crop_h].copy())

            # 3) 가장 최근 완성된 검출 결과를 가져와 그려주기
            xyxy, cls_ids, scores = worker.latest()
            draw_boxes(frame, xyxy, cls_ids, scores)

            # 4) 재생 FPS 측정 및 화면 표시
            now = time.perf_counter()
            if t_prev is not None:
                inst = 1.0 / max(now - t_prev, 1e-6)
                disp_fps = inst if disp_fps == 0 else 0.9 * disp_fps + 0.1 * inst
            t_prev = now

            if SHOW_FPS:
                cv2.putText(
                    frame,
                    f"View {disp_fps:.0f} fps | Detect {worker.infer_fps:.0f} fps",
                    (12, int(28 * font_scale) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * font_scale,
                    (0, 255, 255),
                    max(1, int(2 * font_scale)),
                    cv2.LINE_AA,
                )

            # 파일 저장 큐에 프레임 전달
            if out_q is not None:
                out_q.put(frame)

            # 화면 크기에 맞춰 리사이즈 후 렌더링
            show = cv2.resize(frame, disp_size, interpolation=cv2.INTER_AREA) if disp_scale < 1.0 else frame
            cv2.imshow(WINDOW, show)
            shown += 1

            # 5) 실시간 동기화 제어: 목표 시간 타겟에 맞춰 동적으로 waitKey 조정
            target = t0 + idx * frame_dt
            wait_ms = max(1, int((target - time.perf_counter()) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF

            # 종료 키 체크 (q 또는 ESC)
            if key in (ord("q"), 27):
                print("\n[사용자 중단]")
                break

            # 창 닫기 버튼 클릭 처리
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            # 스페이스바 키 입력 시 일시정지
            if key == ord(" "):
                while True:
                    k = cv2.waitKey(50) & 0xFF
                    if k in (ord(" "), ord("q"), 27):
                        break
                if k in (ord("q"), 27):
                    break
                t0 = time.perf_counter() - idx * frame_dt  # 재생 시계 동기화 재설정
                continue

            # 6) 프레임 스킵 최적화: 검출/재생 연산이 비디오 타임라인보다 크게 뒤쳐지면 프레임 디코딩 스킵
            if writer is None:
                behind = time.perf_counter() - (t0 + idx * frame_dt)
                skip = int(behind / frame_dt)
                if skip > 0:
                    for _ in range(skip):
                        if not cap.grab():  # grab()을 사용해 디코딩 연산을 거치지 않고 프레임을 넘김
                            break
                        idx += 1

    finally:
        # 종료 처리 및 메모리/자원 해제
        worker.stop()
        if out_q is not None:
            out_q.put(_END)
            t_write.join()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    print(f"[완료] 표시 {shown}프레임 / 영상 위치 {idx}프레임 | {elapsed:.1f}초")


if __name__ == "__main__":
    run(INPUT_VIDEO, OUTPUT_VIDEO)
