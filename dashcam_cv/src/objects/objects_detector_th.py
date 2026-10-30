"""
[야간 주행 영상 실시간 객체 검출 - 경량/고속 버전]
- 일반 객체(보행자, 자전거, 승용차, 오토바이, 버스, 트럭): 본네트 크롭 + 축소 + 야간 전처리 후 검출
- 신호등: 화면 상단 영역을 고해상도 원본(전처리 없음)으로 따로 검출하고,
          램프 색/비율 검증으로 건물 조명 오검출을 걸러낸다.
- 조작: q / ESC = 종료, 스페이스바 = 일시정지/재생
- 최적화: 재생/검출 스레드 분리, FP16, 프레임 스킵, 첫 프레임 예열 후 재생 시작

────────────────────────────────────────────────────────────
[전체 파이프라인]
────────────────────────────────────────────────────────────

■ 시작 단계 (run 함수 앞부분, 한 번만 실행)
  영상 열기 → GPU/CPU 선택 → YOLO 모델 로드 → 첫 프레임으로 모델 예열(warmup) → 재생 시작

■ 프레임 처리 단계 (프레임마다 반복)

  [메인 스레드]  ① 영상에서 프레임 1장 읽기 (cap.read)
        │
        ▼
  [메인 스레드]  ② 본네트 영역을 잘라낸 복사본을 검출 스레드에 넘김 (worker.submit)
        │          - 검출이 밀려 있으면 이전에 넘긴 프레임은 버리고 최신 프레임으로 덮어씀
        ▼
  [검출 스레드]  ③ '가장 최신' 프레임 1장을 꺼냄
        │
        ├─▶ ④-A 일반 객체 경로
        │        축소(imgsz) → 야간 전처리(감마 + CLAHE) → YOLO
        │        → 보행자 / 자전거 / 승용차 / 오토바이 / 버스 / 트럭
        │
        └─▶ ④-B 신호등 경로
                 화면 상단 ROI를 고해상도로 축소 (전처리 없음) → YOLO (낮은 신뢰도로 후보를 넓게 수집)
                 → 박스 모양 + 램프 색(적/황/녹) 검증으로 건물 조명 오검출 제거
        │
        ▼
  [검출 스레드]  ⑤ 두 경로의 결과를 합치고, 축소 좌표를 원본 좌표로 복원해서 저장
        │
        ▼      (메인 스레드는 '지금 완료된 가장 최근 결과'를 읽을 뿐, 현재 프레임의 결과를 기다리지 않음)
  [메인 스레드]  ⑥ 가장 최근 검출 결과 가져오기 (worker.latest)
        │
        ▼
  [메인 스레드]  ⑦ 프레임 위에 바운딩 박스 + FPS 그리기
        │
        ▼
  [메인 스레드]  ⑧ 화면 크기에 맞춰 축소 후 창에 출력 (imshow)
        │
        ▼
  [메인 스레드]  ⑨ 영상 원래 속도에 맞춰 대기 → 키 입력 처리 → 뒤처졌으면 프레임 건너뜀

■ 핵심 포인트
  - 재생(메인)과 검출(별도 스레드)이 분리되어 있어, 검출이 느려도 영상은 원래 속도로 재생된다.
    (대신 박스가 영상보다 몇 프레임 늦게 따라온다)
  - 일반 객체는 '전처리 O + 낮은 해상도', 신호등은 '전처리 X + 높은 해상도 + 색 검증'으로
    서로 다르게 최적화한다.
  - YOLO 모델은 1개를 공유하고, 검출 1회마다 YOLO를 2번(일반 / 신호등) 호출한다.
"""

import os          # 파일 존재 확인(os.path.exists), CPU 코어 수 확인(os.cpu_count)
import time        # 시간 측정(perf_counter): FPS 계산, 재생 속도 동기화
import logging     # ultralytics가 출력하는 특정 경고 메시지 걸러내기
import threading   # 검출 전용 스레드와 스레드 간 동기화(Condition, Event)

import cv2                         # 영상 읽기/표시, 리사이즈, 색 공간 변환 등 영상처리
import numpy as np                 # 배열(박스 좌표 등) 연산
import torch                       # GPU 사용 가능 여부 확인, 연산 설정
from ultralytics import YOLO       # YOLO 객체 검출 모델

try:
    from ..common.data_models import ObjectBox
except Exception:
    try:
        from src.common.data_models import ObjectBox
    except Exception:
        ObjectBox = None

COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}

cv2.setUseOptimized(True)          # OpenCV 내부 최적화(SIMD 등) 켜기

# ==========================================
# 설정값 로딩 (dashcam_cv/config/settings.yaml 연동)
# ==========================================
try:
    import yaml
except ImportError:
    yaml = None


def _load_settings():
    if yaml is None:
        return {}
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")),
        os.path.abspath("config/settings.yaml"),
        os.path.abspath("dashcam_cv/config/settings.yaml"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


_cfg = _load_settings()
_v_cfg = _cfg.get("video", {})
_d_cfg = _cfg.get("detector", {})
_tl_cfg = _d_cfg.get("traffic_light", {})
_colors = _d_cfg.get("colors", {})
_ne_cfg = _d_cfg.get("night_enhancer", {})

INPUT_VIDEO = _v_cfg.get("input_path", "images/Test_Video_2.mp4")   # 입력 영상 파일 경로
MODEL_WEIGHT = _d_cfg.get("model_name", "yolo26n.pt")                # 사용할 YOLO 가중치 파일

TRAFFIC_LIGHT_CLASS = int(_tl_cfg.get("class_id", 9))               # 신호등 전용 경로 (COCO 클래스 9)
_raw_classes = list(_d_cfg.get("target_classes", [0, 1, 2, 3, 5, 7]))
TARGET_CLASSES = [c for c in _raw_classes if c != TRAFFIC_LIGHT_CLASS]  # 보행자, 자전거, 승용차, 오토바이, 버스, 트럭

IMGSZ_GPU = int(_d_cfg.get("imgsz_gpu", 640))            # 일반 객체 입력 해상도 (GPU)
IMGSZ_CPU = int(_d_cfg.get("imgsz_cpu", 480))            # 일반 객체 입력 해상도 (CPU)
HOOD_CROP_RATIO = float(_d_cfg.get("hood_crop_ratio", 0.15))     # 하단 본네트 영역 제거 비율
ENABLE_ENHANCE = bool(_d_cfg.get("enable_enhance", True))      # 일반 객체 검출용 야간 전처리
DISPLAY_MAX_WIDTH = int(_d_cfg.get("display_max_width", 1280))   # 화면에 표시할 때의 최대 가로 크기
SHOW_FPS = bool(_d_cfg.get("show_fps", True))            # 화면 좌측 상단에 FPS 표시 여부

CONF = float(_d_cfg.get("confidence_threshold", 0.22))   # 일반 객체 신뢰도 임계값
IOU = float(_d_cfg.get("iou_threshold", 0.45))          # NMS IoU 임계값

# --- 신호등 전용 설정 ---
TL_ROI_RATIO = float(_tl_cfg.get("roi_ratio", 0.6))         # 본네트 제거 후 상단 탐색 영역 비율
TL_IMGSZ_GPU = int(_tl_cfg.get("imgsz_gpu", 1280))        # 신호등 탐색 영역 입력 해상도 (GPU)
TL_IMGSZ_CPU = int(_tl_cfg.get("imgsz_cpu", 640))         # 신호등 탐색 영역 입력 해상도 (CPU)
TL_CONF = float(_tl_cfg.get("confidence_threshold", 0.10))  # 낮은 신뢰도로 후보 수집
TL_MAX_ASPECT = float(_tl_cfg.get("max_aspect_ratio", 5.0)) # 박스 긴 변/짧은 변 허용 최대 비율
TL_MIN_SAT = int(_tl_cfg.get("min_saturation", 80))        # 램프 픽셀 최소 채도
TL_MIN_VAL = int(_tl_cfg.get("min_value", 140))           # 램프 픽셀 최소 밝기
TL_MIN_LAMP_RATIO = float(_tl_cfg.get("min_lamp_ratio", 0.03)) # 램프 픽셀 최소 비율

COLOR_OBJECT = tuple(_colors.get("object", [0, 255, 255]))   # 일반 객체 박스 색 (BGR: 노란색)
COLOR_LIGHT = tuple(_colors.get("traffic_light", [0, 255, 0])) # 신호등 박스 색 (BGR: 녹색)

NE_CLIP_LIMIT = float(_ne_cfg.get("clip_limit", 2.0))
NE_GRID_SIZE = tuple(_ne_cfg.get("tile_grid_size", [8, 8]))
NE_GAMMA = float(_ne_cfg.get("gamma", 0.8))

# 검출 결과가 없을 때 돌려주는 '빈 결과' = (박스 좌표 0개, 클래스 ID 0개)
_EMPTY = (np.empty((0, 4), np.float32), np.empty(0, int))


# ==========================================
# 1. 하드웨어 디바이스 선택
# ==========================================
def pick_device():
    # GPU(CUDA)를 안전하게 쓸 수 있는지 3단계로 점검해서 "cuda" 또는 "cpu"를 돌려준다.
    cuda_build = torch.version.cuda
    print(f"[PyTorch] 버전 {torch.__version__} | CUDA 빌드: {cuda_build or '없음(CPU 전용)'}")

    # 점검 1) PyTorch가 CUDA를 인식하는가?
    if not torch.cuda.is_available():
        if cuda_build is None:
            # PyTorch 자체가 CPU 전용 버전으로 설치된 경우
            print("[경고] CPU 전용 PyTorch가 설치되어 있어 GPU를 쓸 수 없습니다.")
        else:
            # CUDA용 PyTorch인데 드라이버/그래픽카드가 인식되지 않는 경우
            print("[경고] CUDA를 사용할 수 없습니다. 드라이버/그래픽카드 인식을 확인하세요 (nvidia-smi).")
        _print_reinstall_help()
        return "cpu"

    # 점검 2) 설치된 PyTorch 빌드가 이 GPU의 아키텍처(예: sm_120)를 지원하는가?
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        print(f"[경고] 현재 PyTorch 빌드는 {arch}를 지원하지 않아 CPU로 실행합니다.")
        _print_reinstall_help()
        return "cpu"

    # 점검 3) 실제로 GPU 메모리에 텐서를 만들어 연산이 되는지 시험
    try:
        _ = torch.zeros(1, device="cuda") + 1
    except Exception as e:
        print(f"[경고] CUDA 테스트 실패({e}) -> CPU로 실행합니다.")
        _print_reinstall_help()
        return "cpu"

    print(f"[GPU] {torch.cuda.get_device_name(0)} ({arch}) 사용")
    return "cuda"


def _print_reinstall_help():
    # GPU용 PyTorch를 다시 설치하는 방법을 안내한다 (GPU를 못 쓸 때만 호출됨)
    print("       [해결 방법] 같은 파이썬 환경에서 순서대로 실행:")
    print("         python -m pip uninstall -y torch torchvision torchaudio")
    print("         python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")


# ==========================================
# 2. 야간 전처리 (L 채널 감마 + CLAHE)
# ==========================================
class NightEnhancer:
    # 어두운 야간 영상을 검출하기 좋게 밝기/대비를 보정한다.
    # BGR -> LAB로 바꿔 '밝기(L)' 채널에만 보정을 적용하므로 색상이 변하지 않는다.
    def __init__(self, clip_limit=NE_CLIP_LIMIT, tile_grid_size=NE_GRID_SIZE, gamma=NE_GAMMA):
        # CLAHE: 화면을 격자(8x8)로 나눠 지역별로 대비를 높이는 기법 (clip_limit이 클수록 대비 강함)
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        # 감마 보정표(LUT): 0~255 각 값이 보정 후 어떤 값이 되는지 미리 계산해 둔다.
        # gamma < 1 이면 어두운 부분이 밝아진다. 표를 미리 만들어 두면 매 프레임 거듭제곱 계산이 필요 없다.
        self.lut = (((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255).astype(np.uint8)

    def __call__(self, bgr):
        # enhancer(img) 처럼 함수처럼 호출하면 실행된다.
        l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))   # 밝기(l)와 색상(a, b)을 분리
        l = self.clahe.apply(cv2.LUT(l, self.lut))                  # 밝기에만 감마 보정 -> CLAHE 적용
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)  # 다시 합쳐서 BGR로 복원


# ==========================================
# 3. 객체 검출
# ==========================================
class _DropHalfWarning(logging.Filter):
    # ultralytics가 'half' 옵션 관련 경고를 반복 출력하면 콘솔이 지저분해지므로 그 메시지만 숨긴다.
    def filter(self, record):
        return "'half' is deprecated" not in record.getMessage()


class Detector:
    # YOLO 모델 1개를 일반 객체/신호등 두 경로에서 공유한다.
    def __init__(self, weight, device):
        logging.getLogger("ultralytics").addFilter(_DropHalfWarning())   # 위 경고 필터 등록
        self.model = YOLO(weight)   # 가중치 파일을 읽어 모델 로드 (파일이 없으면 자동 다운로드)
        self.device = device        # "cuda" 또는 "cpu"
        self.half = False           # FP16(반정밀도) 사용 여부. 예열 단계에서 GPU일 때 켜본다.

    def predict(self, img, imgsz, classes, conf):
        # img 한 장을 검출한다.
        #   imgsz   : 모델에 넣을 입력 크기
        #   classes : 검출할 클래스 번호 목록 (일반 객체 / 신호등 경로마다 다르게 전달)
        #   conf    : 이 신뢰도 미만의 검출은 버림
        r = self.model.predict(
            source=img,
            imgsz=imgsz,
            conf=conf,
            iou=IOU,
            classes=classes,
            device=self.device,
            max_det=100,       # 한 장당 최대 검출 개수
            verbose=False,     # 추론 로그 출력 끔
            **({"half": True} if self.half else {}),   # FP16이 켜져 있을 때만 half=True 전달
        )[0]   # 입력이 이미지 1장이므로 결과 리스트의 첫 번째 항목만 사용
        b = r.boxes
        if b is None or len(b) == 0:
            return np.empty((0, 4), np.float32), np.empty(0, int), np.empty(0, np.float32)
        # GPU 텐서 -> CPU numpy 배열로 바꿔서 (박스 좌표 xyxy, 클래스 번호, 신뢰도)를 돌려준다.
        return b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()


class AsyncDetector:
    # 검출을 별도 스레드에서 돌려 재생이 검출 속도에 묶이지 않게 한다.
    def __init__(self, detector, enhancer, W, crop_h, imgsz, tl_imgsz, warm_crop):
        self.det = detector       # 공유할 Detector
        self.enh = enhancer       # 야간 전처리기 (None이면 전처리 안 함)
        self.imgsz = imgsz        # 일반 객체 입력 해상도
        self._warm = warm_crop    # 예열에 쓸 실제 첫 프레임(본네트 제거본)

        # ── 일반 객체 경로의 크기 계산 ──
        # 본네트 제거한 프레임(W x crop_h)의 긴 변이 imgsz가 되도록 축소 비율 g를 구한다.
        g = imgsz / max(W, crop_h)
        self.g_size = (max(1, round(W * g)), max(1, round(crop_h * g)))   # 축소 후 (가로, 세로)
        # 축소 이미지 좌표 -> 원본 좌표로 되돌리는 배율 [x, y, x, y] (박스 좌표 4개에 한 번에 곱하려고 2번 반복)
        self.g_k = np.array([W / self.g_size[0], crop_h / self.g_size[1]] * 2, np.float32)

        # ── 신호등 경로의 크기 계산 ──
        # 신호등은 화면 위쪽에 있으므로 위쪽 TL_ROI_RATIO 만큼(roi_h)만 잘라서 사용한다.
        self.roi_h = max(1, int(crop_h * TL_ROI_RATIO))
        # 신호등 입력 크기에 맞춰 줄이되, 원본보다 크게 확대(min 1.0)하지는 않는다.
        t = min(1.0, tl_imgsz / max(W, self.roi_h))
        self.t_size = (max(1, round(W * t)), max(1, round(self.roi_h * t)))   # 축소 후 (가로, 세로)
        # 신호등 경로의 좌표 복원 배율 (ROI는 화면 맨 위부터 시작하므로 위치 보정(오프셋)은 필요 없음)
        self.t_k = np.array([W / self.t_size[0], self.roi_h / self.t_size[1]] * 2, np.float32)
        # YOLO는 32의 배수 크기를 쓰므로, 축소된 긴 변을 32의 배수로 올림해서 모델 입력 크기로 사용
        self.tl_imgsz = int(np.ceil(max(self.t_size) / 32) * 32)

        # ── 스레드 간 공유 상태 ──
        self._cv = threading.Condition()   # 공유 변수를 보호하는 잠금 + '새 프레임이 왔다' 알림 장치
        self._pending = None               # 검출을 기다리는 최신 프레임 (없으면 None)
        self._result = _EMPTY              # 가장 최근에 완료된 검출 결과
        self._stop = False                 # True가 되면 검출 스레드가 종료됨
        self.infer_fps = 0.0               # 검출 속도(초당 검출 횟수), 화면 표시용
        self.error = None                  # 검출 스레드에서 오류가 나면 여기에 저장
        self.ready = threading.Event()     # 예열이 끝나면 켜지는 신호 (메인은 이걸 기다렸다가 재생 시작)

        # 검출 스레드 시작 (daemon=True: 메인이 끝나면 함께 종료)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, crop):
        # [파이프라인 ②] 메인 스레드가 새 프레임을 넘겨준다.
        with self._cv:
            self._pending = crop   # 아직 처리 못 한 프레임이 있어도 최신 프레임으로 덮어쓴다 (밀린 프레임은 버림)
            self._cv.notify()      # 대기 중인 검출 스레드를 깨운다

    def latest(self):
        # [파이프라인 ⑥] 메인 스레드가 가장 최근에 완료된 검출 결과를 읽어간다.
        with self._cv:
            return self._result

    def stop(self):
        # 프로그램 종료 시 검출 스레드를 멈춘다.
        with self._cv:
            self._stop = True
            self._cv.notify()                 # 대기 중이면 깨워서 종료 플래그를 보게 한다
        self._thread.join(timeout=2.0)        # 최대 2초 동안 스레드가 끝나기를 기다림

    def _verify_traffic_lights(self, roi, xyxy, ids):
        # 건물 조명 오검출 제거: 박스 모양 + 박스 안에 적/황/녹색 램프 픽셀이 충분한지 확인
        if not len(ids):
            return _EMPTY   # 후보가 없으면 검증할 것도 없음

        # 이미지를 HSV(색상/채도/밝기) 공간으로 바꿔 '밝고 선명한 적/황/녹색' 픽셀만 골라낸다.
        # (OpenCV의 색상 H는 0~179 범위)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lit = (
            cv2.inRange(hsv, (0, TL_MIN_SAT, TL_MIN_VAL), (12, 255, 255))        # 적색
            | cv2.inRange(hsv, (18, TL_MIN_SAT, TL_MIN_VAL), (95, 255, 255))     # 황색~녹색
            | cv2.inRange(hsv, (165, TL_MIN_SAT, TL_MIN_VAL), (180, 255, 255))   # 적색(반대편 색상)
        )   # lit: 조건을 만족하는 픽셀만 255, 나머지는 0인 마스크 (세 범위를 OR로 합침)

        keep = []   # 검증을 통과한 박스의 번호
        for i, (x1, y1, x2, y2) in enumerate(xyxy.astype(int)):
            w, h = x2 - x1, y2 - y1
            # 검증 1) 모양: 너무 가늘고 긴 박스(긴 변 > 짧은 변 x 5)는 신호등이 아니라 판단
            if min(w, h) < 1 or max(w, h) > TL_MAX_ASPECT * min(w, h):
                continue
            # 검증 2) 색: 박스 안의 램프색 픽셀 수가 (최소 2개, 박스 면적의 3%) 이상이어야 통과
            if cv2.countNonZero(lit[y1:y2, x1:x2]) >= max(2, TL_MIN_LAMP_RATIO * w * h):
                keep.append(i)
        keep = np.array(keep, dtype=int)
        if len(keep) == 0:
            return _EMPTY
        return xyxy[keep], ids[keep]   # 통과한 박스만 돌려줌

    def _infer(self, crop):
        # crop: 본네트를 제거한 원본 해상도 프레임. 두 경로를 모두 실행해 결과를 합친다.

        # [파이프라인 ④-A] 일반 객체: 축소 + 야간 전처리 후 검출
        small = cv2.resize(crop, self.g_size, interpolation=cv2.INTER_AREA)   # 축소 (INTER_AREA: 줄일 때 화질이 좋음)
        if self.enh is not None:
            small = self.enh(small)   # 축소한 뒤에 전처리해야 연산량이 적다 (원본 해상도에서 CLAHE를 돌리지 않음)
        g_xyxy, g_ids, _ = self.det.predict(small, self.imgsz, TARGET_CLASSES, CONF)

        # [파이프라인 ④-B] 신호등: 상단 ROI, 전처리 없이(CLAHE가 주변 조명을 키우는 것을 방지) 고해상도 검출 후 검증
        roi = cv2.resize(crop[: self.roi_h], self.t_size, interpolation=cv2.INTER_AREA)   # 위쪽 ROI만 잘라 축소
        t_xyxy, t_ids, _ = self.det.predict(roi, self.tl_imgsz, [TRAFFIC_LIGHT_CLASS], TL_CONF)   # 신호등만, 낮은 conf
        t_xyxy, t_ids = self._verify_traffic_lights(roi, t_xyxy, t_ids)   # 모양/램프색 검증으로 오검출 제거

        # [파이프라인 ⑤] 원본 좌표계로 복원 (ROI는 원점이 같아 오프셋 불필요)하고 두 경로의 결과를 합친다.
        return (
            np.concatenate((g_xyxy * self.g_k, t_xyxy * self.t_k)),   # 박스 좌표: 축소 좌표 x 배율 -> 원본 좌표
            np.concatenate((g_ids, t_ids)),                           # 클래스 번호
        )

    def _warmup(self):
        # 모델 예열: 첫 추론은 GPU 초기화/메모리 할당 때문에 느리므로, 재생을 시작하기 전에 미리 몇 번 돌려둔다.
        # (검출 스레드 안에서 실행해야 GPU 초기화가 이 스레드에서 끝난다)
        if self.det.device == "cuda":
            # GPU면 FP16(반정밀도)을 먼저 시도한다. 더 빠르지만 환경에 따라 실패할 수 있다.
            self.det.half = True
            try:
                self._infer(self._warm)
            except Exception:
                self.det.half = False   # 실패하면 FP32(기본 정밀도)로 되돌린다
        result = _EMPTY
        for _ in range(3):
            result = self._infer(self._warm)   # 실제 첫 프레임으로 3회 추론해서 연산 경로 전체를 데워둠
        print(
            f"[Detector] {MODEL_WEIGHT} | device={self.det.device.upper()} | "
            f"precision={'FP16' if self.det.half else 'FP32'} | imgsz={self.imgsz} | traffic_light_imgsz={self.tl_imgsz}"
        )
        return result   # 마지막 예열 결과를 첫 검출 결과로 사용 (재생 시작부터 박스가 보이도록)

    def _run(self):
        # 검출 스레드의 본체 (스레드가 시작되면 이 함수가 실행된다)
        try:
            # 1단계: 예열 후 첫 결과를 저장하고, 메인 스레드에 '준비 완료'를 알린다.
            result = self._warmup()
            with self._cv:
                self._result = result
            self.ready.set()

            # 2단계: 종료될 때까지 반복 - 최신 프레임을 꺼내 검출하고 결과를 저장한다.
            while True:
                # [파이프라인 ③] 새 프레임이 올 때까지 대기했다가, 가장 최신 프레임 1장을 꺼낸다.
                with self._cv:
                    while self._pending is None and not self._stop:
                        self._cv.wait()   # 새 프레임/종료 요청이 올 때까지 잠자기 (CPU를 쓰지 않음)
                    if self._stop:
                        return            # 종료 요청이 오면 스레드 끝
                    img, self._pending = self._pending, None   # 꺼내면서 대기열은 비운다

                # 검출은 잠금(lock) 밖에서 실행한다 (오래 걸려도 메인 스레드가 막히지 않도록)
                t = time.perf_counter()
                result = self._infer(img)
                dt = max(time.perf_counter() - t, 1e-6)   # 이번 검출에 걸린 시간(0 나누기 방지용 최소값)

                # 결과 저장 + 검출 FPS 갱신 (지수 이동 평균으로 값이 부드럽게 변하게 함)
                with self._cv:
                    self._result = result
                    self.infer_fps = (1.0 / dt) if self.infer_fps == 0 else 0.8 * self.infer_fps + 0.2 / dt
        except Exception as e:
            # 스레드 안의 오류를 메인 스레드가 알 수 있게 저장하고,
            # 예열 중 오류여도 메인이 영원히 기다리지 않도록 ready 신호를 켠다.
            self.error = e
            self.ready.set()


class ObjectDetector:
    """dashcam_cv main.py 파이프라인에서 직접 호출하는 objects_detector_th 기반 객체 검출기."""

    def __init__(self, cfg=None):
        self.cfg = cfg or _load_settings()
        det_cfg = self.cfg.get("detector", {})
        tl_cfg = det_cfg.get("traffic_light", {})
        ne_cfg = det_cfg.get("night_enhancer", {})

        self.enabled = det_cfg.get("enabled", True)
        self.model_weight = det_cfg.get("model_name", MODEL_WEIGHT)
        self.interval = max(1, int(det_cfg.get("detect_interval", 5)))
        self.conf = float(det_cfg.get("confidence_threshold", CONF))
        self.person_conf = float(det_cfg.get("person_confidence_threshold", self.conf))
        self.iou = float(det_cfg.get("iou_threshold", IOU))

        self.hood_crop_ratio = float(det_cfg.get("hood_crop_ratio", HOOD_CROP_RATIO))
        self.enable_enhance = bool(det_cfg.get("enable_enhance", ENABLE_ENHANCE))
        self.imgsz_gpu = int(det_cfg.get("imgsz_gpu", IMGSZ_GPU))
        self.imgsz_cpu = int(det_cfg.get("imgsz_cpu", IMGSZ_CPU))

        self.tl_class_id = int(tl_cfg.get("class_id", TRAFFIC_LIGHT_CLASS))
        self.tl_roi_ratio = float(tl_cfg.get("roi_ratio", TL_ROI_RATIO))
        self.tl_imgsz_gpu = int(tl_cfg.get("imgsz_gpu", TL_IMGSZ_GPU))
        self.tl_imgsz_cpu = int(tl_cfg.get("imgsz_cpu", TL_IMGSZ_CPU))
        self.tl_conf = float(tl_cfg.get("confidence_threshold", TL_CONF))
        self.tl_max_aspect = float(tl_cfg.get("max_aspect_ratio", TL_MAX_ASPECT))
        self.tl_min_sat = int(tl_cfg.get("min_saturation", TL_MIN_SAT))
        self.tl_min_val = int(tl_cfg.get("min_value", TL_MIN_VAL))
        self.tl_min_lamp_ratio = float(tl_cfg.get("min_lamp_ratio", TL_MIN_LAMP_RATIO))

        raw_classes = list(det_cfg.get("target_classes", [0, 1, 2, 3, 5, 7]))
        self.target_classes = [c for c in raw_classes if c != self.tl_class_id]

        self.device = pick_device()
        self.imgsz = self.imgsz_gpu if self.device == "cuda" else self.imgsz_cpu
        self.tl_imgsz = self.tl_imgsz_gpu if self.device == "cuda" else self.tl_imgsz_cpu

        self.detector = Detector(self.model_weight, self.device)
        self.enhancer = NightEnhancer(
            clip_limit=float(ne_cfg.get("clip_limit", NE_CLIP_LIMIT)),
            tile_grid_size=tuple(ne_cfg.get("tile_grid_size", NE_GRID_SIZE)),
            gamma=float(ne_cfg.get("gamma", NE_GAMMA)),
        ) if self.enable_enhance else None

        self._warmed_up = False
        self.cached_boxes = []

    def detect(self, frame: np.ndarray, frame_idx: int):
        if not self.enabled:
            return [], False

        should_run = ((frame_idx - 1) % self.interval == 0) or (len(self.cached_boxes) == 0 and frame_idx == 1)
        if not should_run:
            return self.cached_boxes, False

        h, w = frame.shape[:2]
        crop_h = int(h * (1.0 - self.hood_crop_ratio))
        crop = frame[:crop_h]

        g = self.imgsz / max(w, crop_h)
        g_size = (max(1, round(w * g)), max(1, round(crop_h * g)))
        g_k = np.array([w / g_size[0], crop_h / g_size[1]] * 2, np.float32)

        roi_h = max(1, int(crop_h * self.tl_roi_ratio))
        t = min(1.0, self.tl_imgsz / max(w, roi_h))
        t_size = (max(1, round(w * t)), max(1, round(roi_h * t)))
        t_k = np.array([w / t_size[0], roi_h / t_size[1]] * 2, np.float32)
        tl_model_sz = int(np.ceil(max(t_size) / 32) * 32)

        if not self._warmed_up:
            if self.device == "cuda":
                self.detector.half = True
            for _ in range(3):
                small_w = cv2.resize(crop, g_size, interpolation=cv2.INTER_AREA)
                if self.enhancer:
                    small_w = self.enhancer(small_w)
                self.detector.predict(small_w, self.imgsz, self.target_classes, self.conf)
            self._warmed_up = True

        # ── 경로 1: 일반 객체 ──
        small = cv2.resize(crop, g_size, interpolation=cv2.INTER_AREA)
        if self.enhancer:
            small = self.enhancer(small)
        g_xyxy, g_ids, g_confs = self.detector.predict(small, self.imgsz, self.target_classes, min(self.conf, self.person_conf))

        # ── 경로 2: 신호등 전용 경로 ──
        roi = cv2.resize(crop[:roi_h], t_size, interpolation=cv2.INTER_AREA)
        t_xyxy, t_ids, t_confs = self.detector.predict(roi, tl_model_sz, [self.tl_class_id], self.tl_conf)
        t_xyxy, t_ids, t_confs = self._verify_traffic_lights(
            roi, t_xyxy, t_ids, t_confs,
            max_aspect=self.tl_max_aspect,
            min_sat=self.tl_min_sat,
            min_val=self.tl_min_val,
            min_lamp_ratio=self.tl_min_lamp_ratio,
        )

        all_boxes = []
        if len(g_ids):
            for (x1, y1, x2, y2), cid, sc in zip(g_xyxy * g_k, g_ids, g_confs):
                min_sc = self.person_conf if cid == 0 else self.conf
                if sc < min_sc:
                    continue
                all_boxes.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), cid, sc))

        if len(t_ids):
            for (x1, y1, x2, y2), cid, sc in zip(t_xyxy * t_k, t_ids, t_confs):
                if sc < self.tl_conf:
                    continue
                all_boxes.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), cid, sc))

        result_boxes = []
        for x1, y1, x2, y2, cid, sc in all_boxes:
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(crop_h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(crop_h - 1, y2))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            label = COCO_CLASS_NAMES.get(int(cid), f"class_{cid}")

            if ObjectBox is not None:
                result_boxes.append(
                    ObjectBox(
                        track_id=-1,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        class_id=int(cid),
                        label=label,
                        confidence=float(sc),
                        center=(cx, cy),
                    )
                )

        self.cached_boxes = result_boxes
        return result_boxes, True

    def _verify_traffic_lights(self, roi, xyxy, ids, confs, max_aspect=5.0, min_sat=80, min_val=140, min_lamp_ratio=0.03):
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
            if min(w, h) < 1 or max(w, h) > max_aspect * min(w, h):
                continue
            if cv2.countNonZero(lit[y1:y2, x1:x2]) >= max(2, min_lamp_ratio * w * h):
                keep.append(i)
        keep = np.array(keep, dtype=int)
        if len(keep) == 0:
            return np.empty((0, 4), np.float32), np.empty(0, int), np.empty(0, np.float32)
        return xyxy[keep], ids[keep], confs[keep]


def draw_boxes(frame, xyxy, cls_ids):
    # 프레임 위에 바운딩 박스(네모)만 그린다 (글씨 없음)
    for (x1, y1, x2, y2), c in zip(xyxy.astype(int), cls_ids):
        if c == TRAFFIC_LIGHT_CLASS:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_LIGHT, 1)    # 신호등: 녹색, 두께 1
        else:
            # 그 외 객체: 노란색. 보행자(0)는 눈에 띄도록 두께 2, 나머지는 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OBJECT, 2 if c == 0 else 1)


# ==========================================
# 4. 재생 및 메인 루프
# ==========================================
WINDOW = "Night Drive Detection"   # 영상이 표시되는 창 이름


def run(input_path):
    # ── 시작 단계 ─────────────────────────────────────────
    # 영상 파일 열기 (상대 경로 및 data 폴더 탐색)
    if not os.path.exists(input_path):
        for alt in [
            os.path.join("dashcam_cv", input_path),
            os.path.join("..", "..", input_path),
            "data/Test_Video_2.mp4",
            "dashcam_cv/data/Test_Video_2.mp4",
            "images/Test_Video_2.mp4",
        ]:
            if os.path.exists(alt):
                input_path = alt
                break

    if not os.path.exists(input_path):
        print(f"[오류] 입력 파일을 찾을 수 없습니다: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[오류] 비디오를 열 수 없습니다: {input_path}")
        return

    # 영상 정보: 가로/세로 크기, 초당 프레임 수(FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0                  # FPS 정보를 읽지 못하면 30으로 가정
    frame_dt = 1.0 / fps            # 프레임 1장의 재생 시간(초)

    # GPU/CPU 선택 후 장치에 맞는 입력 해상도와 최적화 옵션 설정
    device = pick_device()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True        # 입력 크기에 가장 빠른 GPU 연산 방식을 자동 선택
        imgsz, tl_imgsz = IMGSZ_GPU, TL_IMGSZ_GPU
    else:
        torch.set_num_threads(os.cpu_count() or 4)   # CPU 코어를 최대한 사용
        imgsz, tl_imgsz = IMGSZ_CPU, TL_IMGSZ_CPU    # CPU는 느리므로 더 작은 해상도 사용

    detector = Detector(MODEL_WEIGHT, device)                  # YOLO 모델 로드
    enhancer = NightEnhancer() if ENABLE_ENHANCE else None     # 야간 전처리기 (설정이 꺼져 있으면 None)

    crop_h = int(H * (1.0 - HOOD_CROP_RATIO))                  # 본네트를 뺀 높이 (프레임 위쪽 crop_h 행만 검출에 사용)
    disp_scale = min(1.0, DISPLAY_MAX_WIDTH / W)               # 화면 표시용 축소 비율 (확대는 안 함)
    disp_size = (max(1, int(W * disp_scale)), max(1, int(H * disp_scale)))
    font_scale = max(0.5, W / 1800)                            # 영상 크기에 맞춰 글씨 크기 조절

    # 첫 프레임을 미리 읽어 둔다 (예열에 사용하고, 이후 재생 때도 이 프레임부터 보여줌)
    ok, first_frame = cap.read()
    if not ok:
        print("[오류] 첫 프레임을 읽을 수 없습니다.")
        cap.release()
        return

    # 검출 스레드 시작 + 예열. 예열이 끝날 때까지 기다린 뒤에 재생을 시작해야 처음부터 끊기지 않는다.
    print("[정보] 모델 예열 중... (끝나면 재생이 시작됩니다)")
    worker = AsyncDetector(detector, enhancer, W, crop_h, imgsz, tl_imgsz, first_frame[:crop_h].copy())
    worker.ready.wait()   # 예열 완료(또는 오류) 신호를 기다림

    if worker.error is not None:
        print(f"[오류] 모델 예열 실패: {worker.error!r}")
        worker.stop()
        cap.release()
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)   # 크기 조절이 가능한 창 생성

    # 재생 상태 변수
    idx = 0              # 지금까지 읽은 프레임 번호 (건너뛴 프레임 포함)
    shown = 0            # 실제로 화면에 표시한 프레임 수
    disp_fps = 0.0       # 화면 표시 FPS
    t_prev = None        # 직전 프레임을 표시한 시각
    t0 = time.perf_counter()   # 재생 시작 시각: 이 시각을 기준으로 '영상 시간'에 맞춰 재생한다

    # ── 프레임 처리 단계 (영상이 끝나거나 사용자가 종료할 때까지 반복) ──
    try:
        while worker.error is None:   # 검출 스레드에 오류가 생기면 루프 종료
            # [파이프라인 ①] 프레임 읽기. 첫 프레임은 이미 읽어 둔 것을 쓰고, 이후에는 영상에서 읽는다.
            if first_frame is not None:
                ok, frame, first_frame = True, first_frame, None
            else:
                ok, frame = cap.read()
            if not ok:
                break            # 영상 끝
            idx += 1

            # [파이프라인 ②] 프레임에 박스를 그리기 전에 검출용 복사본을 전달 (축소/전처리는 검출 스레드에서 수행)
            #  - frame[:crop_h]: 본네트 영역 제거
            #  - .copy(): 아래에서 frame에 박스를 그리므로, 검출 스레드가 쓰는 데이터와 분리하기 위해 복사
            worker.submit(frame[:crop_h].copy())

            # [파이프라인 ⑥~⑦] 가장 최근 검출 결과를 가져와 프레임 위에 박스를 그린다.
            #  - 방금 넘긴 프레임의 결과가 아니라, 직전에 완료된 결과이므로 박스가 약간 늦게 따라온다.
            xyxy, cls_ids = worker.latest()
            draw_boxes(frame, xyxy, cls_ids)

            # 화면 표시 FPS 계산 (지수 이동 평균으로 부드럽게)
            now = time.perf_counter()
            if t_prev is not None:
                inst = 1.0 / max(now - t_prev, 1e-6)   # 직전 프레임과의 간격으로 구한 순간 FPS
                disp_fps = inst if disp_fps == 0 else 0.9 * disp_fps + 0.1 * inst
            t_prev = now

            # 좌측 상단에 View FPS(재생 속도)와 Detect FPS(검출 속도) 표시
            if SHOW_FPS:
                cv2.putText(
                    frame,
                    f"View {disp_fps:.0f} fps | Detect {worker.infer_fps:.0f} fps",
                    (12, int(28 * font_scale) + 4),     # 글씨 위치
                    cv2.FONT_HERSHEY_SIMPLEX,           # 글꼴
                    0.8 * font_scale,                   # 글씨 크기
                    COLOR_OBJECT,                       # 글씨 색
                    max(1, int(2 * font_scale)),        # 글씨 두께
                    cv2.LINE_AA,                        # 부드러운 글씨 테두리
                )

            # [파이프라인 ⑧] 화면 크기에 맞춰 줄인 뒤 창에 출력 (이미 작으면 그대로 출력)
            show = cv2.resize(frame, disp_size, interpolation=cv2.INTER_AREA) if disp_scale < 1.0 else frame
            cv2.imshow(WINDOW, show)
            shown += 1

            # [파이프라인 ⑨] 실시간 동기화: 이 프레임의 목표 시각(t0 + idx x 프레임 간격)까지 남은 시간만큼 대기
            #  - 처리가 빨랐으면 남은 시간만큼 기다려 영상이 너무 빨리 재생되지 않게 하고,
            #  - waitKey는 대기 중에 키 입력도 받는다 (최소 1ms)
            wait_ms = max(1, int((t0 + idx * frame_dt - time.perf_counter()) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF

            # q 또는 ESC(27)를 누르면 종료
            if key in (ord("q"), 27):
                print("\n[사용자 중단]")
                break

            # 창의 X 버튼으로 닫았는지 확인 (닫혔으면 종료)
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            # 스페이스바: 일시정지. 다시 스페이스바를 누르면 재생, q/ESC면 종료
            if key == ord(" "):
                while True:
                    k = cv2.waitKey(50) & 0xFF
                    if k in (ord(" "), ord("q"), 27):
                        break
                if k in (ord("q"), 27):
                    break
                # 멈춰 있던 시간만큼 기준 시각을 뒤로 밀어, 재생 시 프레임이 한꺼번에 건너뛰어지지 않게 한다
                t0 = time.perf_counter() - idx * frame_dt
                continue

            # 처리가 영상 시간보다 뒤처지면 grab()으로 디코딩 없이 프레임 건너뜀
            #  - behind: 지금 시각이 이 프레임의 목표 시각보다 얼마나 늦었는지(초)
            #  - 프레임 간격 몇 개만큼 늦었는지 계산해 그만큼의 프레임을 건너뛰어 실시간 속도를 따라잡는다
            behind = time.perf_counter() - (t0 + idx * frame_dt)
            for _ in range(int(behind / frame_dt)):
                if not cap.grab():   # grab: 프레임을 디코딩하지 않고 넘김 (read보다 훨씬 빠름)
                    break
                idx += 1

    finally:
        # 정상 종료/중단/오류 어떤 경우에도 자원을 정리한다.
        worker.stop()               # 검출 스레드 종료
        cap.release()               # 영상 파일 닫기
        cv2.destroyAllWindows()     # 창 닫기

    # 검출 스레드 오류로 루프가 끝났다면 원인을 알려준다.
    if worker.error is not None:
        print(f"[오류] 검출 스레드 중단: {worker.error!r}")

    # 최종 요약: 실제 표시한 프레임 수 / 영상 위치 / 걸린 시간
    print(f"[완료] 표시 {shown}프레임 / 영상 위치 {idx}프레임 | {time.perf_counter() - t0:.1f}초")


if __name__ == "__main__":
    run(INPUT_VIDEO)   # 이 파일을 직접 실행했을 때만 시작