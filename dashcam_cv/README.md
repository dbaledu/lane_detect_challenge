# 도로주행 영상 기반 경량 자율주행 비전 분석 시스템 (dashcam_cv)

본 시스템은 **"도로주행 영상 기반 경량 자율주행 비전 분석 시스템 개발 계획서"**를 기반으로 구현된 고성능·경량 컴퓨터 비전 파이프라인입니다. 녹화된 블랙박스 주행 동영상(주·야간, 도심, 교차로)을 입력받아 **차선 및 대표 주행 경로, 객체(차량·보행자) 및 선행 차량 추적, 횡단보도·정지선 검출, 그리고 주행·감속·정지 상태 머신**을 독립된 모듈로 실시간 분석합니다.

---

## 1. 전체 디렉토리 구조 (계획서 제11절 기준)

```
📁 dashcam_cv/
├── 📁 config/
│   └── settings.yaml             # 임계값, 모델 경로, FPS, ROI 등 종합 설정
├── 📁 src/
│   ├── main.py                   # 전체 파이프라인 실행 및 모듈 오케스트레이션
│   ├── 📁 video/
│   │   ├── video_reader.py       # 동영상 입력, 해상도 리사이즈, 프레임/타임스탬프 관리
│   │   ├── preprocessing.py      # CLAHE 적응형 평활화, HLS/LAB 색상 마스크, Auto-Canny
│   │   └── quality_analyzer.py   # 주·야간 조도 분석, 흔들림/블러 지수 측정
│   ├── 📁 lane/
│   │   ├── lane_detector.py      # HoughLinesP, 기울기 필터링, 1차 다항식 피팅
│   │   ├── lane_tracker.py       # EMA 시계열 스무딩, 차로폭 메모리, 편측 가림 복구
│   │   └── lane_change.py        # 횡방향 오차 드리프트 기반 차선 변경 감지
│   ├── 📁 objects/
│   │   ├── detector.py           # Ultralytics YOLO (Nano) + OpenCV 백업 검출기
│   │   ├── object_tracker.py     # Centroid/IoU ID 추적, 선행 차량(Lead Car) 및 거리 추세
│   │   └── road_markings.py      # 횡단보도(지브라 패턴) 및 정지선(수평선) 탐지
│   ├── 📁 state/
│   │   ├── motion_estimator.py   # 특징점 광학 흐름(Optical Flow) 및 중앙값 필터링
│   │   └── decision_fusion.py    # 자차 움직임 + 앞차 움직임 + 보행자 + 정지선 종합 판단
│   ├── 📁 output/
│   │   ├── video_writer.py       # 주석 오버레이 비디오 인코딩 및 저장
│   │   ├── csv_logger.py         # frame_log.csv, performance.csv 실시간 기록
│   │   └── metrics.py            # 처리 지연시간(p95), 통계 요약 summary.json 산출
│   └── 📁 common/
│       ├── data_models.py        # 모듈 간 표준화된 데이터 클래스 (DataClasses)
│       ├── visualizer.py         # 반투명 주행 코리더, HUD 카드, 디버그 미니맵 렌더러
│       └── geometry.py           # 최적 사다리꼴 ROI 계산, 다항식 좌표 평가, IoU 산출
├── 📁 tests/
│   ├── test_lane.py              # 차선 검출 및 트래커 단위 테스트
│   ├── test_detector.py          # 객체 검출/트래커/노면 표시 단위 테스트
│   └── test_state.py             # 상태 머신 및 의사결정 통합 단위 테스트
├── 📁 data/                      # 입력 동영상 (Test_video_1.mp4, Test_Video_2.mp4)
├── 📁 models/                    # YOLO 모델 가중치 (.pt, .onnx)
├── 📁 outputs/                   # 결과 영상 (.mp4), 로그 (.csv), 요약 보고서 (.json)
├── requirements.txt
└── README.md
```

---

## 2. 주요 핵심 알고리즘 및 구현 특징

### ① 최적화된 ROI 설계 및 프레임 전처리 (차량 보닛 & OSD 자막 배제)
- **상단 (약 0.65h)**: 앞차 상체, 횡단보도 보행자 및 상단 배경 노이즈를 제외하는 차선 영역의 상단 경계로 사용
- **하단 (약 0.88h)**: 영상 최하단의 차량 보닛 반사 및 OSD 자막 에지 원천 배제
- **좌우 (약 0.18w ~ 0.82w)**: 광각 렌즈 왜곡이나 보도블록 노이즈를 줄이기 위한 EM 차선 ROI 경계로 사용
- 공통 전처리는 `gray`, `CLAHE enhanced`, `Gaussian Blur + Canny edges`를 한 번 생성하며, quality/motion과 정지선 검출 및 EM 차선 어댑터가 이를 재사용합니다.
- `QualityAnalyzer`는 평균 밝기로 야간 여부를 판별하고, Grayscale Laplacian variance로 블러 여부를 계산합니다.

### ② 야간 저조도 및 조명 불균형 대응 전처리
- **CLAHE (대비 제한 적응형 히스토그램 평활화)** 적용으로 어두운 노면과 차선의 국소 대비 증폭
- **Auto-Canny 동적 임계값**: ROI 내부 픽셀 중앙값(Median)을 기반으로 조명 변화에 맞추어 `low/high` 임계치 자동 튜닝
- **Morphology Closing**: 점선 차선의 끊어진 에지를 매끄럽게 연결

### ③ 차선 검출기 선택 및 차선 가림(Occlusion) 복구 (Fail-Safe)
- `lane_line` 모드와 `lanes_detector_v1_em` 모드는 `lane.detector` 설정 또는 `--lane-detector` CLI 옵션으로 선택합니다.
- 공통 파이프라인은 두 모드의 좌우 차선 피팅값을 동일한 `LaneTracker` 인터페이스로 전달합니다.
- EM 모드는 메인 전처리의 `gray/enhanced/edges`를 재사용하고, 사다리꼴 ROI와 EMA 기반 추적기로 차선을 검출합니다.
- 선택된 차선 모듈의 `draw_lane_hud()`가 visualizer에서 호출되어 차선 통로, 좌우 차선, 중앙 점선을 렌더링합니다.
- **차로폭 메모리 학습**: 좌·우 차선이 모두 보일 때 차로폭($W_{lane}$)을 학습하여 저장
- **편측 차선 가림 대응**: 왼쪽 또는 오른쪽 차선 하나만 검출되더라도 반대편 차선을 합성 투영하여 중앙 주행 경로 유지
- **시간적 예측 및 상실 관리**: 완전 미검출 시 Kalman 예측값을 유지하되, `max_lost_frames`(기본 30프레임) 초과 시 차선을 `LOST` 상태로 전환
- 횡단보도 감지 시에는 신규 차선 측정을 제외하고 기존 추적 결과를 유지하여 가로 줄무늬의 차선 오검출을 줄입니다.

### ④ 초경량 객체 검출 및 선행 차량(Lead Vehicle) 분석
- **인터벌 추론 (Interval Inference)**: 기본 설정에서는 5프레임마다 YOLO 추론을 수행하고, 중간 프레임은 최근 검출 캐시와 객체 트랙 예측값을 사용합니다.
- GPU 기본 추론 크기는 `640`, CPU는 `480`이며 CUDA FP16 warmup과 LAB L채널 기반 야간 보정을 지원합니다.
- 객체 추적은 IoU와 중심점 거리로 매칭하며, 면적 변화로 `APPROACHING/RECEDING`, 트랙 속도로 `STOPPED/MOVING`을 판정합니다.
- **선두 차량 판별**: 차선 중심과 가까우면서 화면 하단에 위치한 차량/버스/트럭을 선두 차량 후보로 자동 선택하고, Bounding Box 면적 변화와 트랙 속도로 접근/후퇴/정지 상태를 분석합니다.

### ⑤ 배경 Optical Flow 기반 자차 상태 분석
- 차량 객체 영역을 마스킹하고 도로 배경 특징점에 대해서만 **Lucas-Kanade PyrLK Optical Flow**를 계산합니다.
- 유효 특징점 이동량의 중앙값과 이동거리 중앙값을 사용하여 노면 반사 같은 일부 이상치를 완화합니다.
- 특징점이 부족하면 신뢰도를 낮추고 추적점을 재생성합니다. 현재 구현의 자차 상태는 다음과 같습니다.
  - `NORMAL_DRIVING`: 배경 이동량이 정지 임계값 이상이고 블러 상태가 아닌 경우
  - `STOPPED_SUSPECTED`: 배경 이동량이 설정 임계값 미만인 경우
  - `UNCERTAIN`: Optical Flow 신뢰도 저하 또는 영상 블러로 판단을 보류한 경우
- 현재 소스에는 `DECELERATION_SUSPECTED`와 `START_DETECTED` 상태를 별도로 산출하는 로직이 없습니다.

### ⑥ 신호등 색상 판별 및 fallback
- YOLO에서 검출된 `traffic light` 박스의 HSV 영역을 먼저 분류하고, 밝은 광고판이나 후미등 때문에 색상 분류에 실패한 경우 주변 ROI 기반 컨투어 검출을 실행합니다.
- fallback은 원형도, 어두운 housing 주변 밝기, 최소 면적과 빨강/녹색 점유율을 확인하여 가짜 후보를 줄입니다.
- 후보 색상은 설정된 디바운스 프레임 수만큼 안정된 경우에만 신호등 상태와 HUD에 반영됩니다.

### ⑦ 상황 판단 통합 엔진 (Decision Fusion)
- 자차 움직임, 영상 품질, 차선 및 차선 변경, 선행차, 보행자, 정지선, 횡단보도, 신호등 정보를 융합합니다.
- 횡단보도 위 또는 차로 내 근접 보행자, 적색 신호등은 즉시 `STOP_REQUIRED`로 전환합니다.
- 그 외 경고 상태에는 디바운스와 최소 홀드 조건을 적용하여 깜빡임을 줄이고, 판단 근거와 신뢰도도 최종 상태에 포함합니다.
- 정지선과 횡단보도 지속 카운터를 사용하며 최종 융합 결과는 `NORMAL`, `SLOW_DOWN`, `STOP_REQUIRED`, `STOPPED`, `LANE_CHANGE`, `UNCERTAIN` 등이 될 수 있습니다.

### ⑧ Optical Flow 및 객체 영역 배제
- Lucas-Kanade PyrLK Optical Flow는 추적 중인 객체 Bounding Box 내부의 특징점을 제외한 도로 배경 점만 사용합니다.
- 유효 특징점의 `median_dx`, `median_dy`, 이동거리 중앙값을 계산하며, 추적점이 부족하거나 흐름 계산이 실패하면 신뢰도를 낮춥니다.
- 특징점이 다시 부족해지는 경우 주기적으로 `goodFeaturesToTrack`으로 추적점을 재생성합니다.
- 현재 구현의 자차 상태는 `NORMAL_DRIVING`, `STOPPED_SUSPECTED`, `UNCERTAIN`입니다. 기존 설명의 `DECELERATION_SUSPECTED`, `START_DETECTED`는 현재 판단 입력 값이 아니므로 별도 상태로 출력하지 않습니다.

### ⑨ 시각화, 추적 책임 및 성능 측정
- 공통 visualizer는 객체, 정지선, 횡단보도, 신호등, 주행 상태 HUD와 선택적 edge minimap을 합성합니다.
- 차선 검출 자체와 차선 렌더링을 분리하여, 선택된 차선 모듈의 `draw_lane_hud()`가 visualizer 내부에서 호출됩니다.
- 중간 프레임의 YOLO 객체는 다음 검출 결과로 갱신될 때까지 최신 캐시와 트랙 예측을 유지합니다.
- 프레임별 전처리, 차선, 검출, 추적, 노면 표시, 모션, 판단, 렌더링 시간을 `FrameTimings`에 기록합니다.
- 종료 시 평균/p95 지연시간, 실효 FPS, 차선 상태 분포, 객체 수, 주행 상태 분포를 `summary.json`으로 집계합니다.

---

## 3. 설치 및 실행 방법

### 1. 환경 준비
```bash
# 가상환경 생성 (권장)
python -m venv .venv
.venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```
*(선택사항) GPU 또는 최신 YOLO 가중치를 활용하려면 `pip install ultralytics`를 설치하면 자동으로 YOLO 백엔드가 활성화됩니다.*

### 2 파이프라인 실행
```bash
# 기본 설정으로 실행 (Test_Video_2.mp4 대상)
python src/main.py

# CLI 옵션으로 입력 영상 및 설정 변경
python src/main.py --input data/Test_video_1.mp4 --width 960 --detect-interval 5

# 실시간 GUI 창 활성화 (q: 종료, p: 일시정지, d: 미니맵 토글)
python src/main.py --display

# 빠른 벤치마크 테스트 (앞쪽 100프레임만 처리)
python src/main.py --max-frames 100 --no-display

# 차선 검출 알고리즘 선탯
python src/main.py --lane-detector lane_line
python src/main.py --lane-detector lanes_detector_v1_em
```

### 3. 단위 테스트 실행
```bash
python -m unittest discover -s tests
```

---

## 4. 생성되는 최종 결과물

1. **`outputs/annotated_video.mp4`**:
   - 반투명 주행 코리더(안전: 초록 / 감속: 주황 / 위험: 빨강)
   - 좌/우 차선 및 중앙 대표 경로선
   - 선행 차량 강조 바운딩 박스 및 보행자 박스
   - 횡단보도 및 정지선 표시
   - 실시간 자차/선행차 상태 HUD 대시보드 카드
   - 우측 상단 ROI 에지 미니맵
2. **`outputs/frame_log.csv`**:
   - `frame_id, timestamp_sec, lane_status, lane_confidence, lane_error_px, object_count, lead_vehicle_id, lead_state, ego_driving_state, fusion_decision, decision_reason, crosswalk_detected, stopline_detected, optical_flow_magnitude`
3. **`outputs/performance.csv`**:
   - 프레임별 `decode_ms, preprocess_ms, lane_ms, detect_ms, tracking_ms, motion_ms, decision_ms, render_ms, total_ms, detection_updated`
4. **`outputs/summary.json`**:
   - 총 처리 프레임, 평균/p95 처리 지연시간, 실효 FPS
   - 유효 차선 검출률 및 객체 감지 통계
   - 주행 상태별 구간 지속 시간 및 백분율 통계


## 5. 경량화 비교 실험 가이드 (계획서 제8절)

| 실험 조건 | 실행 명령어 | 중점 측정 지표 |
|---|---|---|
| **기준 모델 (1프레임 간격)** | `python src/main.py --detect-interval 1` | 최고 객체 검출률, 지연 시간 (ms) |
| **경량화 (5프레임 간격)** | `python src/main.py --detect-interval 5` | 실시간 40+ FPS, 선행차 ID 유지율 |
| **극단적 경량화 (10프레임)** | `python src/main.py --detect-interval 10` | 60+ FPS, 객체 최초 감지 지연 평가 |
| **해상도 비교 (640 vs 960)** | `python src/main.py --width 640` | 연산량 50% 절감 vs 원거리 보행자 검출률 |
