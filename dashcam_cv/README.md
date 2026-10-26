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

### ① 최적화된 ROI 설계 (차량 보닛 & OSD 자막 배제)
- **상단 (0.65h)**: 타임스퀘어 교차로 등 정차 시 앞차(버스/택시) 번호판 및 횡단보도 보행자 상체 노이즈 차단
- **하단 (0.88h)**: 영상 최하단의 차량 보닛 반사 및 깜빡이는 OSD 자막(속도, 센서값) 에지 원천 배제
- **좌우 (0.18w ~ 0.82w)**: 광각 렌즈 왜곡으로 휘어지는 인도 상점 및 보도블록 노이즈 차단

### ② 야간 저조도 및 조명 불균형 대응 전처리
- **CLAHE (대비 제한 적응형 히스토그램 평활화)** 적용으로 어두운 노면과 차선의 국소 대비 증폭
- **Auto-Canny 동적 임계값**: ROI 내부 픽셀 중앙값(Median)을 기반으로 조명 변화에 맞추어 `low/high` 임계치 자동 튜닝
- **Morphology Closing**: 점선 차선의 끊어진 에지를 매끄럽게 연결

### ③ 차선 가림(Occlusion) 및 대표 차로 복구 (Fail-Safe)
- **차로폭 메모리 학습**: 좌·우 차선이 모두 보일 때 차로폭($W_{lane}$)을 학습하여 저장
- **편측 차선 가림 대응**: 왼쪽 또는 오른쪽 차선 하나만 검출되더라도 반대편 차선을 합성 투영하여 중앙 주행 경로 유지
- **시간적 예측 및 상실 관리**: 완전 미검출 시 직전 경로를 EMA 스무딩하여 유지하되, `max_lost_frames`(기본 30프레임 ~ 1초) 초과 시 가짜 차선을 그리지 않고 `LOST`로 전환

### ④ 초경량 객체 검출 및 선행 차량(Lead Vehicle) 분석
- **인터벌 추론 (Interval Inference)**: YOLO 추론을 매 프레임 돌리지 않고 5프레임마다 1회 수행, 중간 프레임은 트래커가 보간하여 CPU 환경에서도 45+ FPS 달성
- **선행 차량 판별**: 자차 주행 차선 내부에서 가장 가까운 전방 차량을 자동 선별하고, 바운딩 박스 크기 변화($\Delta Area$) 및 $y$좌표 변화율로 `APPROACHING`, `STOPPED`, `MOVING` 분석

### ⑤ 배경 광학 흐름 기반 주행 상태 머신 (Section 6 & image-3 준수)
- 차량 객체 영역을 마스킹하고 도로 배경 특징점에 대해서만 **Lucas-Kanade Optical Flow**를 계산
- 특이치(노면 반사, 와이퍼 등)의 영향을 배제하기 위해 이동량에 **중앙값(Median)** 적용
- **히스테리시스 상태 머신**:
  - `NORMAL_DRIVING`: 유의미한 배경 이동량 지속
  - `DECELERATION_SUSPECTED`: 시계열 창(25프레임) 내 이동량 감소 추세 감지
  - `STOPPED_SUSPECTED`: 이동량 0.85px 미만 상태가 15프레임 이상 유지
  - `START_DETECTED`: 정지 상태에서 이동량 급증 감지

### ⑥ 상황 판단 통합 엔진 (Decision Fusion)
- 자차 상태, 선행차 상태, 보행자 근접도, 정지선 및 횡단보도를 융합하여 `NORMAL`, `SLOW_DOWN`, `STOP_REQUIRED`, `STOPPED`, `LANE_CHANGE` 결정

---

## 3. 설치 및 실행 방법

### 1) 환경 준비
```bash
# 가상환경 생성 (권장)
python -m venv .venv
.venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```
*(선택사항) GPU 또는 최신 YOLO 가중치를 활용하려면 `pip install ultralytics`를 설치하면 자동으로 YOLO 백엔드가 활성화됩니다.*

### 2) 파이프라인 실행
```bash
# 기본 설정으로 실행 (Test_Video_2.mp4 대상)
python src/main.py

# CLI 옵션으로 입력 영상 및 설정 변경
python src/main.py --input data/Test_video_1.mp4 --width 960 --detect-interval 5

# 실시간 GUI 창 활성화 (q: 종료, p: 일시정지, d: 미니맵 토글)
python src/main.py --display

# 빠른 벤치마크 테스트 (앞쪽 100프레임만 처리)
python src/main.py --max-frames 100 --no-display
```

### 3) 단위 테스트 실행
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

---

## 5. 경량화 비교 실험 가이드 (계획서 제8절)

| 실험 조건 | 실행 명령어 | 중점 측정 지표 |
|---|---|---|
| **기준 모델 (1프레임 간격)** | `python src/main.py --detect-interval 1` | 최고 객체 검출률, 지연 시간 (ms) |
| **경량화 (5프레임 간격)** | `python src/main.py --detect-interval 5` | 실시간 40+ FPS, 선행차 ID 유지율 |
| **극단적 경량화 (10프레임)** | `python src/main.py --detect-interval 10` | 60+ FPS, 객체 최초 감지 지연 평가 |
| **해상도 비교 (640 vs 960)** | `python src/main.py --width 640` | 연산량 50% 절감 vs 원거리 보행자 검출률 |
