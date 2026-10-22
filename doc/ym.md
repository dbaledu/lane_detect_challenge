## 객체 추적(Object Tracking)

> `객체 검출(Object Detection)`: 현재 프레임에 자동차가 있는가?  
> `객체 추적(Object Tracking)`:이전 프레임의 자동차와 현재 프레임의 자동차가 같은 자동차인가?

- Centroid(중심점)와 IoU(박스겹침 비율)가 객체를 직접 검출하는 모델은 아닙니다. YOLO나 다른 검출기가 제공한 **바운딩 박스를 프레임 간에 연결**하는 역할
- Centroid와 IoU를 이용하면, 무거운 딥러닝 추적 모델 없이도 차량·보행자·자전거 등의 객체를 프레임 간에 연결하여 추적

![alt text](images/detect03.png)

![alt text](images/detect02.png)

![alt text](./images/detect01.png)

#### 경량 트래커의 전체 처리 흐름
> `IoU 함수 구현` → `Centroid 계산` → 간단한 `ID 매칭` → `객체 미검출 처리` → YOLO `검출 결과 연결` → 차량 이동 및 정지 판단

![alt text](images/detect04.png)

### Centroid 기반 추적
Centroid는 객체를 감싸는 바운딩 박스의 중심점입니다.
![alt text](images/detect05.png)

예시)
(100, 200), 현재 중심점이 (108, 204)라면,  
$d=\sqrt{8^2+4^2}\approx8.94$  
거리가 설정한 임계값보다 작으면 같은 객체로 판단할 후보가 됩니다.  

**장점**: 계산이 매우 빠르고 구현이 쉽습니다.
**단점**: 차량 두 대가 가까이 지나가거나 객체가 서로 교차하면 ID가 바뀔 수 있습니다. 객체의 크기나 모양 변화도 충분히 반영하지 못합니다.

#### IoU 기반 추적
IoU는 Intersection over Union의 약자이며, 두 바운딩 박스가 얼마나 겹치는지 측정합니다.  

![alt text](images/detect06.png)  
![alt text](images/detect07.png)

#### Centroid와 IoU를 함께 사용하는 이유

두 방법은 서로 보완적입니다.

|비교 항목|Centroid|IoU|
|---|---|---|
|비교 대상|중심점 사이 거리|박스 겹침 비율|
|객체 이동|중심점 이동으로 판단|박스의 겹침으로 판단|
|크기 변화 반영|직접 반영하지 않음|어느 정도 반영|
|객체가 빠르게 이동|거리가 멀어지면 연결 실패 가능|박스가 겹치지 않으면 연결 실패 가능|
|계산 비용|매우 낮음|낮음|
|주요 문제|가까운 객체 간 ID 혼동|겹치지 않는 박스의 연결 어려움|

실제 경량 추적기에서는 두 조건을 조합해 동일 객체 여부를 판단할 수 있습니다.
예를 들어 다음과 같은 규칙입니다.

- 중심점 거리가 충분히 가까움
- IoU가 일정 수준 이상임
- 두 조건을 종합한 점수가 가장 좋은 기존 ID를 선택함

다만, 중심점 거리와 IoU를 반드시 동시에 만족해야 하는 것은 아닙니다. 빠르게 이동하는 객체는 IoU가 낮더라도 중심점이 가까울 수 있고, 크기가 큰 객체는 중심점 이동이 있어도 박스가 겹칠 수 있습니다.



## Object_tracker.py 소스 분석

#### 전체 실행 흐름
1. `RoadMarkingDetector.detect(frame, enhanced, edges)`
    - `_detect_crosswalk()` ROI 내 밝은 가로 줄무늬를 이용해 횡단보도를, 
    - `_detect_stopline()` Hough 선분을 이용해 정지선을 검출합니다. 

2. `ObjectDetector.detect(frame, frame_idx)`
    - 설정된 검출 간격에 맞는 프레임이면 검출을 실행하고, 
    - 그렇지 않으면 직전 검출 결과를 재사용합니다. 
    - YOLO 백엔드이면 `_detect_yolo()`를, 그 외에는 `_detect_opencv_fallback()`을 호출합니다. 
    - 반환값은 검출 객체와 이번 프레임에 실제 추론을 수행했는지 여부입니다.

3. `ObjectTracker.update(...)`가 새 검출 결과를 기존 추적과 연결합니다.
    - 검출을 건너뛴 프레임에는 기존 객체 위치를 예측하고, 객체별 이동 상태·거리 추세를 계산합니다. 
    - 차로 중앙 앞쪽의 차량은 선행 차량으로 표시하고, 
    - 횡단보도가 검출된 경우 그 영역과 겹치는 보행자를 표시합니다.

#### 세부 내용

###### ObjectBox 데이터 객체에는 추적 중인 객체의 정보를 에 담습니다.

```python
ObjectBox(
    track_id=tid,
    bbox=track.bbox,
    class_id=track.class_id,
    label=track.label,
    confidence=track.confidence,
    center=track.center,
    velocity=track.velocity,
    is_lead=False,
    relative_distance_trend=trend,
    status=status
)
```

- `track_id`=tid: 추적기가 객체에 부여한 고유 ID입니다. 프레임이 바뀌어도 같은 객체를 구분하는 데 사용합니다.
- `bbox`=track.bbox: 객체를 감싸는 바운딩 박스 좌표 (x1, y1, x2, y2)입니다. 왼쪽 위와 오른쪽 아래 좌표를 뜻합니다.
- `class_id`=track.class_id: 모델이 분류한 객체 클래스의 숫자 ID입니다. 예를 들어 COCO 기준 2는 자동차입니다.
- `label`=track.label: 클래스 이름입니다. 예: "car", "person".
- `confidence`=track.confidence: 객체 검출에 대한 모델의 신뢰도 점수입니다. 보통 0~1 범위입니다.
- `center`=track.center: 바운딩 박스의 중심 좌표 (x, y)입니다.
- `velocity`=track.velocity: 추적 기록을 바탕으로 계산한 화면 좌표상의 이동 속도/방향 벡터입니다. 실제 도로 속도 단위(예: km/h)는 아닙니다.
- `is_lead`=False: 생성 시점에는 선행 차량이 아니라고 초기화합니다. 이후 차로 중앙 부근의 차량 중 조건에 맞는 후보가 선택되면 True로 바뀝니다.
- `relative_distance_trend`=trend: 객체 크기의 시간에 따른 변화를 이용한 상대 거리 추세입니다. 
    - "APPROACHING"은 가까워지는 것으로 판단, 
    - "RECEDING"은 멀어지는 것으로 판단, 
    - "STABLE"은 큰 변화가 없는 경우입니다. 
    - 충분한 추적 이력이 없으면 초기값 "UNKNOWN"입니다.
- `status`=status: 화면상 움직임 상태입니다. 차량의 실제 정차 여부를 직접 측정한 값은 아닙니다.
    - 추적기의 속도 크기가 기준보다 작으면 "STOPPED", 
    - 그렇지 않으면 "MOVING"입니다.
- `is_on_crosswalk`는 이 생성 코드에서 전달하지 않지만, ObjectBox의 기본값 False가 적용됩니다. 이후 객체가 횡단보도 위로 판별되면 별도로 True로 표시됩니다.



###### update 함수

```python
    def update(
        self,
        detections: List[ObjectBox],        # 검출 결과
        detection_executed: bool,           # 이번 프레임에서 검출 실행 여부
        lane: Optional[LaneResult],         # 차선 정보
        frame_shape: Tuple[int, int, int],  # 프레임 크기
        road_markings: Optional[Any] = None # 도로 표시 정보
    ) -> List[ObjectBox]:
```


```
함수 update(검출 결과, 이번 프레임에서 검출 실행 여부, 차선 정보, 프레임 크기, 도로 표시 정보):

    프레임 높이와 너비를 가져온다
    차선 중앙을 정한다
        차선 중앙 정보가 있으면 그 값을 사용
        없으면 화면 중앙을 사용

    만약 이번 프레임에서 객체 검출을 실행했다면:
        매칭된 트랙 ID 목록과 새 검출 객체 목록을 준비한다

        각 검출 객체에 대해:
            기존 트랙 중 아직 매칭되지 않은 것을 찾는다
            객체 라벨 호환성과 박스 겹침 정도(IoU), 중심 거리로 매칭 점수를 계산한다

            조건에 맞는 기존 트랙이 있으면:
                트랙 정보를 새 검출 결과로 갱신한다
                트랙 ID를 매칭 완료로 표시한다
            아니면:
                새 검출 객체 목록에 추가한다

        매칭되지 않은 기존 트랙은 위치를 예측하고 미검출 횟수를 늘린다
        미검출 횟수가 허용 한도를 넘은 트랙은 제거한다

        매칭되지 않은 새 검출 객체마다 새 트랙 ID를 발급해 등록한다

    아니면 (이번 프레임에서 검출을 건너뛰었다면):
        모든 트랙의 위치를 속도로 예측하고 미검출 횟수를 늘린다
        허용 한도를 넘은 트랙은 제거한다

    추적 중인 트랙마다:
        과거와 현재 바운딩 박스 크기를 비교해 접근/후퇴/안정 추세를 계산한다
        속도가 기준보다 작으면 정지, 아니면 이동 상태로 정한다
        트랙 정보를 ObjectBox로 만들어 결과 목록에 추가한다

        트럭·버스·자동차이고,
        차선 중앙에서 설정된 거리 이내이며, (settings.yaml의 lead_vehicle_corridor_margin_px에서 설정하며, 기본값은 130px)
        화면 높이의 절반보다 아래에 있으면:
            선행 차량 후보로 기록한다
            후보 중 바운딩 박스의 아래쪽이 가장 아래에 있는 차량을 선택한다

    선택된 후보의 is_lead를 True로 표시한다

    횡단보도 검출 정보가 있으면:
        각 사람 객체의 박스가 횡단보도 박스와 겹치는지 확인한다
        겹치면 is_on_crosswalk를 True로 표시한다

    추적 객체 목록을 반환한다
```


#### common/visualizer.py/draw 함수

현재 프레임과 분석 결과들을 받아, 정보를 그려 넣은 영상 프레임을 반환하는 함수입니다.

```python
    def draw(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        lane: LaneResult,
        objects: List[ObjectBox],
        markings: RoadMarkingResult,
        motion: MotionResult,
        state: DrivingStateResult,
        timings: FrameTimings,
        quality: QualityMetric,
        debug_edge_map: Optional[np.ndarray] = None
    ) -> np.ndarray
```

- `self`: Visualizer 객체 자신입니다. 설정이나 시각화 옵션에 접근할 때 사용합니다.
- `frame`: np.ndarray: 입력 영상 프레임입니다. 보통 OpenCV 형식의 이미지 배열(BGR)이며, 이 프레임을 복사해 시각화 결과를 그립니다.
- `frame_idx`: int: 현재 프레임의 번호입니다. HUD 등에 표시하거나 프레임을 식별하는 데 사용합니다.
- `timestamp_sec`: float: 영상 시작부터 현재 프레임까지의 시간(초)입니다. 재생 위치 표시 등에 사용합니다.
- `lane`: LaneResult: 차선 검출 결과입니다. 차선 경계, 주행 통로, 차선 상태 등을 그리는 데 사용합니다.
- `objects`: List[ObjectBox]: 추적된 객체 목록입니다. 차량·사람의 바운딩 박스와 라벨, 선행 차량 여부 등을 표시합니다.
- `markings`: RoadMarkingResult: 횡단보도와 정지선 검출 결과입니다. 해당 영역과 라벨을 프레임에 그리는 데 사용합니다.
- `motion`: MotionResult: 차량의 움직임 분석 결과입니다. 화면의 움직임·정차 관련 정보를 표시할 때 사용합니다.
- `state`: DrivingStateResult: 현재 주행 상태와 융합 판단 결과입니다. 예를 들어 감속·정지 같은 상태를 HUD에 표시합니다.
- `timings`: FrameTimings: 프레임 처리 단계별 시간 정보입니다. 처리 성능을 화면에 표시할 때 사용합니다.
- `quality`: QualityMetric: 프레임 품질 분석 결과입니다. 밝기나 흐림 여부 등을 표시하는 데 사용합니다.
- `debug_edge_map`: Optional[np.ndarray] = None: 선택적인 디버그용 에지 맵입니다. 전달하면 디버그 화면에 활용하고, 생략하거나 None이면 에지 맵 없이 시각화합니다.
- `반환값` -> np.ndarray: 시각화 요소가 그려진 출력 프레임입니다. 원본 frame을 직접 수정하는 대신 복사본에 그려 반환합니다.


## 성능 분석
```python
 "performance": {
    "processed_frames": 678,
    "effective_processing_fps": 46.27,   # 처리한 프레임 하나당 평균 지연시간을 밀리초(ms)
    "mean_total_ms": 21.61,
    "p95_total_ms": 45.14,
    "module_latency_mean_ms": {   # 처리한 프레임 하나당 평균 지연시간(ms)
      "decode": 0.0,
      "preprocess": 7.13,
      "lane": 2.18,
      "detect": 7.55,
      "tracking": 0.15,
      "motion": 2.28,
      "decision": 0.05,
      "render": 1.19
    }
```


. `module_latency_mean_ms`의 각 수치는 **처리한 프레임 하나당 평균 지연시간을 밀리초(ms)로 나타낸 것**입니다. `mean_total_ms`와 `p95_total_ms`도 프레임 처리시간이며, `effective_processing_fps`만 **초당 처리 프레임 수(FPS)**입니다.

|항목|값|의미|
|---|--:|---|
|`processed_frames`|678|실제 처리한 프레임 수|
|`effective_processing_fps`|46.27 FPS|초당 평균 처리 프레임 수|
|`mean_total_ms`|21.61ms|프레임 한 장을 처리하는 데 걸린 평균 시간|
|`p95_total_ms`|45.14ms|처리 프레임의 95%가 이 시간 이내에 완료됨. 나머지 5%는 더 오래 걸림|
|`decode`|0.00ms|영상 프레임 디코딩 시간. 0으로 측정됐거나 반올림·측정 범위에서 제외됐을 수 있어, 디코딩 작업이 실제로 전혀 없었다고 단정할 수는 없음|
|`preprocess`|7.13ms|프레임 크기 조정 등 모델 입력을 위한 전처리|
|`lane`|2.18ms|차선 검출 처리|
|`detect`|7.55ms|객체 검출 처리|
|`tracking`|0.15ms|객체 추적 및 ID 갱신|
|`motion`|2.28ms|움직임 분석|
|`decision`|0.05ms|검출 결과를 바탕으로 한 주행 판단|
|`render`|1.19ms|결과를 프레임 위에 그리는 시각화|

평균 총 처리시간 `21.61ms`를 FPS로 환산하면 약 **46.3 FPS**이므로, `effective_processing_fps`와 거의 일치합니다. 모듈별 평균을 더하면 약 **20.53ms**로 총 시간보다 조금 짧은데, 측정 범위에 차이가 있거나 모듈별 시간에 포함되지 않는 오버헤드가 있을 수 있습니다.


## 표시전용 EMA(Exponential Moving Average) 평활화
- EMA(Exponential Moving Average) : 지수이동평균
자율주행 영상 분석에서 표시전용 EMA 평활화는 객체의 실제 검출 결과는 그대로 유지하면서, 화면에 표시하는 위치나 값을 부드럽게 만드는 기법입니다
- 또한 객체 ID가 바뀌거나 객체가 오랫동안 검출되지 않았다가 다시 나타나는 경우에는 이전 EMA 값을 초기화하거나 재연결 여부를 확인
- 추적기에서 나온 detected_box를 변경하지 않고, display_box만 시각화에 사용

![alt text](images/detect08.png)