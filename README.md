# ROI[README.md](https://github.com/user-attachments/files/33016252/README.md)
# CCTV 영상 ROI 기반 사람 침입 감지 시스템

CCTV 영상에서 사람이 지정 구역(ROI)에 **완전히 들어오고 나가는 순간**을 자동으로 판정해
경보(Alarm) 시각을 기록하는 시스템입니다. (세종대학교 컴퓨터비전 과제)

## 판정 조건
1. **침입(ON)** : 사람이 ROI에 완전히 들어간 뒤 1초가 경과한 시각
2. **종료(OFF)** : ROI에 들어갔던 사람의 몸이 완전히 빠져나온 시각
3. **오탐 배제** : 일부만 걸친 사람이나 사람이 아닌 물체에는 반응하지 않음

## 파이프라인
```
프레임 입력 → ① 사람 검출 → ② 오탐 필터 → ③ 추적 → ④ ROI 진입 판정 → ⑤ 이벤트 생성 → 결과 출력
```
| 단계 | 내용 |
|------|------|
| ① 사람 검출 | YOLOv8n(person) 검출, 야간 영상은 CLAHE 전처리 |
| ② 오탐 필터 | (야간) 위치 분산으로 정지 물체(간판·조명) 제거 |
| ③ 추적 | 최근접 매칭 + coast(끊김 보완) |
| ④ ROI 진입 판정 | 박스 15점 샘플링 → ROI 포함 비율 85% 이상 |
| ⑤ 이벤트 생성 | 진입 1초 유지 ON / 완전 이탈 OFF (히스테리시스) |

## 사용 기술
- **언어** : Python
- **라이브러리** : OpenCV, NumPy, Ultralytics(YOLOv8)
- **검출 모델** : YOLOv8n (COCO 사전학습)

## 실행 방법
```bash
pip install -r requirements.txt
python roi_intrusion_detection.py
```
- 최초 실행 시 `yolov8n.pt`가 자동 다운로드됩니다.
- 같은 폴더에 입력 영상(`test_*.mp4`)과 `cv_hw2_mp4script.txt`가 있어야 합니다.
- 결과 영상은 `Result/` 폴더에, 경보 로그는 `event_detection_result.txt`에 저장됩니다.

## 입력 형식 (cv_hw2_mp4script.txt)
```
test_snowhouse.mp4
<Point>430,160</Point>
<Point>673,156</Point>
...
```
영상 파일명과 ROI 다각형 꼭짓점 좌표를 나열합니다.

## 결과
| 영상 | Alarm 이벤트 |
|------|-------------|
| test_snowhouse | ON 00:31 / OFF 00:33 |
| test_street | ON 00:38 / OFF 00:43 |
| test_snownight | ON 00:39 / OFF 00:42, ON 00:48 / OFF 00:50, ON 00:59 / OFF 01:02 |

## 개발 과정에서의 주요 선택
- **검출 모델** : HOG → MobileNet-SSD → **YOLOv8** (야간·소형 객체 검출 안정성)
- **야간 오탐** : 화환·간판이 사람으로 오인되는 문제를 **위치 분산 필터**로 해결
- **ROI 판정** : 중심 1점 → 발 기준 → **15점 포함 비율**(몸 전체 진입 판정)
- **경보 안정화** : 단일 임계 → **히스테리시스**(진입 1초 / 이탈 0.5초)로 깜빡임 제거

## 참고
입력 영상(test_*.mp4)과 결과 영상은 용량이 커 저장소에 포함하지 않았습니다.
