# 퍼스널컬러 추정 API (올리브송)

웹프로젝트 **올리브송**의 퍼스널컬러 페이지에 연결한 AI 서버입니다.
얼굴 사진을 보내면 봄웜·여름쿨·가을웜·겨울쿨 중 하나를 추정해서 돌려줍니다.

- 주소: https://personal-color-api-nqrx.onrender.com
- 직접 써 보기: https://personal-color-api-nqrx.onrender.com/docs

> 수업 과제용 시제품입니다. 정확도는 검증하지 않았고, 실제 퍼스널컬러 진단(드레이핑, 전문가 진단)을 대신하지 않습니다.

## 어떻게 동작하나요

1. **조명 보정**: 사진 전체의 색 치우침을 약하게 줄입니다(Shades of Gray 방식, 보정 폭 제한).
2. **피부 영역 찾기**: MediaPipe Face Mesh로 얼굴 점을 찾고, 이마와 양 볼 영역만 잘라냅니다.
3. **색 특징 계산**: 그 영역의 Lab, HSV 값(중앙값)과 눈썹 밝기를 숫자로 뽑습니다.
4. **계절 예측**: 코랩에서 사진 90장으로 학습한 SVM 모델(`model.joblib`)이 4계절 중 하나를 고릅니다.
5. **참고 설명**: 규칙 기반으로 사진의 밝기, 노란 기, 피부와 눈썹의 명도 차이, 사진 품질 문제를 함께 알려 줍니다.

학습할 때와 예측할 때 같은 조명 보정과 같은 특징 계산을 씁니다.

## 사용법

`POST /predict`에 사진 파일을 `file`이라는 이름으로 보냅니다.

```bash
curl -X POST -F "file=@face.jpg" https://personal-color-api-nqrx.onrender.com/predict
```

- JPG, PNG, WebP, BMP, 10MB 이하
- 정면 얼굴이 보여야 합니다. 얼굴을 못 찾으면 422 오류를 돌려줍니다.
- 결과에는 `season_ko`(예: 봄웜), 두 번째 후보 `strongest_alternative`, 사진 품질 메모 `image_issue` 등이 들어 있습니다.

그 밖의 주소

| 주소 | 내용 |
|---|---|
| `GET /` | 서버 정보 |
| `GET /health` | 모델이 불러와졌는지 확인 |
| `GET /docs` | 브라우저에서 바로 테스트하는 화면 |

## 파일 구성

| 파일 | 역할 |
|---|---|
| `app.py` | FastAPI 서버 (`/predict` 등) |
| `color_features.py` | 조명 보정, 얼굴 피부 영역 찾기, 색 특징 계산 |
| `free_classifier.py` | 규칙 기반 참고 설명, 사진 품질 확인 |
| `model.joblib` | 코랩에서 학습한 SVM 모델 |
| `four_season_prompt.txt` | 사진 한 장을 4계절로 나누는 기준을 적은 AI 프롬프트(참고용) |
| `requirements.txt` | 필요한 패키지 |

## 내 컴퓨터에서 실행하기

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

실행 후 http://127.0.0.1:8000/docs 에서 테스트할 수 있습니다.

## 한계

- 학습 사진 90장에는 AI와 수동으로 붙인 **임시 라벨**을 썼습니다. 정답이 검증된 데이터가 아닙니다.
- 사진의 조명, 카메라 화이트밸런스, 화장에 따라 결과가 크게 바뀔 수 있습니다.
- 학습에 쓴 사진은 이 저장소에 올리지 않았습니다.
