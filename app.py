"""No-paid-API four-season image prototype; POST /predict with file image."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import cv2
import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from color_features import (
    PREPROCESS_METHOD, SkinFeatureExtractor, conservative_shades_of_gray,
    resize_for_analysis,
)
from free_classifier import FULL, SEASON_KO, estimate_four_seasons

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).with_name("model.joblib"))))
_lock = Lock()
_extractor = SkinFeatureExtractor()


def _load_trained_model():
    if not MODEL_PATH.is_file():
        return None
    # Never load an untrusted joblib: it can run code when deserialized.
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or not {"model", "feature_names", "class_counts"}.issubset(bundle):
        raise RuntimeError("model.joblib이 올바른 학습 모델 파일이 아닙니다.")
    if bundle.get("preprocessing") != PREPROCESS_METHOD:
        raise RuntimeError("학습 모델의 조명 보정 방식이 API와 다릅니다. 최신 코랩 노트로 다시 90장을 학습하세요.")
    return bundle


_bundle = _load_trained_model()

app = FastAPI(
    title="Personal Color - Free 4-Season Demo",
    description="Trained on the user's 90 weakly labeled photos; prototype only, not a validated diagnosis.",
    version="0.3.0",
)

# For a public demo only; for production set CORS_ORIGINS to the team's site.
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/")
def root():
    return {"service": "personal-color-demo", "docs": "/docs", "ready": _bundle is not None, "paid_api_required": False}


@app.get("/health")
def health():
    return {"status": "ready" if _bundle is not None else "model_missing",
            "method": "trained_svm_plus_heuristic_review", "model_loaded": _bundle is not None,
            "preprocessing": PREPROCESS_METHOD,
            "paid_api_required": False}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _bundle is None:
        raise HTTPException(status_code=503, detail="학습된 model.joblib이 없습니다. 코랩에서 90장 학습 셀을 먼저 실행하세요.")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="JPG/PNG/WebP/BMP 사진만 올려주세요.")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="사진은 10MB 이하만 허용됩니다.")
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")
    image = resize_for_analysis(image)
    preprocessed = conservative_shades_of_gray(image)

    with _lock:
        # EXACTLY the same correction is applied to the 90 training images.
        features = _extractor.extract(preprocessed)
        raw_features = _extractor.extract(image) if features is not None else None

    if features is None:
        raise HTTPException(status_code=422, detail="분석 가능한 얼굴을 찾지 못했습니다. 정면 얼굴 사진이 필요합니다.")

    names = _bundle["feature_names"]
    if any(name not in features for name in names):
        raise HTTPException(status_code=500, detail="학습 모델과 API의 얼굴 특징 이름이 일치하지 않습니다.")
    x = np.array([[features[name] for name in names]], dtype=np.float32)
    if not np.isfinite(x).all():
        raise HTTPException(status_code=422, detail="얼굴 색 특징에 사용할 수 없는 값이 있습니다.")

    season = str(_bundle["model"].predict(x)[0])
    if season not in SEASON_KO:
        raise HTTPException(status_code=500, detail="학습 모델이 유효하지 않은 라벨을 반환했습니다.")

    # Prompt-inspired rule comparison supplies cautious photo observations,
    # but the primary season is predicted by the ACTUALLY TRAINED SVM.
    review = estimate_four_seasons(features, preprocessed, raw_features)
    alternative = (review["strongest_alternative"] if review["season"] == season
                   else FULL[review["season"]])
    return {
        **review,
        "season": season,
        "season_ko": SEASON_KO[season],
        "season_full": FULL[season],
        "strongest_alternative": alternative,
        "status": "experimental_unvalidated",
        "confidence": "low",
        "method": "trained_svm_plus_heuristic_review",
        "preprocessing": PREPROCESS_METHOD,
        "class_counts": _bundle["class_counts"],
        "message": "약한 조명 보정을 동일하게 적용해 90장 AI/수동 임시 라벨로 학습한 SVM 결과입니다. 정확도는 미검증입니다.",
    }
