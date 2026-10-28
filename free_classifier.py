"""Offline, no-fee heuristic comparison of four styling seasons.

It is NOT a personal-color diagnostic model, a literal execution of a ChatGPT
prompt, or an evidence-based prediction of a person's optimal drape colors.
The scores are only internal ranking heuristics, not calibrated probabilities.
"""

from __future__ import annotations

import math
import cv2
import numpy as np

SEASON_KO = {"spring": "봄웜", "summer": "여름쿨", "autumn": "가을웜", "winter": "겨울쿨"}
FULL = {"spring": "spring-warm", "summer": "summer-cool", "autumn": "autumn-warm", "winter": "winter-cool"}


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _image_quality(image: np.ndarray, feat: dict) -> tuple[str, list[str]]:
    """Screen observable technical limitations, without asserting true WB."""
    problems: list[str] = []
    l = float(feat["skin_L"])
    if l < 30:
        problems.append("얼굴 피부 영역이 매우 어둡게 촬영됨")
    elif l > 88:
        problems.append("얼굴 피부 영역이 지나치게 밝게 촬영됨")
    region_l = [float(feat[f"{r}_L"]) for r in ("left_cheek", "right_cheek", "forehead")]
    if max(region_l) - min(region_l) > 15:
        problems.append("얼굴 부위별 밝기 차이가 큼(그림자/혼합 조명 가능)")
    # Image sharpness is about usability only; it is NOT 'clear' season evidence.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, round(w * min(1.0, 800/max(h, w)))), max(1, round(h * min(1.0, 800/max(h, w))))))
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
    if sharpness < 18:
        problems.append("사진이 흐릿할 가능성 있음")
    # Bright clipping and deep shadows are image-exposure issues, not undertone.
    if float(np.mean(gray > 249)) > 0.18:
        problems.append("사진의 밝은 영역 일부가 과다 노출됨")
    if float(np.mean(gray < 10)) > 0.25:
        problems.append("사진의 어두운 영역 일부가 뭉개짐")
    return ("limited" if problems else "usable"), problems


def estimate_four_seasons(feat: dict, image: np.ndarray, corrected_feat: dict | None = None) -> dict:
    """Rank four seasons using weak, explicit image-feature proxies.

    Warm/cool cannot be established from skin pixels alone. A* and B* below
    are *not* calibrated undertone measurements. Eyebrow/skin contrast is a
    weak proxy for photographed facial contrast, not palette suitability.
    Every result is therefore LOW confidence and a tentative forced choice.
    """
    if feat is None or image is None:
        raise ValueError("얼굴을 찾지 못했습니다.")
    a, b, l = (float(feat[f"skin_{k}"]) for k in "abL")
    # Empirical *demonstration* thresholds; NOT clinical/seasonal ground truth.
    temperature_axis = _clamp((b - a - 4.0) / 13.0)
    lightness_axis = _clamp((l - 60.0) / 22.0)
    brow_l = feat.get("brow_L")
    contrast_diff = max(0.0, l - float(brow_l)) if brow_l is not None else None
    contrast_axis = _clamp((contrast_diff - 25.0) / 20.0) if contrast_diff is not None else 0.0

    # Value/depth receives only a small supporting weight. Bright vs muted
    # palette preference cannot actually be observed without a real drape.
    ranking = {
        "spring": temperature_axis + 0.40 * contrast_axis + 0.08 * lightness_axis,
        "summer": -temperature_axis - 0.40 * contrast_axis + 0.08 * lightness_axis,
        "autumn": temperature_axis - 0.40 * contrast_axis - 0.08 * lightness_axis,
        "winter": -temperature_axis + 0.40 * contrast_axis - 0.08 * lightness_axis,
    }
    names = sorted(ranking, key=lambda k: (-ranking[k], k))
    season, alternative = names[:2]
    usability, problems = _image_quality(image, feat)

    wb_sensitive = False
    if corrected_feat is not None:
        ca, cb = float(corrected_feat["skin_a"]), float(corrected_feat["skin_b"])
        corrected_t = _clamp((cb - ca - 4.0) / 13.0)
        # Gray-World-like corrections are a *sensitivity check*, not a
        # calibrated recovery of true skin color.
        wb_sensitive = (abs(corrected_t - temperature_axis) > 0.6 or
                        ((temperature_axis > 0.12) != (corrected_t > 0.12)))
        if wb_sensitive:
            usability = "limited"
            problems.append("자동 색 보정 시 웜/쿨 추정이 크게 바뀜(조명 영향 가능)")

    if abs(temperature_axis) < 0.17:
        temp_text = "불확실: 피부 픽셀만으로 웜/쿨을 나누기 어려움"
    elif temperature_axis > 0:
        temp_text = "사진상 노란색 성분이 상대적으로 강함; 실제 웜톤이라는 뜻은 아님"
    else:
        temp_text = "사진상 노란색 성분이 상대적으로 약함; 실제 쿨톤이라는 뜻은 아님"

    if l >= 70:
        value_text = "사진상 피부 명도가 밝은 편"
    elif l < 47:
        value_text = "사진상 피부 명도가 어두운 편"
    else:
        value_text = "사진상 피부 명도가 중간 범위"

    if contrast_diff is None:
        contrast_text = "확인 불충분: 눈썹 영역 명도를 추출하지 못함"
    elif contrast_axis > 0.20:
        contrast_text = "피부와 눈썹 영역의 사진상 명도 차이가 비교적 큼"
    elif contrast_axis < -0.20:
        contrast_text = "피부와 눈썹 영역의 사진상 명도 차이가 비교적 작음"
    else:
        contrast_text = "피부와 눈썹 영역의 사진상 명도 차이가 중간 정도"

    evidence = [
        "이마·양 볼의 노출된 피부 영역에서 색 성분과 명도를 계산함",
        temp_text,
        contrast_text,
    ]
    message = "한 장의 사진에서 계산한 미검증 스타일링 추정입니다. 실제 드레이핑·전문가 진단을 대체하지 않습니다."
    return {
        "season": season,
        "season_ko": SEASON_KO[season],
        "season_full": FULL[season],
        "status": "tentative_forced_choice",
        "confidence": "low",
        "image_usability": usability,
        "image_issue": "; ".join(problems) if problems else "뚜렷한 촬영 품질 문제는 감지되지 않음(색 정확도 검증 불가)",
        "observed_traits": {
            "apparent_temperature": temp_text,
            "value": value_text,
            "contrast": contrast_text,
            "color_clarity": "불확실: 실제 색상 팔레트를 착용한 반응을 사진에서 관찰할 수 없음",
        },
        "evidence": evidence,
        "strongest_alternative": FULL[alternative],
        "main_unresolved_distinction": "조명·카메라 화이트밸런스와 실제 드레이핑 반응을 확인할 수 없어 네 계절 간 경계가 불확실함",
        "method": "offline_heuristic_unvalidated",
        "message": message,
    }
