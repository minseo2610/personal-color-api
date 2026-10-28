"""Feature extraction shared by the Colab experiment and the demonstration API.

This computes observed image colors, NOT a scientifically validated color season.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


REGIONS = {
    "left_cheek": [50, 101, 119, 100, 142, 203, 206, 205, 187, 123, 116, 117, 118],
    "right_cheek": [280, 330, 348, 329, 371, 423, 426, 425, 411, 352, 345, 346, 347],
    "forehead": [67, 109, 10, 338, 297, 299, 337, 336, 285, 417, 8, 193, 55, 107, 66, 69],
}
ERODE_PX = 7
MAX_SIDE = 1280
PREPROCESS_METHOD = "conservative_shades_of_gray_v1"
CORRECTION_STRENGTH = 0.5
CORRECTION_MAX_GAIN = 0.15


def conservative_shades_of_gray(img: np.ndarray) -> np.ndarray:
    """Apply bounded, *uncalibrated* global color-cast reduction.

    Apply this EXACT transformation for SVM training and API prediction.
    It cannot recover true skin colors or guarantee a better color season.
    """
    pixels = img.astype(np.float32)
    magnitude = (pixels ** 6).reshape(-1, 3).mean(axis=0) ** (1.0 / 6)
    goal = float(magnitude.mean())
    gains = np.clip(goal / (magnitude + 1e-5),
                    1.0 - CORRECTION_MAX_GAIN,
                    1.0 + CORRECTION_MAX_GAIN)
    weak_gains = 1.0 + CORRECTION_STRENGTH * (gains - 1.0)
    return np.clip(pixels * weak_gains, 0, 255).astype(np.uint8)


def resize_for_analysis(img: np.ndarray) -> np.ndarray:
    """Use the same image-size policy in both training and prediction."""
    h, w = img.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return img
    scale = MAX_SIDE / max(h, w)
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def _trim_mask(lightness: np.ndarray, mask: np.ndarray, lo: int = 10, hi: int = 90) -> np.ndarray:
    values = lightness[mask > 0]
    if values.size == 0:
        return mask
    low, high = np.percentile(values, [lo, hi])
    result = mask.copy()
    result[(lightness < low) | (lightness > high)] = 0
    return result


class SkinFeatureExtractor:
    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        )

    def close(self):
        self._face_mesh.close()

    def extract(self, img: np.ndarray) -> dict[str, float] | None:
        if img is None or img.ndim != 3 or img.shape[2] != 3:
            return None
        img = resize_for_analysis(img)
        h, w = img.shape[:2]
        detected = self._face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not detected.multi_face_landmarks:
            return None

        points = np.array(
            [[p.x * w, p.y * h] for p in detected.multi_face_landmarks[0].landmark],
            dtype=np.int32,
        )
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] *= 100.0 / 255.0
        lab[:, :, 1:] -= 128.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX, ERODE_PX))
        features: dict[str, float] = {}
        union = np.zeros((h, w), dtype=bool)

        for name, indices in REGIONS.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(points[indices]), 255)
            mask = cv2.erode(mask, kernel)
            selected = _trim_mask(lab[:, :, 0], mask) > 0
            if int(selected.sum()) < 50:
                return None
            union |= selected
            for channel, name_lab in enumerate("Lab"):
                features[f"{name}_{name_lab}"] = float(np.median(lab[:, :, channel][selected]))

        for channel, name_lab in enumerate("Lab"):
            features[f"skin_{name_lab}"] = float(np.median(lab[:, :, channel][union]))
        for channel, name_hsv in enumerate("HSV"):
            features[f"skin_{name_hsv}"] = float(np.median(hsv[:, :, channel][union]))

        # A limited *photographed* skin–brow contrast indicator, not a proof of
        # suitable bright/muted palettes. Brows can be covered by cosmetics.
        brow_indices = [70, 63, 105, 66, 107, 300, 293, 334, 296, 336]
        brow_mask = np.zeros((h, w), dtype=np.uint8)
        face_width = max(1, int(points[:, 0].max() - points[:, 0].min()))
        radius = max(2, int(face_width * 0.009))
        for idx in brow_indices:
            x, y = map(int, points[idx])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(brow_mask, (x, y), radius, 255, -1)
        brows = lab[:, :, 0][brow_mask > 0]
        if brows.size >= 10:
            features["brow_L"] = float(np.percentile(brows, 25))
        return features


def read_image(path: str) -> np.ndarray | None:
    """Support Korean filenames on Windows as well as Linux paths."""
    try:
        buffer = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None
