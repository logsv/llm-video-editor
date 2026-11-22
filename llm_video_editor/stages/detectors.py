"""Face / subject detectors for take-geometry.

Default: MediaPipe Tasks FaceDetector (CPU, ``vision`` extra). Fallback: YOLO
person boxes (``pro`` extra). With neither installed the stage still runs and
keeps every crop centred.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np

FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                  "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
Box = tuple[str, tuple[float, float, float, float], float]  # (label, bbox, score)


class Detector(Protocol):
    name: str

    def detect(self, rgb: np.ndarray) -> list[Box]: ...

    def close(self) -> None: ...


def _clip01(v: float) -> float:
    return min(1.0, max(0.0, v))


def face_model_path() -> Path:
    env = os.environ.get("LLMVE_FACE_MODEL")
    if env:
        return Path(env)
    path = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "llmve" / "models"
    path = path / "blaze_face_short_range.tflite"
    if not path.exists():
        import requests

        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(FACE_MODEL_URL, timeout=60)
        r.raise_for_status()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)
    return path


class MediaPipeFaceDetector:
    name = "mediapipe-face"

    def __init__(self, min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(face_model_path())),
            min_detection_confidence=min_confidence)
        self._det = vision.FaceDetector.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> list[Box]:
        h, w = rgb.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(rgb))
        out: list[Box] = []
        for d in self._det.detect(img).detections:
            b = d.bounding_box
            score = d.categories[0].score if d.categories else 0.0
            out.append(("face", (_clip01(b.origin_x / w), _clip01(b.origin_y / h),
                                 _clip01((b.origin_x + b.width) / w),
                                 _clip01((b.origin_y + b.height) / h)), float(score)))
        return out

    def close(self) -> None:
        self._det.close()


class YoloPersonDetector:
    name = "yolo-person"

    def __init__(self, model: str = "yolo11n.pt", min_confidence: float = 0.4):
        from ultralytics import YOLO

        self._model = YOLO(yolo_weights(model))
        self.min_confidence = min_confidence

    def detect(self, rgb: np.ndarray) -> list[Box]:
        h, w = rgb.shape[:2]
        res = self._model.predict(rgb[:, :, ::-1], classes=[0], verbose=False,
                                  conf=self.min_confidence)
        out: list[Box] = []
        for r in res:
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                           strict=True):
                out.append(("person", (_clip01(x1 / w), _clip01(y1 / h), _clip01(x2 / w),
                                       _clip01(y2 / h)), float(c)))
        return out

    def close(self) -> None:
        pass


def yolo_weights(name: str) -> str:
    """Weights in ~/.cache/llmve/models (downloaded there once), never the working dir."""
    if os.sep in name or "/" in name:
        return name
    d = Path(os.environ.get("LLMVE_CACHE") or Path.home() / ".cache" / "llmve") / "models"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / name)


class YoloObjectDetector:
    """Everyday objects (COCO's 80 classes: animals such as cow, dog, horse, bird,
    elephant; vehicles; people; ...) with YOLO, locally. Needs the `pro` extra."""

    name = "yolo-objects"

    def __init__(self, model: str = "yolo11n.pt", min_confidence: float = 0.35):
        from ultralytics import YOLO

        self._model = YOLO(yolo_weights(model))
        self.min_confidence = min_confidence

    def detect(self, rgb: np.ndarray) -> list[Box]:
        h, w = rgb.shape[:2]
        res = self._model.predict(rgb[:, :, ::-1], verbose=False, conf=self.min_confidence)
        out: list[Box] = []
        for r in res:
            for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                              r.boxes.cls.tolist(), strict=True):
                out.append((str(r.names[int(k)]), (_clip01(x1 / w), _clip01(y1 / h),
                                                   _clip01(x2 / w), _clip01(y2 / h)), float(c)))
        return out

    def close(self) -> None:
        pass


class NullDetector:
    name = "none"

    def detect(self, rgb: np.ndarray) -> list[Box]:
        return []

    def close(self) -> None:
        pass


def get_detector(name: str = "auto") -> Detector:
    if name in ("auto", "mediapipe"):
        try:
            return MediaPipeFaceDetector()
        except Exception:
            if name == "mediapipe":
                raise
    if name in ("auto", "yolo"):
        try:
            return YoloPersonDetector()
        except Exception:
            if name == "yolo":
                raise
    if name in ("auto", "none"):
        return NullDetector()
    raise ValueError(f"Unknown detector {name!r}: auto | mediapipe | yolo | none")
