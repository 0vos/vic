"""2D 姿态估计（MediaPipe Pose，Apache 2.0，Apple Silicon 友好）。

只用 2D 关键点，不涉及任何 3D 人体模板。首次运行会自动下载官方模型文件。
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_MODEL_NAME = "pose_landmarker_lite.task"


def _ensure_model(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / _MODEL_NAME
    if not model_path.exists():
        print(f"[pose] 下载 MediaPipe 模型 -> {model_path} ...")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    return model_path


class PoseEstimator:
    def __init__(self, model_dir: str = "models"):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model_path = _ensure_model(Path(model_dir))
        base = python.BaseOptions(model_asset_path=str(model_path))
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray):
        """输入 RGB 图 (H, W, 3) uint8，返回 (landmarks_xy (33,2) float32, visibility (33,) float32)。

        未检测到人时返回 None。
        """
        h, w = rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]
        xy = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        vis = np.array([p.visibility for p in lm], dtype=np.float32)
        return xy, vis
