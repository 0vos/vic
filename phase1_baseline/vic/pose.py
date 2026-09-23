"""2D 姿态估计（MediaPipe legacy solutions，Apache 2.0）。

为什么用 legacy `mp.solutions` 而非新版 Tasks API：
新版 `PoseLandmarker` 在 macOS 上，即使设置 `delegate=CPU`，graph 里的
`TensorsToDetectionsCalculator` 仍会强制初始化 Metal GPU（DrishtiMetalHelper），
直接 abort（`Check failed: service_ Service is unavailable`），这是预编译 wheel 的死结。
legacy `mp.solutions.pose` 走的是纯 CPU 的预编译 graph，绕开 Metal，稳定。

关键点：legacy 与 Tasks 共用同一套 33 个 landmark，索引语义一致，
因此 canonical.py 的 A-pose 布局无需改动。只用 2D 关键点，不涉及 3D 人体模板。
"""
from __future__ import annotations

import numpy as np


class PoseEstimator:
    def __init__(self, model_complexity: int = 1):
        import mediapipe as mp

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "pose"):
            raise RuntimeError(
                "当前 mediapipe 版本已移除 legacy mp.solutions.pose，"
                "请改用 onnxruntime + RTMPose 方案（见 README 的 Phase 2 说明）"
            )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,      # 逐帧独立检测（非视频流）
            model_complexity=model_complexity,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )

    def detect(self, rgb: np.ndarray):
        """输入 RGB 图 (H, W, 3) uint8，返回 (landmarks_xy (33,2) float32, visibility (33,) float32)。

        未检测到人时返回 None。
        """
        h, w = rgb.shape[:2]
        rgb = np.ascontiguousarray(rgb)  # legacy API 要求连续内存、可写
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        xy = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        vis = np.array([p.visibility for p in lm], dtype=np.float32)
        return xy, vis

    def close(self) -> None:
        self._pose.close()
