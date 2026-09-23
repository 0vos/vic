"""2D 姿态估计（RTMPose + onnxruntime，Apache 2.0）。

为什么不用 MediaPipe：新版 Tasks 在 macOS 上强制初始化 Metal GPU 崩溃，
legacy mp.solutions 又被新版移除。改用 RTMPose（OpenMMLab，Apache 2.0），
onnxruntime 纯 CPU 推理，跨平台稳定，输出 COCO 17 关键点。

rtmlib 封装了 RTMPose 的预处理（crop+resize+normalize）与 SimCC 解码，
首次运行会自动下载模型（约 40MB）。
"""
from __future__ import annotations

import numpy as np

_MODEL_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)


class PoseEstimator:
    def __init__(self):
        from rtmlib import RTMPose

        self._pose = RTMPose(
            onnx_model=_MODEL_URL,
            backend="onnxruntime",
            device="cpu",
        )

    def detect(self, bgr: np.ndarray, bbox):
        """输入 BGR 图 (H, W, 3) + 人体 bbox (x1, y1, x2, y2)（xyxy 像素坐标）。

        返回 (kp (17,2) float32, scores (17,) float32)。
        """
        # rtmlib 的 RTMPose.__call__(image, bboxes=[...])，bboxes 是 list
        keypoints, scores = self._pose(bgr, bboxes=[list(bbox)])
        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if keypoints.ndim == 3:  # (1, 17, 2) -> (17, 2)
            keypoints = keypoints[0]
            scores = scores[0]
        return keypoints, scores
