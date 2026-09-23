"""Canonical A-pose 关键点布局（template-free，COCO 17 关键点）。

这是「位置对应」的**目标**：把视频每一帧检测到的 2D 关键点，通过分片仿射变换
warp 到这个统一的 A-pose 布局上。

关键点采用 COCO 17（RTMPose 输出格式），是我们**自己定义的 2D 坐标**，
不依赖任何 3D 人体模板（SMPL 等），license 干净，也天然可替换成其他类别。
"""
from __future__ import annotations

import numpy as np

# COCO 17 关键点（顺序即索引 0..16）
COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def canonical_landmarks(width: int = 512, height: int = 768) -> np.ndarray:
    """返回 canonical A-pose 的 17 个关键点坐标 (17, 2)。

    坐标按 [x, y]，基准画布 512x768；若尺寸不同按比例缩放。
    """
    pts = {
        0: (256, 118),   # nose
        1: (242, 104),   # left_eye
        2: (270, 104),   # right_eye
        3: (232, 118),   # left_ear
        4: (280, 118),   # right_ear
        5: (204, 196),   # left_shoulder
        6: (308, 196),   # right_shoulder
        7: (166, 300),   # left_elbow
        8: (346, 300),   # right_elbow
        9: (136, 396),   # left_wrist
        10: (376, 396),  # right_wrist
        11: (228, 348),  # left_hip
        12: (284, 348),  # right_hip
        13: (226, 500),  # left_knee
        14: (286, 500),  # right_knee
        15: (220, 668),  # left_ankle
        16: (292, 668),  # right_ankle
    }
    arr = np.array([pts[i] for i in range(17)], dtype=np.float32)

    if width != 512 or height != 768:
        arr[:, 0] *= width / 512.0
        arr[:, 1] *= height / 768.0
    return arr


# 参与三角剖分的关键点索引：去掉两只眼睛（1,2，离鼻子太近导致三角形过碎），
# 保留鼻子/双耳/双肩/双肘/双腕/双髋/双膝/双踝。
WARP_LANDMARK_IDS = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def body_part_for_landmark(idx: int) -> str:
    """把关键点索引映射到身体部位名，用于可视化标注。"""
    part = {
        0: "head", 1: "head", 2: "head", 3: "head", 4: "head",
        5: "torso", 6: "torso", 11: "torso", 12: "torso",
        7: "left_arm", 9: "left_arm",
        8: "right_arm", 10: "right_arm",
        13: "left_leg", 15: "left_leg",
        14: "right_leg", 16: "right_leg",
    }
    return part.get(idx, "other")
