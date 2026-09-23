"""Canonical A-pose 关键点布局（template-free）。

这是「位置对应」的**目标**：我们把视频每一帧里检测到的人体关键点，
通过分片仿射变换 warp 到这个统一的 A-pose 布局上。

关键点索引采用 MediaPipe Pose 的 33 个 landmark（见下方 LANDMARK_NAMES）。
这套布局是我们**自己定义的 2D 坐标**，不依赖任何 3D 人体模板（SMPL 等），
因此 license 干净，也天然可以替换成动物/玩具等其他类别（Phase 3）。

坐标说明：x 向右、y 向下，画布尺寸来自 config.CANONICAL_WIDTH/HEIGHT。
"""
from __future__ import annotations

import numpy as np

# MediaPipe Pose 33 个 landmark 的语义名（顺序即索引 0..32）
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def canonical_landmarks(width: int = 512, height: int = 768) -> np.ndarray:
    """返回 canonical A-pose 的 33 个关键点坐标 (N, 2)。

    坐标按 [x, y] 排列，x 归一化到 width、y 归一化到 height 的比例。
    这里直接写绝对像素坐标（针对 512x768），调用方若换了画布尺寸，
    按比例缩放即可。
    """
    # 基准画布 512 x 768
    pts = {
        0: (256, 118),    # nose
        1: (242, 104),    # left_eye_inner
        2: (236, 102),    # left_eye
        3: (230, 106),    # left_eye_outer
        4: (270, 104),    # right_eye_inner
        5: (276, 102),    # right_eye
        6: (282, 106),    # right_eye_outer
        7: (228, 122),    # left_ear
        8: (284, 122),    # right_ear
        9: (243, 142),    # mouth_left
        10: (269, 142),   # mouth_right
        11: (204, 196),   # left_shoulder
        12: (308, 196),   # right_shoulder
        13: (166, 300),   # left_elbow
        14: (346, 300),   # right_elbow
        15: (136, 396),   # left_wrist
        16: (376, 396),   # right_wrist
        17: (128, 414),   # left_pinky
        18: (384, 414),   # right_pinky
        19: (142, 408),   # left_index
        20: (370, 408),   # right_index
        21: (134, 390),   # left_thumb
        22: (378, 390),   # right_thumb
        23: (228, 348),   # left_hip
        24: (284, 348),   # right_hip
        25: (226, 500),   # left_knee
        26: (286, 500),   # right_knee
        27: (220, 668),   # left_ankle
        28: (292, 668),   # right_ankle
        29: (214, 692),   # left_heel
        30: (298, 692),   # right_heel
        31: (238, 692),   # left_foot_index
        32: (274, 692),   # right_foot_index
    }
    arr = np.array([pts[i] for i in range(33)], dtype=np.float32)

    # 若画布尺寸不是 512x768，按比例缩放
    if width != 512 or height != 768:
        arr[:, 0] *= width / 512.0
        arr[:, 1] *= height / 768.0
    return arr


# 参与三角剖分的关键点索引：
# 去掉脸部 6 个细节点（1~6）和嘴两侧（9,10），保留鼻子/耳朵/肩/肘/腕/手/髋/膝/踝/足，
# 避免脸部三角形过碎导致仿射变形走样。
WARP_LANDMARK_IDS = [
    0, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
]


def body_part_for_landmark(idx: int) -> str:
    """把 landmark 索引映射到身体部位名，用于可视化标注「这一块是哪个部位」。"""
    part = {
        0: "head", 7: "head", 8: "head",
        11: "torso", 12: "torso", 23: "torso", 24: "torso",
        13: "left_arm", 15: "left_arm", 17: "left_arm", 19: "left_arm", 21: "left_arm",
        14: "right_arm", 16: "right_arm", 18: "right_arm", 20: "right_arm", 22: "right_arm",
        25: "left_leg", 27: "left_leg", 29: "left_leg", 31: "left_leg",
        26: "right_leg", 28: "right_leg", 30: "right_leg", 32: "right_leg",
    }
    return part.get(idx, "other")
