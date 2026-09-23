"""可视化：把聚合结果和「每一块的来源位置」渲染成可看的图。

输出三类图：
1. canonical_rgb.png —— 聚合后的 canonical 图（最终参考图）
2. source_map.png —— 每个像素用颜色标记「它来自哪一帧」（位置对应来源）
3. coverage_map.png —— 每个像素被多少帧覆盖（可见性/置信度热力图）
"""
from __future__ import annotations

import numpy as np
import cv2


# MediaPipe Pose 骨架连线（用于可选的关键点可视化）
BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]


def _tab20_color(i: int) -> tuple[int, int, int]:
    """给帧索引分配一个可区分的 BGR 颜色（tab20 调色板循环）。"""
    palette = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
        (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
        (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
    ]
    r, g, b = palette[i % len(palette)]
    return (b, g, r)  # palette 是 RGB，转 BGR


def render_source_map(source_map: np.ndarray, num_frames: int) -> np.ndarray:
    """source_map (H, W) int32 -> BGR 彩色图。未覆盖（-1）显示为白色。"""
    h, w = source_map.shape
    out = np.full((h, w, 3), 255, dtype=np.uint8)  # 背景白
    for fi in range(num_frames):
        mask = source_map == fi
        out[mask] = _tab20_color(fi)
    return out


def render_coverage_map(coverage_map: np.ndarray) -> np.ndarray:
    """coverage (H, W) -> 热力图（蓝=低覆盖，红=高覆盖，白=未覆盖）。"""
    cov = coverage_map.astype(np.float32)
    max_cov = max(int(cov.max()), 1)
    norm = np.zeros_like(cov)
    valid = cov > 0
    norm[valid] = cov[valid] / max_cov
    # 映射到 colormap：0 -> 蓝(255,0,0)? 用 cv2 JET
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored[~valid] = (255, 255, 255)  # 未覆盖白
    return colored


def draw_skeleton(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """在图上画关键点 + 骨架连线（用于 debug 可视化）。"""
    out = img.copy()
    for a, b in BODY_CONNECTIONS:
        pa = tuple(landmarks[a].astype(int))
        pb = tuple(landmarks[b].astype(int))
        cv2.line(out, pa, pb, (0, 255, 0), 2)
    for p in landmarks:
        cv2.circle(out, tuple(p.astype(int)), 3, (0, 0, 255), -1)
    return out


def build_legend(num_frames: int) -> np.ndarray:
    """生成「帧号 -> 颜色」图例条。"""
    cell = 28
    legend = np.full((cell * ((num_frames + 7) // 8), 8 * cell, 3), 255, dtype=np.uint8)
    for fi in range(num_frames):
        r, c = divmod(fi, 8)
        y0, x0 = r * cell, c * cell
        cv2.rectangle(legend, (x0, y0), (x0 + cell - 2, y0 + cell - 2), _tab20_color(fi), -1)
        cv2.putText(legend, str(fi), (x0 + 4, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return legend
