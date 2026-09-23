"""分片仿射 warp + canonical 聚合（核心：位置对应 + 来源标记）。

思路（纯 2D、零训练、template-free）：
1. 用 2D 关键点（MediaPipe）在 canonical A-pose 布局上做 Delaunay 三角剖分。
2. 每一帧，把「当前帧关键点构成的三角形」用仿射变换 warp 到「canonical 对应三角形」，
   即把该帧里每个身体部位的可见像素「搬」到 canonical 画布的正确位置。
3. 多帧 warp 结果聚合：每个像素记录「来自哪一帧」+「被多少帧覆盖」，
   这就是最终结果图里「每一块对应的位置」的来源标记。
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.spatial import Delaunay


def _boundary_points(w: int, h: int) -> np.ndarray:
    """画布/图像四角 + 四边中点，作为 Delaunay 的外围锚点，保证人体被三角形覆盖。"""
    return np.array(
        [
            [0, 0], [w // 2, 0], [w - 1, 0],
            [w - 1, h // 2], [w - 1, h - 1], [w // 2, h - 1],
            [0, h - 1], [0, h // 2],
        ],
        dtype=np.float32,
    )


def _triangulate(dst_pts: np.ndarray):
    """在 canonical 目标点上做 Delaunay，返回三角形顶点索引 (M, 3)。"""
    tri = Delaunay(dst_pts)
    return tri.simplices


def warp_frame(
    bgr: np.ndarray,
    mask: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    out_size: tuple[int, int],
):
    """把一帧 warp 到 canonical 画布。

    参数
    ----
    bgr : (H, W, 3) 帧图像（背景已处理）
    mask : (H, W) 前景 mask（0/255）
    src_pts : (N, 2) 当前帧的关键点（含边界点），像素坐标
    dst_pts : (N, 2) canonical 关键点（含边界点），画布坐标
    out_size : (width, height) canonical 画布尺寸

    返回
    ----
    warped_rgb : (H_out, W_out, 3) float32
    warped_mask : (H_out, W_out) float32（0~1 的软权重）
    """
    out_w, out_h = out_size
    warped = np.zeros((out_h, out_w, 3), dtype=np.float32)
    wmask = np.zeros((out_h, out_w), dtype=np.float32)

    simplices = _triangulate(dst_pts)
    for tri in simplices:
        s = src_pts[tri].astype(np.float32)
        d = dst_pts[tri].astype(np.float32)
        if _degenerate(s) or _degenerate(d):
            continue
        M = cv2.getAffineTransform(s, d)
        w_img = cv2.warpAffine(bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
        w_msk = cv2.warpAffine(mask, M, (out_w, out_h), flags=cv2.INTER_LINEAR)

        tri_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, d.astype(np.int32), 255)
        inside = tri_mask > 0

        warped[inside] = w_img[inside]
        wmask[inside] = np.maximum(wmask[inside], w_msk[inside] / 255.0)

    return warped, wmask


def _degenerate(pts: np.ndarray) -> bool:
    """判断三点是否近似共线（面积接近 0）。"""
    area = cv2.contourArea(pts.astype(np.float32).reshape(-1, 1, 2))
    return abs(area) < 1.0


def aggregate(warped_list: list[tuple[np.ndarray, np.ndarray]]):
    """聚合多帧 warp 结果到 canonical。

    参数
    ----
    warped_list : [(warped_rgb, warped_mask), ...] 每帧一对

    返回
    ----
    canonical_rgb : (H, W, 3) uint8 聚合图
    source_map : (H, W) int32，每个像素贡献最大的帧索引（-1 表示未被覆盖）
    coverage_map : (H, W) int32，每个像素被多少帧覆盖（可见性/置信度）
    """
    h, w = warped_list[0][0].shape[:2]
    acc = np.zeros((h, w, 3), dtype=np.float64)
    wsum = np.zeros((h, w), dtype=np.float64)
    best_w = np.zeros((h, w), dtype=np.float64)
    source = np.full((h, w), -1, dtype=np.int32)
    coverage = np.zeros((h, w), dtype=np.int32)

    for fi, (warped, wmask) in enumerate(warped_list):
        valid = wmask > 0.0
        weight = wmask  # 0~1 软权重
        acc[valid] += warped[valid] * weight[valid][:, None]
        wsum[valid] += weight[valid]
        coverage[valid] += 1
        # 记录「贡献最大的帧」
        update = valid & (weight > best_w)
        source[update] = fi
        best_w[update] = weight[update]

    denom = np.maximum(wsum, 1e-8)
    canonical = (acc / denom[:, :, None]).astype(np.uint8)
    return canonical, source, coverage
