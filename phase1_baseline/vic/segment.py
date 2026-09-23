"""前景分割 / 抠图（rembg，MIT，CPU 可跑）。

把人从背景里抠出来，得到前景 mask。mask 同时作为后续 warp 的「可见性」依据。

注意：rembg 的 session 加载模型较慢，务必复用同一个 session（见 load_session）。
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def load_session(model_name: str = "u2net"):
    """加载并返回 rembg session（只调用一次，复用）。"""
    from rembg import new_session

    return new_session(model_name)


def remove_background(bgr: np.ndarray, session) -> np.ndarray:
    """输入 BGR 图 (H, W, 3) uint8，返回二值 mask (H, W) uint8（255=前景）。"""
    from rembg import remove

    rgb = bgr[..., ::-1]  # BGR -> RGB
    img = Image.fromarray(rgb)
    out = remove(img, session=session)  # RGBA
    out = np.array(out)
    alpha = out[..., 3]
    mask = (alpha > 128).astype(np.uint8) * 255
    return mask


def apply_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把背景置为纯白（用于后续 warp，避免背景像素污染 canonical 图）。"""
    out = bgr.copy()
    out[mask == 0] = (255, 255, 255)
    return out
