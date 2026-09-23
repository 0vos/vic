"""Phase 2 最小验证：两帧之间的 dense 语义对应（DINOv2 特征）。

回答一个问题：不同姿态下，「这一帧的手臂像素」能否正确对应到「另一帧的同一部位」？

这是整个 canonical accumulation 的地基——对应不对，累积就是错的。
先用现成的 DINOv2 视觉先验（零训练、Apache 2.0）验证这一点，再决定后续怎么做。

用法：
    # 两张图
    python correspondence_demo.py --img-a a.jpg --img-b b.jpg
    # 或从一个视频抽两帧
    python correspondence_demo.py --video v.mp4 --frame-a 10 --frame-b 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PATCH = 14          # dinov2 vits14 的 patch 大小
GRID = 16           # 224 / 14 = 16 个 patch/边
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_dinov2(device: str = "cpu"):
    import torch

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    model.to(device)
    return model


def extract_features(model, rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    """rgb: (H,W,3) uint8 -> patch 特征 (16,16,C)。"""
    import torch

    img = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.forward_features(tensor)  # (1, 257, C)
    feats = feats[:, 1:, :]                      # 去掉 cls token
    return feats.reshape(GRID, GRID, -1).cpu().numpy()


def dense_match(feat_a: np.ndarray, feat_b: np.ndarray):
    """对 A 的每个 patch，找 B 里余弦最相似的 patch。"""
    c = feat_a.shape[-1]
    fa = feat_a.reshape(-1, c)
    fb = feat_b.reshape(-1, c)
    fa /= np.linalg.norm(fa, axis=1, keepdims=True) + 1e-8
    fb /= np.linalg.norm(fb, axis=1, keepdims=True) + 1e-8
    sim = fa @ fb.T                       # (256, 256)
    idx = sim.argmax(axis=1)              # 每个 A patch 的最近 B patch
    score = sim.max(axis=1)               # 相似度（置信度）
    return idx.reshape(GRID, GRID), score.reshape(GRID, GRID)


def _patch_center(idx: int):
    i, j = divmod(int(idx), GRID)
    return j * PATCH + PATCH // 2, i * PATCH + PATCH // 2


def visualize(img_a: np.ndarray, img_b: np.ndarray,
              match_idx: np.ndarray, match_score: np.ndarray, step: int = 2):
    """并排两帧，采样 patch 画对应连线。绿=高置信，红=低置信。"""
    h = w = 224
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = cv2.resize(img_a, (w, h))
    canvas[:, w:] = cv2.resize(img_b, (w, h))

    for i in range(0, GRID, step):
        for j in range(0, GRID, step):
            ax, ay = _patch_center(i * GRID + j)
            bx, by = _patch_center(match_idx[i, j])
            bx += w
            sc = float(match_score[i, j])
            color = (0, int(255 * sc), int(255 * (1 - sc)))  # BGR: 绿->红
            cv2.line(canvas, (ax, ay), (bx, by), color, 1)
            cv2.circle(canvas, (ax, ay), 2, (0, 0, 255), -1)
            cv2.circle(canvas, (bx, by), 2, (255, 0, 0), -1)
    return canvas


def _read_frame(video: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"无法读取第 {idx} 帧: {video}")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-a")
    ap.add_argument("--img-b")
    ap.add_argument("--video")
    ap.add_argument("--frame-a", type=int, default=0)
    ap.add_argument("--frame-b", type=int, default=30)
    ap.add_argument("--step", type=int, default=2, help="连线采样步长（越小越密）")
    ap.add_argument("--device", default="cpu", help="cpu / mps")
    ap.add_argument("--out", default="correspondence.png")
    args = ap.parse_args()

    if args.img_a and args.img_b:
        img_a = cv2.imread(args.img_a)
        img_b = cv2.imread(args.img_b)
    elif args.video:
        img_a = _read_frame(args.video, args.frame_a)
        img_b = _read_frame(args.video, args.frame_b)
    else:
        raise SystemExit("请提供 --img-a/--img-b 或 --video")

    print("[load] 加载 DINOv2（首次会下载权重，约 90MB）...")
    model = load_dinov2(args.device)

    print("[feat] 提取两帧 patch 特征 ...")
    fa = extract_features(model, img_a[..., ::-1], args.device)  # BGR->RGB
    fb = extract_features(model, img_b[..., ::-1], args.device)

    print("[match] dense 语义匹配 ...")
    idx, score = dense_match(fa, fb)
    mean_score = float(score.mean())
    print(f"[match] 平均匹配置信度: {mean_score:.3f}（越高说明两帧语义越接近）")

    vis = visualize(img_a, img_b, idx, score, args.step)
    out = Path(args.out)
    cv2.imwrite(str(out), vis)
    print(f"[done] 可视化已保存 -> {out}")
    print("       左=帧A，右=帧B；每条线把 A 的一个 patch 连到它在 B 的对应位置")
    print("       绿线=高置信对应，红线=低置信（可能是被遮挡/跨姿态难对应）")


if __name__ == "__main__":
    main()
