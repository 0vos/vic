"""Phase 1 主入口：跑完整零训练基线。

用法：
    python run.py path/to/video.mp4 [--frames 24] [--out output]

流程：
    抽帧 -> 分割(rembg) -> 2D pose(MediaPipe) -> 分片仿射 warp 到 canonical -> 聚合
    -> 输出 canonical 图 + 来源标记图 + 覆盖热力图
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from vic import canonical, pose, segment, visualize, warp


def _downscale(frame: np.ndarray, max_size: int) -> np.ndarray:
    """若最长边超过 max_size，等比缩小（用 INTER_AREA，适合缩小）。"""
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return frame


def extract_frames(video_path: str, num_frames: int,
                   max_size: int | None = None) -> list[np.ndarray]:
    """均匀抽 num_frames 帧。max_size 非空时先等比缩小到最长边 max_size。"""
    import os
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"视频文件不存在: {video_path}\n"
            f"当前目录: {os.getcwd()}\n"
            f"当前目录下的文件: {sorted(os.listdir('.'))}"
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频: {video_path}\n"
            f"（文件存在但 OpenCV 无法解码，可能是编码/容器格式问题，试试 ffmpeg 转码）"
        )
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("视频没有帧")
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            if max_size:
                frame = _downscale(frame, max_size)
            frames.append(frame)
    cap.release()
    print(f"[extract] 共抽取 {len(frames)}/{num_frames} 帧"
          + (f"，已下采样到最长边 {max_size}" if max_size else ""))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: canonical observation 零训练基线")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("--frames", type=int, default=config.NUM_FRAMES, help="抽取帧数")
    parser.add_argument("--out", default=str(config.OUTPUT_DIR), help="输出目录")
    parser.add_argument("--max-size", type=int, default=1024,
                        help="最长边下采样到多少像素（0=不缩放，4K 视频建议 1024）")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    inter_dir = out_dir / "intermediate"
    if config.SAVE_INTERMEDIATE:
        inter_dir.mkdir(parents=True, exist_ok=True)

    # 1. 抽帧
    frames = extract_frames(args.video, args.frames, args.max_size or None)
    if not frames:
        sys.exit("没有可用帧")

    # 2. 准备 canonical 目标点 + 分割 session + pose 估计器
    can_pts = canonical.canonical_landmarks(config.CANONICAL_WIDTH, config.CANONICAL_HEIGHT)
    can_sub = can_pts[canonical.WARP_LANDMARK_IDS]
    can_boundary = warp._boundary_points(config.CANONICAL_WIDTH, config.CANONICAL_HEIGHT)
    dst_pts = np.vstack([can_sub, can_boundary])

    seg_session = segment.load_session(config.REMBG_MODEL)
    pose_est = pose.PoseEstimator()

    # 3. 逐帧：分割 -> pose -> warp
    warped_list = []
    for fi, frame in enumerate(frames):
        h, w = frame.shape[:2]
        mask = segment.remove_background(frame, seg_session)
        if mask.sum() < 100:  # 没抠到人
            print(f"[frame {fi}] 分割无前景，跳过")
            continue

        clean = segment.apply_mask(frame, mask)

        # pose（用原始帧检测更稳，背景信息有助于定位；warp 时才用白底 clean 图）
        rgb = frame[..., ::-1]
        det = pose_est.detect(rgb)
        if det is None:
            print(f"[frame {fi}] 未检测到人，跳过")
            continue
        lm, vis = det

        src_sub = lm[canonical.WARP_LANDMARK_IDS]
        src_boundary = warp._boundary_points(w, h)
        src_pts = np.vstack([src_sub, src_boundary])

        warped, wmask = warp.warp_frame(
            clean, mask, src_pts, dst_pts,
            (config.CANONICAL_WIDTH, config.CANONICAL_HEIGHT),
        )
        warped_list.append((warped, wmask))

        if config.SAVE_INTERMEDIATE:
            cv2.imwrite(str(inter_dir / f"frame_{fi:02d}_pose.png"),
                        visualize.draw_skeleton(clean.copy(), lm))
            cv2.imwrite(str(inter_dir / f"frame_{fi:02d}_mask.png"), mask)
            cv2.imwrite(str(inter_dir / f"frame_{fi:02d}_warp.png"), warped.astype(np.uint8))

    if not warped_list:
        sys.exit("没有任何帧成功 warp，检查视频里是否有人")

    print(f"[aggregate] 聚合 {len(warped_list)} 帧")

    # 4. 聚合
    canon_rgb, source_map, coverage_map = warp.aggregate(warped_list)

    # 5. 可视化 + 保存
    cv2.imwrite(str(out_dir / "canonical_rgb.png"), canon_rgb)
    src_color = visualize.render_source_map(source_map, len(warped_list))
    cv2.imwrite(str(out_dir / "source_map.png"), src_color)
    cov_color = visualize.render_coverage_map(coverage_map)
    cv2.imwrite(str(out_dir / "coverage_map.png"), cov_color)
    legend = visualize.build_legend(len(warped_list))
    cv2.imwrite(str(out_dir / "legend.png"), legend)

    print("\n完成。输出：")
    print(f"  {out_dir / 'canonical_rgb.png'}  聚合后的 canonical 图")
    print(f"  {out_dir / 'source_map.png'}     每个像素来自哪一帧（来源标记）")
    print(f"  {out_dir / 'coverage_map.png'}   每个像素被多少帧覆盖（可见性）")
    print(f"  {out_dir / 'legend.png'}         帧号 -> 颜色 图例")


if __name__ == "__main__":
    main()
