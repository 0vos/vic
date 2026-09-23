# Phase 1 初级版本：零训练 baseline（位置对应 + 来源标记）

这个文件夹是 vic 项目的**第一个可跑版本**。目标不是做出好结果，而是把最核心的一条链路打通并**可视化**：

> 动态视频 → 2D 分割 → 2D 姿态 → 分片仿射 warp 到 canonical A-pose → 聚合 → 标记每一块来自哪一帧

## 核心思想（一句话）

不生成新图，而是「拼图」：把视频里真实拍到的、不同姿态不同角度的可见身体表面，用 **2D 关键点 + 分片仿射变换** 拼到一张统一的 canonical 图上，每个像素都记录它来自哪一帧、被多少帧覆盖。

这就是范式 B（appearance aggregation）的规则版——零训练、纯 2D、template-free（不依赖 SMPL 等 3D 人体模板）。

## 位置对应是怎么做的

1. 用 MediaPipe 得到 33 个 2D 关键点（肩膀/肘/腕/髋/膝/踝…）。
2. 在 canonical A-pose 布局（`vic/canonical.py` 里**自己定义**的 2D 坐标）上做 Delaunay 三角剖分。
3. 每一帧，把「当前帧关键点构成的三角形」用仿射变换 warp 到「canonical 对应三角形」—— 这相当于把每帧每个身体部位的像素「搬」到 canonical 图上的正确位置。
4. 多帧 warp 结果聚合，记录来源（source map）和覆盖次数（coverage map）。

**这就是「位置对应」的可视化结果**：`source_map.png` 里每种颜色代表一帧，你一眼能看出「手臂这一块来自第 13 帧、躯干来自第 5 帧」。

## 环境与安装

需要 Mac（Apple Silicon 或 Intel 均可，CPU 就跑得动）：

```bash
cd phase1_baseline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

首次运行会自动下载 MediaPipe 模型（约 5MB）和 rembg 模型（约 170MB）。

## 运行

```bash
python run.py path/to/video.mp4 --frames 24 --out output
```

参数：
- `--frames`：抽多少帧参与聚合（越多覆盖越全，越慢）
- `--out`：输出目录

## 输出说明（都在 output/ 下）

| 文件 | 含义 |
|---|---|
| `canonical_rgb.png` | 聚合后的 canonical 图（最终喂给 Image-to-3D 的参考图） |
| `source_map.png` | **来源标记**：每个像素的颜色对应它来自哪一帧 |
| `coverage_map.png` | 覆盖热力图：蓝=只被 1 帧看到，红=被多帧看到（可见性/置信度） |
| `legend.png` | 帧号 → 颜色 的图例 |
| `intermediate/` | 每帧的 pose / mask / warp 中间结果（debug 用） |

## 当前局限（诚实标注）

- **粗对应**：correspondence 是 part/三角形级别的，不是 pixel-level dense 对应。Phase 2 会引入 DINOv2 + SD 特征或光流做更细的对应。
- **canonical 布局是手写的**：A-pose 坐标是 `canonical.py` 里硬编码的 2D 关键点，不是学出来的，后续可换成 learned canonical layout。
- **单一 canonical 视角**：当前只拼一张正面 A-pose，还没有「正面/背面/左右」多视角互补。后续扩展。
- **分割用 rembg**：对复杂背景/多人场景可能不稳，可替换为 SAM2。

## 下一步（Phase 2）

- dense correspondence（DINOv2 / SD 特征 / 光流）替代 part 级仿射
- 多视角 canonical（front/back/left/right）
- 引入 visibility/confidence 的 soft 权重
- 喂给 Trellis 2 / Rodin，验证「聚合后的 canonical observation 是否优于随机帧」
