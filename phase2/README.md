# Phase 2：correspondence 最小验证

先用一个最小实验回答最关键的问题：

> 不同姿态下，「这一帧的手臂像素」能否通过视觉先验（DINOv2 特征）正确对应到「另一帧的同一部位」？

这是 canonical accumulation 的地基。对应不对，累积出来的就是错位的碎片。

## 依赖

```bash
pip install torch torchvision opencv-python numpy
```

（DINOv2 权重首次运行会从 torch.hub 自动下载，约 90MB）

## 运行

两张图：

```bash
python correspondence_demo.py --img-a a.jpg --img-b b.jpg
```

或从同一段视频抽两帧（不同姿态）：

```bash
python correspondence_demo.py --video v.mp4 --frame-a 10 --frame-b 40
```

Apple Silicon 可加 `--device mps` 加速。

## 输出怎么看

`correspondence.png`：左边是帧 A，右边是帧 B，每条线把 A 的一个 patch 连到它在 B 的对应位置。

- **绿线** = 高置信对应（两个 patch 语义很接近）
- **红线** = 低置信（可能被遮挡，或跨姿态后 patch 内容变化太大）

**判断标准**：如果 A 帧「手臂」上的采样点，线大多连到了 B 帧的「手臂」位置（而不是乱连到腿/背景），说明视觉特征对应是可信的，可以继续往上搭累积。如果线乱飞，说明需要更强的对应方法（SD 特征 / 光流 / 训练）。
