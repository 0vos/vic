# vic

> 从动态单目视频 → surface-consistent canonical observation → 可驱动 3D asset

**当前进度**：`phase1_baseline/`（初级版本）——零训练、纯 2D 视觉、template-free 的 baseline，目标是跑通「位置对应 + 来源标记」这条最核心的链路，验证「聚合后的 canonical observation 是否优于随机帧」。

## 核心哲学

不在 2D 里硬碰硬做 3D 重建。用成熟的 2D 视觉技巧（分割、2D 姿态、特征对应、光流、图像处理）把动态视频里的可见表面「拼」成一组身份一致、视角互补的 canonical observation，再把最难的 3D 几何推断整个交给现成的 Image-to-3D foundation model（Trellis 2 / Rodin）。

## 约束

- **template-free**：不使用 SMPL / DensePose 等 3D 人体模板（license 问题）
- **license-friendly**：只依赖 Apache 2.0 / MIT / BSD 的组件
- **纯 2D**：所有中间表示都在 2D 图像空间

## 目录

```
vic/
├── phase1_baseline/     # 初级版本：零训练基线（位置对应 + 来源标记）
└── ...
```

详见 `phase1_baseline/README.md`。
