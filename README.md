# Visible Light Visual Positioning — Landing Keypoint Detection & Pose Estimation

基于可见光图像的着陆标志关键点检测与相机实时位姿解算系统。  
使用 Blender 仿真的火箭垂直回收着陆场景，通过检测地面着陆标志上 9 个关键点的 2D 位置，经 PnP 算法反算相机 6-DoF 位姿。

提供两套独立方案，可对比评估：

| | Scheme A: RT-DETRv2 | Scheme B: YOLOv8-pose |
|---|---|---|
| 骨干网络 | ResNet50-vd + HybridEncoder | CSPDarknet (YOLOv8s) |
| 检测范式 | DETR (set prediction) | Anchor-free single-stage |
| 关键点回归 | 自定义多头 Decoder | 原生 Pose head |
| Loss 设计 | 7 项联合损失 (含结构一致性) | YOLOv8 默认 pose loss |
| 位姿解算 | 后处理 + 外部 PnP | 内置 PnP 管线 |
| 预训练权重 | COCO 目标检测 (本地离线) | COCO 关键点检测 (本地离线) |

---

## 项目结构

```
rocket/
│
├── Simulation/                        # 轨迹 CSV + Blender 5.0 渲染脚本
│   ├── generate_trajectory.py         # 6-DoF 轨迹数值仿真
│   ├── generate_vertical_recovery_traj.py  # 可选：姿态扰动
│   └── rocket_trajectory_blender.py   # 在 Blender 内运行：场景与渲染
│
├── datasets/                          # 数据与标注（大量路径见 .gitignore）
│   ├── rocket_render_01/rgb/          # 序列 1 图像 (约 2000 帧)
│   ├── rocket_render_02/rgb/          # 序列 2 图像 (约 2000 帧)
│   ├── kp.json                        # 9 个关键点的 3D 坐标
│   ├── annotations_coco_keypoints.json# COCO 格式关键点标注
│   ├── annotations_coco_test.json     # 测试集标注 (含 range_label)
│   ├── generate_coco_keypoints.py     # 标注生成脚本
│   └── pretrain/
│       ├── rtdetrv2_r50vd_6x_coco_ema.pth  # RT-DETR 预训练
│       └── yolov8s-pose.pt                  # YOLOv8 预训练
│
├── RT-DETR-landing/                   # Scheme A: RT-DETRv2
│   └── rtdetrv2_pytorch/
│       ├── src/zoo/rtdetr/
│       │   ├── landing_decoder.py     # 关键点 Decoder
│       │   ├── landing_criterion.py   # 多任务 Loss (7 项)
│       │   └── landing_postprocessor.py
│       ├── src/data/dataset/
│       │   └── landing_dataset.py     # 数据集类
│       ├── configs/landing/           # 训练配置
│       └── train_landing.py           # 训练入口
│
├── yolo_landing/                      # Scheme B: YOLOv8
│   ├── convert_dataset.py            # COCO → YOLO 格式转换
│   ├── train.py                       # 训练入口
│   ├── pose_solver.py                 # PnP 位姿解算器
│   ├── ellipse_pose_solver.py         # 椭圆混合位姿解算器
│   ├── infer.py                       # 完整推理管线
│   └── yolo_dataset/                  # 转换后的 YOLO 数据集
│
├── evaluation/                        # 统一测试与可视化工具
│   ├── common_metrics.py              # 统一指标计算与图表导出
│   ├── test_yolo.py                   # Scheme B 测试
│   ├── test_rtdetr.py                 # Scheme A 测试
│   ├── visualize_training.py          # 训练过程参数可视化
│   ├── visualize_all.py               # 综合可视化 (训练+测试+轨迹)
│   ├── trajectory_3d.py              # YOLO 轨迹 3D 交互可视化
│   └── trajectory_3d_compare.py       # PnP vs 椭圆轨迹对比
│
├── visualization/                     # 可视化输出目录
│   ├── training/                      # 训练曲线图
│   │   ├── training_comparison.png    # YOLO vs RT-DETR 训练对比
│   │   ├── yolo_training_detail.png   # YOLO 训练详情
│   │   └── rtdetr_training_detail.png # RT-DETR 训练详情
│   ├── comparison/                    # 测试指标对比
│   │   ├── overall_comparison.png     # 总体指标对比
│   │   ├── range_comparison.png       # 近中远分段对比
│   │   ├── keypoint_error_comparison.png
│   │   ├── pose_error_comparison.png
│   │   ├── speed_comparison.png
│   │   └── summary_table.png         # 综合摘要表格
│   ├── trajectory/                    # 轨迹可视化
│   │   ├── gt_trajectory_3d.png       # GT 轨迹 3D
│   │   ├── gt_trajectory_topview.png  # GT 轨迹俯视
│   │   ├── error_along_trajectory.png # 轨迹误差分布
│   │   └── altitude_vs_error.png      # 高度-误差关系
│   ├── testing/yolo/                  # YOLO 测试原始数据
│   └── testing/rtdetr/                # RT-DETR 测试原始数据
│
├── requirements.txt
└── README.md
```

---

## 1. 环境配置

```bash
conda create -n rocket python=3.11 -y
conda activate rocket
pip install -r requirements.txt
```

### 1.1 预训练权重 (离线)

| 文件 | 用途 | 大小 |
|------|------|------|
| `datasets/pretrain/rtdetrv2_r50vd_6x_coco_ema.pth` | RT-DETRv2 骨干网络 | 165MB |
| `datasets/pretrain/yolov8s-pose.pt` | YOLOv8-pose | 22MB |

如需手动下载：
- RT-DETR: [rtdetrv2_r50vd](https://github.com/lyuwenyu/RT-DETR/releases)
- YOLOv8: [yolov8s-pose.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-pose.pt)

---

## 2. 数据集

### 2.1 数据来源

使用 `Simulation/rocket_trajectory_blender.py` 在 **Blender 5.0** 中仿真火箭垂直回收场景。  
相机安装在火箭侧面，俯视地面着陆标志。两条轨迹共约 4000 帧，按 7:2:1 划分 train/test/val。

### 2.2 关键点定义 (`kp.json`)

着陆标志上 9 个关键点的 3D 世界坐标 (Z=0 共面)：

| 编号 | 含义 | 坐标 (X, Y, Z) |
|------|------|-----------------|
| kp0 | 中心 | (0, 0, 0) |
| kp1 | +X 臂端 | (21.5, 0, 0) |
| kp2 | +Y 臂端 | (0, 21.5, 0) |
| kp3 | -X 臂端 | (-21.5, 0, 0) |
| kp4 | -Y 臂端 | (0, -21.5, 0) |
| kp5–kp8 | 内侧标记 | (±3.9, ±9.0, 0) |

### 2.3 相机内参

```
fx = fy = 720.0  (24mm / 36mm × 1080)
cx = 540.0, cy = 360.0
无畸变 (仿真数据)
```

---

## 3. Scheme A: RT-DETRv2 Landing

### 3.1 架构

```
Image → PResNet50-vd → HybridEncoder → RTDETRTransformerLanding → {bbox, class, keypoints, visibility}
                                              │
                           在每个 Decoder 层增加:
                           ├── dec_kpt_head: MLP → K×2 (关键点坐标, sigmoid)
                           └── dec_vis_head: Linear → K  (可见性 logit)
```

### 3.2 Loss 函数 (7 项联合损失)

```
L = λ₁·L_vfl + λ₂·L_bbox + λ₃·L_giou + λ₄·L_kpt + λ₅·L_oks + λ₆·L_vis + λ₇·L_struct
```

| 损失 | 权重 | 作用 |
|------|------|------|
| `L_vfl` | 1 | 分类 |
| `L_bbox` | 5 | 边界框回归 |
| `L_giou` | 2 | 边界框形状 |
| **`L_kpt`** | **10** | **关键点坐标精度** |
| **`L_oks`** | **4** | **尺度感知关键点相似度** |
| `L_vis` | 1 | 可见性预测 |
| **`L_struct`** | **2** | **结构一致性（刚体约束）** |

### 3.3 训练

```bash
cd RT-DETR-landing/rtdetrv2_pytorch
python train_landing.py -c configs/landing/rtdetrv2_r50vd_landing.yml
```

配置：60 epochs, AdamW, lr=1e-4 (backbone 1/20), batch=8, EMA, AMP。

---

## 4. Scheme B: YOLOv8-pose Landing

### 4.1 架构

```
Image → YOLOv8s-pose → 9 keypoints (u,v,conf) → EPnP + LM Refine → Camera Pose (R,t)
         ~12ms/frame                                    <0.4ms/frame
```

### 4.2 训练

```bash
cd yolo_landing
python convert_dataset.py --copy   # COCO → YOLO 格式
python train.py                     # 100 epochs, batch=128
```

训练策略：禁用翻转和 Mosaic (保护关键点方位语义)，仅色彩和缩放增强。

### 4.3 推理 + PnP 位姿解算

```bash
python infer.py --weights runs/landing_pose/weights/best.pt --source ../datasets/rocket_render_01/rgb/ --pnp epnp
```

PnP 算法：`epnp` (默认) | `ippe` (共面优化) | `sqpnp` | `ransac` | `iterative`

### 4.4 椭圆混合位姿解算器

`ellipse_pose_solver.py` 提供了基于椭圆几何的混合解算策略，在高空远距场景中利用着陆标志的椭圆形状约束辅助 PnP。

---

## 5. 评估与可视化

### 5.1 统一测试

```bash
# YOLO 测试
python evaluation/test_yolo.py --weights yolo_landing/runs/landing_pose/weights/best.pt

# RT-DETR 测试
python evaluation/test_rtdetr.py --weights RT-DETR-landing/rtdetrv2_pytorch/output/rtdetrv2_r50vd_landing/best.pth
```

输出 `metrics_summary.json`，包含：关键点误差、PCK、位姿误差、速度、近中远分段指标。

### 5.2 综合可视化

```bash
python evaluation/visualize_all.py
```

一键生成全部对比图表到 `visualization/` 目录：

| 图表 | 路径 | 说明 |
|------|------|------|
| 训练曲线对比 | `training/training_comparison.png` | YOLO vs RT-DETR 四宫格 |
| 总体指标对比 | `comparison/overall_comparison.png` | 关键点/位姿/速度 |
| 近中远分段 | `comparison/range_comparison.png` | 6 项指标 × 3 段 |
| 误差分布 | `comparison/*_comparison.png` | 叠加直方图 |
| 摘要表格 | `comparison/summary_table.png` | 论文级对比表 |
| 轨迹误差 | `trajectory/error_along_trajectory.png` | 帧号 vs 误差 |
| 高度-误差 | `trajectory/altitude_vs_error.png` | 远距离误差放大 |
| GT 轨迹 | `trajectory/gt_trajectory_3d.png` | 3D + 俯视图 |

### 5.3 轨迹交互可视化

```bash
python evaluation/trajectory_3d.py                   # YOLO + PnP 轨迹
python evaluation/trajectory_3d_compare.py            # PnP vs 椭圆混合对比
```

生成可在浏览器中旋转/缩放的 3D HTML 文件。

---

## 6. 实验结果

### 6.1 总体对比

| 指标 | YOLOv8-pose | RT-DETRv2 |
|------|-------------|-----------|
| 检测率 | 100% | 100% |
| 关键点均值误差 (px) | 7.61 | 10.77 |
| 关键点中值误差 (px) | 0.65 | 0.72 |
| PCK@5px | 96.8% | 76.1% |
| PCK@10px | 97.1% | 80.9% |
| 平移误差 (均值) | 82.4 | 252.8 |
| 平移误差 (中值) | 34.2 | 21.3 |
| 旋转误差 (均值/°) | 4.02 | 4.75 |
| 旋转误差 (中值/°) | 4.14 | 3.09 |
| 推理速度 (FPS) | **82.4** | 39.6 |
| 参数量 | 11.6M | 43.5M |

### 6.2 近中远分段 (YOLO)

| 分段 | 样本数 | PCK@5px | 平移误差 | 旋转误差/° |
|------|--------|---------|----------|------------|
| Near (<100m) | 434 | 94.2% | 10.9 | 2.21 |
| Mid (100-300m) | 273 | 100% | 127.7 | 5.99 |
| Far (>300m) | 93 | 100% | 283.0 | 6.66 |

### 6.3 关键发现

- **检测阶段**两种方法均达到 100% 检测率，训练效果良好
- **关键点精度** YOLO 显著优于 RT-DETR (PCK@5: 96.8% vs 76.1%)
- **位姿解算**远距离场景误差显著增大，是当前主要瓶颈
- **速度** YOLO 快 2 倍以上 (82 vs 40 FPS)
- 位姿中值误差 RT-DETR 略优 (21.3 vs 34.2)，但均值因远距离离群点而偏高

---

## 7. TensorBoard

```bash
tensorboard --logdir yolo_landing/runs/landing_pose           # YOLO
tensorboard --logdir RT-DETR-landing/rtdetrv2_pytorch/output  # RT-DETR
```
