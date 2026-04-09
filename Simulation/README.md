# Simulation — 可复用火箭着陆段仿真数据集生成

---

## 流程总览

```text
generate_trajectory.py          ← 轨迹生成 (Falcon 9 物理仿真)
        │
        ▼
  trajectory.csv  (50 Hz, 6-DoF)
        │
        ▼
render_dataset.py               ← 合成图像渲染 (透视投影 + 数据增广)
        │
        ▼
  dataset/                       ← RGB + 深度图 + 语义掩码 + YOLO 标注
        │
        ▼
train.py                        ← PyTorch 目标检测训练
        │
        ▼
  best_model.pth                 ← 着陆标志检测器
```

---

## 文件结构

```text
Simulation/
├── generate_trajectory.py   # Module 1: 轨迹生成器 (Falcon 9 物理模型)
├── generate_trajectory.m    # Module 1: MATLAB 参考实现
├── render_dataset.py        # Module 2: 合成图像渲染器
├── export_video.py          # 按仿真时间将 rgb 帧导出为视频
├── train.py                 # Module 3: PyTorch 训练流程
├── requirements.txt         # Python 依赖
├── README.md                # 本文件
└── dataset/                 # [生成产物] 渲染输出数据集
    ├── rgb/                 #   frame_XXXXX.png  (1920×1080 RGB)
    ├── depth/               #   frame_XXXXX.png  (uint16 深度图)
    ├── mask/                #   frame_XXXXX.png  (着陆标志掩码)
    ├── annotations/labels/  #   frame_XXXXX.txt  (YOLO 格式)
    ├── pose.csv             #   逐帧 6-DoF 位姿
    ├── velocity.csv         #   逐帧速度
    ├── acceleration.csv     #   逐帧加速度
    ├── dataset_meta.json    #   数据集元信息
    └── landing_simulation.mp4  # [可选] 按仿真时间导出的视频
```

> **注**: `trajectory.csv`、`trajectory_plot.png` 和 `dataset/` 均为脚本生成产物，
> 不纳入版本管理。可通过下方"快速启动"步骤重新生成。

---

## Module 1 — 轨迹生成 (`generate_trajectory.py`)

基于 Falcon 9 Block 5 一级着陆段真实参数进行数值积分仿真。

### 物理模型

| 子系统 | 模型 |
|--------|------|
| 大气 | 指数密度: ρ(z) = 1.225 · exp(−z / 8500) |
| 气动阻力 | F = ½ρv²C_dA,  C_d = 1.3 (含栅格翼),  A = π·1.83² m² |
| 发动机 | Merlin 1D: 854 kN, 比冲 282 s, 质量实时递减 |
| 垂直制导 | 恒减速剖面 + 速度跟踪修正 |
| 水平制导 | ZEM/ZEV 多项式导引 + 低空阻尼 |
| 点火判据 | 基于能量约束估算制动高度, 15% 安全裕度 |
| 栅格翼 | 自由下落段侧向 PD 修正 (≤ 阻力 5%) |
| 姿态 | 由推力矢量方向推算 + 一阶低通滤波 |

### 箭体参数

| 参数 | 值 |
|------|-----|
| 干质量 | 25 800 kg |
| 着陆段燃料 | 7 000 kg |
| 箭体直径 | 3.66 m |
| 初始高度 | 3 000 m |
| 初始速度 | 1 000 km/h (277.8 m/s) |
| 目标触地速度 | −2 m/s |

### 飞行阶段

| 阶段 | 高度范围 | 主要作用 |
|------|----------|----------|
| 气动减速段 | 3000 m → ~1600 m | 大气阻力自然减速, 栅格翼水平修正 |
| 着陆制动段 | ~1600 m → 0 m | Merlin 1D 发动机制动, 闭环制导收敛 |

### 输出格式

`trajectory.csv` — 50 Hz, 13 列:

```text
time, x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az
```

坐标系: X = East, Y = North, Z = Up,  原点 = 着陆标志物中心

### 运行

```bash
# 生成轨迹
python generate_trajectory.py

# 生成轨迹 + 可视化图表
python generate_trajectory.py --plot

# 指定输出路径
python generate_trajectory.py --out my_trajectory.csv --plot
```

---

## Module 2 — 合成图像渲染 (`render_dataset.py`)

读取轨迹 CSV, 逐帧渲染箭载下视相机视角的合成图像。

### 渲染内容

- **地面**: 5000 m × 5000 m 混凝土纹理 (含裂缝、网格线)
- **标志物**: 外圆环 (直径 40 m, 环宽 2.5 m) + H 型符号
- **相机**: 透视投影, FOV = 60°
- **增广**: 高斯噪声、运动模糊 (高度自适应)

### 相机内参

```text
fx = fy = 1920 / (2·tan(30°)) ≈ 1662.8 px
cx = 960,  cy = 540
```

### 运行

```bash
# 渲染全量数据集
python render_dataset.py --traj trajectory.csv --out dataset

# 降采样 (每 3 帧取 1 帧)
python render_dataset.py --traj trajectory.csv --out dataset --skip 3

# 限制最大帧数
python render_dataset.py --traj trajectory.csv --out dataset --max_frames 200
```

### 导出视频 (`export_video.py`)

将 `dataset/rgb` 中的帧按 `pose.csv` 的仿真时间顺序合成为 MP4，**播放时长 = 仿真时长**（默认帧率 = 总帧数 / 总仿真时间，约 50 fps）。

```bash
# 导出到 dataset/landing_simulation.mp4（叠加时间与高度）
python export_video.py

# 指定数据集目录与输出文件名
python export_video.py --data dataset --out my_landing.mp4

# 不叠加时间/高度、固定 30 fps
python export_video.py --no-overlay --fps 30
```

---

## Module 3 — 训练流程 (`train.py`)

### 模型: `LandingMarkerDetector`

- 骨干网络: 8 层卷积 (步距 32), ~8M 参数
- 检测头: 3-anchor 单尺度 YOLO
- 损失: L = λ\_obj · BCE + λ\_noobj · BCE + λ\_box · MSE

### 运行

```bash
# 安装 PyTorch (如未安装)
pip install torch torchvision

# 训练
python train.py --data dataset/ --epochs 50 --batch 8

# 仅验证数据集 (无需 PyTorch)
python train.py --data dataset/
```

### 对接主流框架

```bash
# YOLOv8
yolo train data=dataset_yolo.yaml model=yolov8n.pt epochs=100
```

---

## 快速启动

```bash
# 0. 安装依赖
pip install -r requirements.txt

# 1. 生成轨迹 (约 15 秒)
python generate_trajectory.py --plot

# 2. 渲染数据集 (全量约 10~30 分钟)
python render_dataset.py --traj trajectory.csv --out dataset

# 3. 训练检测器 (需 PyTorch)
python train.py --data dataset/ --epochs 50 --batch 8
```

---

## 环境依赖

详见 `requirements.txt`。核心依赖:

| 包 | 版本 | 用途 |
|----|------|------|
| numpy | ≥ 1.24 | 数值计算 |
| opencv-python | ≥ 4.8 | 图像渲染 |
| Pillow | ≥ 10.0 | 图像 I/O |
| matplotlib | ≥ 3.7 | 轨迹可视化 (可选) |
| torch | ≥ 2.0 | 训练 (可选) |
| torchvision | ≥ 0.15 | 训练 (可选) |

> 轨迹生成器 (`generate_trajectory.py`) 仅依赖 Python 标准库 (`math`, `csv`),
> 无需额外安装即可运行。可视化功能需要 `matplotlib` 和 `numpy`。

---

*2026 年 3 月*
