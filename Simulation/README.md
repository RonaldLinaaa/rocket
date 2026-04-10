# Simulation — 轨迹仿真与 Blender 渲染脚本

本目录包含两类内容：

1. **Python 轨迹仿真**：在本地解释器中运行，生成 50 Hz 的六自由度 CSV，供 Blender 读入。
2. **`rocket_trajectory_blender.py`**：在 **Blender 5.0.x** 的 Scripting 工作区中运行，清空并搭建沙漠着陆场景、箭体与机载相机，按 CSV 关键帧驱动动画，并配置 Cycles 渲染输出。

**数据与标注文件**（渲染序列、`kp.json`、COCO JSON、预训练权重等）放在仓库 **`datasets/`** 下；其中大量路径已写入根目录 **`.gitignore`**，本地生成后使用，不一定随 Git 提交。

---

## 与全项目数据流水线的关系

```text
Simulation/
  generate_trajectory.py  ──►  trajectory.csv  (默认名，50 Hz, 6-DoF)
        │
        │  （可选）generate_vertical_recovery_traj.py  ──►  traj.csv（姿态扰动）
        ▼
  编辑 rocket_trajectory_blender.py 内 CSV_PATH，使其指向上述 CSV
        │
        ▼
  Blender 5.0：Run Script → 预览 / Ctrl+F12 渲染图像序列
        │
        ▼
datasets/
  rocket_render_01/、rocket_render_02/  …  （各序列 rgb 等，见 .gitignore）
  kp.json、annotations_coco_*.json       …  （标注与划分，见 .gitignore）
  generate_coco_keypoints.py 等           （标注生成，若置于该目录）
```

**坐标约定**：世界系 X 东、Y 北、Z 上；着陆区中心与标志物几何与 **`datasets/kp.json`**、项目根 **`README.md`** 中关键点表一致（`kp.json` 常被忽略不入库，以你本地文件为准）。

---

## 目录结构（本目录）

```text
Simulation/
├── README.md
├── requirements.txt              # 轨迹 --plot 可选依赖（numpy、matplotlib）
├── generate_trajectory.py        # Falcon 9 一级着陆段量级 6-DoF 数值仿真
├── generate_vertical_recovery_traj.py  # 可选：在 CSV 上叠加姿态摆动
└── rocket_trajectory_blender.py  # Blender 5.0：场景 + 关键帧 + Cycles 渲染参数
```

**生成物**（如 `trajectory.csv`、`traj.csv`、`*_plot.png`）及 Blender 相对路径下的 `rocket_render/` 等，若不需要可删除；**`Simulation/dataset/`** 已列入 `.gitignore`。

---

## `generate_trajectory.py`

基于 Falcon 9 Block 5 一级着陆段量级参数数值积分，输出理想条件下的下降轨迹。

### 输出 CSV

表头为：

```text
time, x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az
```

单位为：s、m、rad、m/s、m/s²；采样率 **50 Hz**。

### 运行

```bash
conda activate rocket
cd Simulation

python generate_trajectory.py
python generate_trajectory.py --plot
python generate_trajectory.py --out trajectory_01.csv --plot
```

脚本内 **`CSV_PATH` 默认示例**（见 `rocket_trajectory_blender.py`）曾使用 `trajectory_01.csv`；若你使用默认 `trajectory.csv`，请在 Blender 脚本中把 `CSV_PATH` 改为同一路径，或先用 `--out` 生成与脚本一致的文件名。

---

## `generate_vertical_recovery_traj.py`（可选）

在**不改变**位置、速度、加速度的前提下，对姿态角叠加受高度包络调制的扰动；近地面衰减至零。

```bash
python generate_vertical_recovery_traj.py --src trajectory.csv --out traj.csv
```

默认读取当前目录下 **`trajectory.csv`**，写出 **`traj.csv`**。

---

## `rocket_trajectory_blender.py`（Blender 内运行）

- **环境**：Blender **5.0.x**，内置 `bpy`，勿用系统 `python` 直接执行。
- **使用前必改**：文件顶部 **`BASE_DIR`**、**`CSV_PATH`**（指向上一步生成的 CSV）、**`ENV_DIR`**（外部 `.blend` 环境资源路径，以你本机为准）。
- **场景要点**（脚本注释摘要）：箭体喷口朝下软着陆；起始约 (50, −25, 3000) m，落点附近受控；地面为沙漠地形资产；着陆区为圆形标志 + **H 字**标志（海拔 0）；箭体尺寸与真实脚本内常量一致。
- **时间轴**：`FPS = 50`，每行轨迹数据对应一帧；`KEYFRAME_STEP` 控制隔多少行插关键帧（默认 1 即每行一关键帧）。
- **相机**：安装在箭体局部坐标系，欧拉角约 (15°, 0, −90°)（XYZ），下视着陆区。
- **渲染**：Cycles，**1080×720**，128 samples，降噪；**运动模糊**开启；输出 PNG 序列，默认相对路径 **`//rocket_render//`**（相对于当前 Blender 工程文件目录）。可将输出目录整理到 **`datasets/rocket_render_01/rgb/`** 等供训练脚本读取。

---

## `datasets/` 目录（仓库根下，与 Simulation 配合）

逻辑上用于存放：

| 内容 | 说明 |
|------|------|
| `rocket_render_01/`、`rocket_render_02/` | 各轨迹渲染得到的图像序列等 |
| `visualizations/` | 可选可视化输出 |
| `kp.json` | 9 个关键点 3D 坐标 |
| `annotations_coco_keypoints.json` 及 train/val/test 划分 | COCO 关键点标注 |
| `sequences.example.json` | 序列列表示例（若使用） |
| `pretrain/` | RT-DETR / YOLO 预训练权重 |

上述路径**多数在 `.gitignore 中**，以本地实际为准；论文或协作时需说明「生成方式」而非假定仓库内必含二进制数据。

---

## 依赖（仅轨迹 Python 脚本）

| 脚本 | 运行环境 | 可选 |
|------|-----------|------|
| `generate_trajectory.py` | 标准库即可 | `numpy`、`matplotlib`（`--plot`） |
| `generate_vertical_recovery_traj.py` | 标准库 | 无 |
| `rocket_trajectory_blender.py` | **仅 Blender 自带 Python** | — |

```bash
pip install -r requirements.txt
```

---

## 论文撰写时可对应的小节

1. **标志物与坐标系**：H 字/圆形着陆区、`kp.json` 中 9 点定义。  
2. **仿真轨迹**：本目录 `generate_trajectory.py` 物理模型、50 Hz、终端约束；可选姿态扰动脚本。  
3. **仿真场景**：Blender 5.0、`rocket_trajectory_blender.py`、Cycles 与分辨率/运动模糊。  
4. **数据采集**：渲染 PNG 序列组织、`datasets/` 中序列目录命名。  
5. **标注**：COCO 关键点 JSON 及划分脚本。  
6. **数据集特性**：轨迹条数、帧数、7:2:1 划分、距离分层评测等（与评测章节一致）。

---

*与当前仓库 `.gitignore` 及目录布局对齐：2026 年 4 月*
