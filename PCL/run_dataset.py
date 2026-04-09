"""
PCL 方法 —— 仿真数据集端到端测试
===================================
基于论文：Meng et al., "Satellite Pose Estimation via Single Perspective
Circle and Line," IEEE TAES, Vol.54, No.6, Dec.2018

输入数据源：
    datasets/rocket_render_0x/rgb/              —— 仿真 RGB 图像序列（RGBA格式）
    datasets/rocket_render_0x/mask/             —— 语义掩码（255=标记,1=台面,0=背景）
    datasets/rocket_render_0x/trajectory_0x.csv —— 飞行轨迹（世界坐标系）

图像与轨迹的对应关系：
    trajectory_0x.csv 行数与 rgb/ 图像数量一一对应（除表头外）

检测模式（--mode）：
    mask  —— 从语义掩码直接提取（精确，用于验证算法理论上限）
    rgb   —— 从 RGB 图像检测（CLAHE 增强 + 双边滤波 + 自适应 Canny）

输出：
    outputs/pcl_overview_{id}_{mode}.png   —— 4面板误差总览图
    outputs/pcl_detection_{id}_{mode}.png  —— 多高度关键帧检测可视化
    outputs/pcl_timeseries_{id}_{mode}.png —— 时序误差曲线

用法：
    python run_dataset.py [--dataset 01|02] [--sample N] [--mode mask|rgb]
"""

import argparse
import csv
import math
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse as MEllipse
from scipy.spatial.transform import Rotation

# ── 导入核心 PCL 算法 ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from pcl_pose_estimation import (  # noqa: E402
    build_camera_matrix,
    ellipse_to_matrix,
    recover_dual_poses_p1c,
    compute_plane_normal_from_line_image,
    recover_full_pose_case1,
    select_correct_pose,
)

plt.rcParams['font.family'] = 'DejaVu Sans'

# ═════════════════════════════════════════════════════════════════════════════
# 场景与传感器参数（与 Blender 渲染脚本保持一致）
# ═════════════════════════════════════════════════════════════════════════════

# ── 相机光学参数（来自 Blender OnboardCam 设置）
# cam.data.lens        = 24        → 焦距 24 mm
# cam.data.sensor_width = 36       → 传感器宽 36 mm（水平方向）
# cam.data.sensor_fit  = 'HORIZONTAL'
# 图像分辨率 img_w × img_h（运行时自动检测）
# fx = fy = lens_mm / (sensor_w_mm / img_w) = lens_mm * img_w / sensor_w_mm
BLENDER_LENS_MM   = 24.0    # Blender cam.data.lens (mm)
BLENDER_SENSOR_W  = 36.0    # Blender cam.data.sensor_width (mm)

# ── 降落区标记几何参数 (m)
# 默认值（防止配置缺失导致报错；最终会尝试从 marker_config.json 覆盖）
CIRCLE_R = 20.0         # 圆半径 (m)
H_HALF_W = 5.4          # H 形横杆半宽（用于位姿验证点）(m)

# 标志物几何配置（来自 Blender 设计先验）
_CONFIG_PATH = os.path.join(_HERE, "marker_config.json")
_DEBUG_LINE_EXTRACTION = False
_DEBUG_LINE_PRINT_LIMIT = 0
_DEBUG_LINE_PRINT_COUNT = 0
try:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)
    CIRCLE_R = float(_cfg["pcl"]["circle_radius_r"])
    H_HALF_W = float(_cfg["pcl"]["h_half_width"])
    _DEBUG_LINE_EXTRACTION = bool(_cfg.get("pcl", {}).get("debug_line_extraction", False))
    _DEBUG_LINE_PRINT_LIMIT = int(_cfg.get("pcl", {}).get("debug_line_print_limit", 3))
except FileNotFoundError:
    print(f"[WARN] 未找到 marker_config.json：{_CONFIG_PATH}，使用默认 CIRCLE_R/H_HALF_W。")
except Exception as e:
    print(f"[WARN] 读取 marker_config.json 失败（{e}），使用默认 CIRCLE_R/H_HALF_W。")

# ── 相机在箭体坐标系中的安装位置 (m)（来自 Blender cam.location）
# cam.location = (R_cam + 0.15,  0.0,  L_cam * 0.85)
# ROCKET_RADIUS ≈ 1.85 m，ROCKET_LENGTH ≈ 42 m（根据仿真脚本中的 R_cam / L_cam）
ROCKET_RADIUS = 1.85    # 与渲染脚本 ROCKET_RADIUS 一致 (m)
ROCKET_LENGTH = 42.0    # 与渲染脚本 ROCKET_LENGTH 一致 (m)
CAM_OFFSET = np.array([ROCKET_RADIUS + 0.15, 0.0, ROCKET_LENGTH * 0.85])

# ── 相机在箭体坐标系中的安装朝向（来自 Blender cam.rotation_euler）
# cam.rotation_mode = 'XYZ'
# cam.rotation_euler = (radians(15),  0,  radians(-90))
#
# Blender 相机局部坐标系：-Z 为视线方向，+Y 为上方
# OpenCV/PCL 相机坐标系：+Z 为视线方向，+Y 向下
# 转换关系：R_bl2cv = diag(1, -1, -1)
#
# 从 Blender 欧拉角计算：箭体坐标系 → OpenCV 相机坐标系
_R_CAM_BODY_BL = Rotation.from_euler(
    'xyz', [math.radians(15), 0.0, math.radians(-90)]).as_matrix()
_R_BL2CV = np.diag([1.0, -1.0, -1.0])          # Blender相机帧 → OpenCV相机帧

# R_BODY_TO_CAM：箭体坐标系 → OpenCV 相机坐标系（预计算，帧间复用）
# 用法：R_wc = R_BODY_TO_CAM @ R_body.T   （R_body 为箭体→世界旋转）
R_BODY_TO_CAM = _R_BL2CV @ _R_CAM_BODY_BL.T


# ═════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════════════════

def imread_unicode(path: str) -> np.ndarray:
    """
    兼容路径含中文字符的图像读取（cv2.imread 在 Windows 上不支持非 ASCII 路径）。
    使用 numpy 先以二进制读取文件，再由 cv2.imdecode 解码为 BGR 图像。
    """
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def build_K(img_w: int, img_h: int) -> np.ndarray:
    """
    由图像分辨率构造相机内参矩阵 K。

    参数来自 Blender 渲染脚本：
        lens = BLENDER_LENS_MM mm，sensor_width = BLENDER_SENSOR_W mm，
        sensor_fit = 'HORIZONTAL'  →  fx = fy = lens * img_w / sensor_w
    """
    fx = fy = BLENDER_LENS_MM * img_w / BLENDER_SENSOR_W
    return build_camera_matrix(fx, fy, img_w / 2.0, img_h / 2.0)


def body_to_world_cam(R_body: np.ndarray) -> np.ndarray:
    """
    由箭体姿态旋转矩阵计算世界→OpenCV相机的旋转矩阵 R_wc。

    推导：
        R_wc = R_BL2CV  @  R_cam_body^T  @  R_body^T
             = R_BODY_TO_CAM  @  R_body^T

    其中 R_body（箭体→世界）来自轨迹 Euler 角，
    R_cam_body 来自 Blender cam.rotation_euler = (15°, 0, -90°) XYZ。

    Parameters
    ----------
    R_body : np.ndarray (3,3)  箭体→世界旋转矩阵（来自轨迹 roll/pitch/yaw）

    Returns
    -------
    R_wc : np.ndarray (3,3)  世界→OpenCV相机坐标系旋转矩阵
    """
    return R_BODY_TO_CAM @ R_body.T


# ═════════════════════════════════════════════════════════════════════════════
# 检测模式 A：从语义掩码直接提取（精确）
# ═════════════════════════════════════════════════════════════════════════════

def detect_from_mask(mask: np.ndarray, img_w: int):
    global _DEBUG_LINE_PRINT_COUNT
    """
    从语义掩码中精确提取椭圆（圆环）和横杆直线。

    掩码值约定（Blender 渲染时写入）：
        255 → 白色标记像素（圆环轮廓 + 横杆）
        254 → 标记与台面的抗锯齿过渡像素
          1 → 降落台面（深灰色平台区域）
          0 → 真实背景（天空/远景）

    步骤：
        1. 提取 255 像素，膨胀后找轮廓
        2. 按面积和圆度打分，选圆环轮廓 → fitEllipse
        3. 在椭圆 ROI 内用 HoughLinesP 检测近水平横杆

    Returns
    -------
    ell   : OpenCV ellipse 元组，或 None
    l_hat : 直线齐次参数 [a,b,c]，或 None
    seg   : 线段端点 (x1,y1,x2,y2)，或 None
    """
    img_h = mask.shape[0]
    # 同时纳入抗锯齿过渡像素（254/255），减少漏边导致 Hough 候选不完整
    m255  = (mask >= 254).astype(np.uint8) * 255
    if m255.max() == 0:
        return None, None, None

    # 膨胀连接可能断裂的像素（1次即可，避免膨胀过大）
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m255 = cv2.dilate(m255, k, iterations=1)

    cnts, _ = cv2.findContours(m255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # ── 椭圆检测：找最大且形状接近圆的轮廓
    best_ell, best_score = None, 0.0
    for c in cnts:
        if len(c) < 15:
            continue
        try:
            ell = cv2.fitEllipse(c)
        except Exception:
            continue
        (_, _), (MA, ma), _ = ell
        if MA < 8 or ma < 8:
            continue
        ratio = ma / (MA + 1e-9)
        if ratio < 0.25:          # 过扁的轮廓是横杆，跳过
            continue
        score = math.pi * (MA / 2) * (ma / 2) * (ratio ** 0.4)
        if score > best_score:
            best_score, best_ell = score, ell

    if best_ell is None:
        return None, None, None

    # ── 横杆检测：在椭圆 ROI 内的 HoughLinesP
    (ex, ey), (MA, _), _ = best_ell
    pad = int(MA * 0.60)
    rx1, rx2 = max(0, int(ex - pad)), min(img_w, int(ex + pad))
    ry1, ry2 = max(0, int(ey - pad)), min(img_h, int(ey + pad))

    l_hat, best_seg = None, None
    if rx2 > rx1 and ry2 > ry1:
        roi_m = m255[ry1:ry2, rx1:rx2]
        lines = cv2.HoughLinesP(roi_m, 1, np.pi / 180, threshold=10,
                                 minLineLength=max(5, int(MA * 0.08)),
                                 maxLineGap=20)
        if lines is not None:
            # 评分：优先选择通过椭圆中心的候选直线（d_center 最小），
            # 次要准则选择更长的线段（seg_len 最大）
            best_key = (float("inf"), float("inf"))  # (d_center, -seg_len)
            best_a, best_b, best_c = None, None, None
            debug_candidates = [] if _DEBUG_LINE_EXTRACTION else None
            for ln in lines:
                ax1, ay1, ax2, ay2 = ln[0]
                seg_len = math.hypot(ax2 - ax1, ay2 - ay1)
                if seg_len < 1e-6:
                    continue

                # 将线段转换为图像直线齐次参数 [a,b,c]：a*u + b*v + c = 0
                gx1, gy1 = ax1 + rx1, ay1 + ry1
                gx2, gy2 = ax2 + rx1, ay2 + ry1
                dx, dy = gx2 - gx1, gy2 - gy1
                n = math.hypot(dx, dy) + 1e-12
                a = -dy / n
                b = dx / n
                c = -(a * gx1 + b * gy1)

                # 由于 a^2+b^2≈1，故 |a*ex+b*ey+c| 就是到椭圆中心的像素距离
                d_center = abs(a * ex + b * ey + c)

                if debug_candidates is not None:
                    debug_candidates.append((d_center, seg_len, gx1, gy1, gx2, gy2))

                key = (d_center, -seg_len)
                if key < best_key:
                    best_key = key
                    best_seg = (gx1, gy1, gx2, gy2)
                    best_a, best_b, best_c = a, b, c

        if best_seg is not None:
            if best_a is not None:
                l_hat = np.array([best_a, best_b, best_c])
            else:
                gx1, gy1, gx2, gy2 = best_seg
                dx, dy = gx2 - gx1, gy2 - gy1
                n     = math.hypot(dx, dy) + 1e-9
                a, b  = -dy / n, dx / n
                l_hat = np.array([a, b, -(a * gx1 + b * gy1)])

            # 仅前几帧打印候选排序，避免控制台刷屏
            if _DEBUG_LINE_EXTRACTION and _DEBUG_LINE_PRINT_COUNT < _DEBUG_LINE_PRINT_LIMIT:
                if debug_candidates is not None and len(debug_candidates) > 1:
                    debug_candidates.sort(key=lambda t: (t[0], -t[1]))
                    top = debug_candidates[:3]
                    print(f"[DEBUG][mask] topCandidates(ex={ex:.1f},ey={ey:.1f}): " +
                          ", ".join([f"(d={t[0]:.2f},len={t[1]:.1f})" for t in top]))
                    _DEBUG_LINE_PRINT_COUNT += 1

    return best_ell, l_hat, best_seg


# ═════════════════════════════════════════════════════════════════════════════
# 检测模式 B：从 RGB 图像检测（含预处理）
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_for_detection(img: np.ndarray) -> np.ndarray:
    """
    RGB/RGBA 图像预处理流水线，增强白色标记的可检测性。

    步骤：
        1. 丢弃 alpha 通道（如有），得到 BGR
        2. 转 LAB 色彩空间，对 L 通道做 CLAHE（局部对比度增强）
        3. 双边滤波去噪（保边，保留标记轮廓）
        4. 转灰度输出

    Returns
    -------
    gray : np.ndarray (H,W)  预处理后的灰度图
    """
    # 处理 RGBA 格式（alpha 全 255，直接丢弃）
    bgr = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img.copy()

    # LAB + CLAHE 增强局部对比度
    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lch, a, b = cv2.split(lab)
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq      = clahe.apply(lch)
    bgr_eq    = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    # 双边滤波（d=9，保边去噪）
    gray = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray


def detect_ellipse_rgb(gray: np.ndarray):
    """
    从预处理灰度图中检测圆形标记（椭圆投影）。

    相比原始版本的改进：
        - 自适应 Canny 阈值（基于图像中位灰度）
        - 形态学闭运算连接断裂的椭圆边缘
        - 轮廓长度要求提高（min 20 点）

    Returns
    -------
    OpenCV ellipse 元组，或 None
    """
    # 自适应 Canny 阈值
    med   = float(np.median(gray))
    lo    = max(10,  int(med * 0.4))
    hi    = min(220, int(med * 1.6))
    edges = cv2.Canny(gray, lo, hi)

    # 闭运算：连接圆弧上的小断口
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    best, best_score = None, 0.0
    for c in cnts:
        if len(c) < 20:
            continue
        try:
            ell = cv2.fitEllipse(c)
        except Exception:
            continue
        (_, _), (MA, ma), _ = ell
        if MA < 8 or ma < 8:
            continue
        ratio = ma / (MA + 1e-9)
        if ratio < 0.20:
            continue
        score = math.pi * (MA / 2) * (ma / 2) * (ratio ** 0.4)
        if score > best_score:
            best_score, best = score, ell
    return best


def detect_crossbar_rgb(gray: np.ndarray, ell, img_w: int):
    global _DEBUG_LINE_PRINT_COUNT
    """
    在椭圆 ROI 内从预处理灰度图检测水平横杆。

    Returns
    -------
    l_hat : 直线齐次参数 [a,b,c]，或 None
    seg   : 线段端点 (x1,y1,x2,y2)，或 None
    """
    if ell is None:
        return None, None
    img_h = gray.shape[0]
    (ex, ey), (MA, _), _ = ell
    pad = int(MA * 0.55)
    x1, x2 = max(0, int(ex - pad)), min(img_w, int(ex + pad))
    y1, y2 = max(0, int(ey - pad)), min(img_h, int(ey + pad))
    if x2 <= x1 or y2 <= y1:
        return None, None

    roi   = gray[y1:y2, x1:x2]
    edges = cv2.Canny(cv2.GaussianBlur(roi, (3, 3), 1), 25, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=12,
                             minLineLength=max(8, int(MA * 0.10)),
                             maxLineGap=15)
    if lines is None:
        return None, None

    # 评分：优先通过椭圆中心的候选直线（d_center 最小），
    # 次要准则选择更长的线段（seg_len 最大）
    best_key = (float("inf"), float("inf"))  # (d_center, -seg_len)
    best_a, best_b, best_c = None, None, None
    best_seg = None
    debug_candidates = [] if _DEBUG_LINE_EXTRACTION else None
    for ln in lines:
        ax1, ay1, ax2, ay2 = ln[0]
        seg_len = math.hypot(ax2 - ax1, ay2 - ay1)
        if seg_len < 1e-6:
            continue

        gx1, gy1 = ax1 + x1, ay1 + y1
        gx2, gy2 = ax2 + x1, ay2 + y1
        dx, dy = gx2 - gx1, gy2 - gy1
        n = math.hypot(dx, dy) + 1e-12
        a = -dy / n
        b = dx / n
        c = -(a * gx1 + b * gy1)

        d_center = abs(a * ex + b * ey + c)
        key = (d_center, -seg_len)
        if debug_candidates is not None:
            debug_candidates.append((d_center, seg_len, gx1, gy1, gx2, gy2))
        if key < best_key:
            best_key = key
            best_seg = (gx1, gy1, gx2, gy2)
            best_a, best_b, best_c = a, b, c

    if best_seg is None:
        return None, None

    if best_a is not None:
        l_hat = np.array([best_a, best_b, best_c])
    else:
        gx1, gy1, gx2, gy2 = best_seg
        dx, dy = gx2 - gx1, gy2 - gy1
        n     = math.hypot(dx, dy) + 1e-9
        a, b  = -dy / n, dx / n
        l_hat = np.array([a, b, -(a * gx1 + b * gy1)])

    if _DEBUG_LINE_EXTRACTION and _DEBUG_LINE_PRINT_COUNT < _DEBUG_LINE_PRINT_LIMIT:
        if debug_candidates is not None and len(debug_candidates) > 1:
            debug_candidates.sort(key=lambda t: (t[0], -t[1]))
            top = debug_candidates[:3]
            print(f"[DEBUG][rgb] topCandidates(ex={ex:.1f},ey={ey:.1f}): " +
                  ", ".join([f"(d={t[0]:.2f},len={t[1]:.1f})" for t in top]))
            _DEBUG_LINE_PRINT_COUNT += 1
    return l_hat, best_seg


# ═════════════════════════════════════════════════════════════════════════════
# 真值计算
# ═════════════════════════════════════════════════════════════════════════════

def compute_ground_truth(R_wc: np.ndarray,
                          cam_pos: np.ndarray,
                          lz: np.ndarray):
    """
    由轨迹数据计算当前帧的位姿真值。

    Parameters
    ----------
    R_wc    : 世界 → 相机坐标系旋转矩阵（look-at）
    cam_pos : 相机在世界坐标系中的位置 (m)
    lz      : 降落区圆心在世界坐标系中的位置 (m)

    Returns
    -------
    O_gt   : 圆心在相机坐标系中的位置向量 (m)
    nZB_gt : 圆平面法向量在相机坐标系中的方向（单位向量）
    """
    O_gt   = R_wc @ (lz - cam_pos)
    nZB_gt = R_wc @ np.array([0.0, 0.0, 1.0])
    if nZB_gt[2] < 0:
        nZB_gt = -nZB_gt
    return O_gt, nZB_gt


# ═════════════════════════════════════════════════════════════════════════════
# PCL 单帧估计（Case 1：横杆平行于局部 YB 轴）
# ═════════════════════════════════════════════════════════════════════════════

def run_pcl_single(ell, l_hat: np.ndarray,
                   K: np.ndarray,
                   circle_r: float,
                   h_half_w: float):
    """
    对单帧图像运行 PCL 位姿估计。

    仅使用 Case 1（直线平行于局部 YB 轴），适用于 H 形降落标记的横杆。
    双义性由横杆附近的一个验证点重投影误差消除。

    Parameters
    ----------
    ell      : OpenCV ellipse 元组
    l_hat    : 直线齐次参数 [a, b, c]
    K        : 相机内参矩阵
    circle_r : 圆半径 (m)
    h_half_w : H 标记半宽（验证点 Y 坐标，m）

    Returns
    -------
    pose : dict 含 'O_B', 'n_ZB', 'R_BC', 'n_XB', 'n_YB'，或 None（失败）
    """
    C_q   = ellipse_to_matrix(ell)
    duals = recover_dual_poses_p1c(C_q, K, circle_r)
    n_pi1 = compute_plane_normal_from_line_image(l_hat, K)

    # 对两组 P1C 解分别用 Case 1 恢复完整位姿
    candidates = []
    for sol in duals:
        try:
            pose = recover_full_pose_case1(sol, n_pi1)
            candidates.append(pose)
        except Exception:
            pass

    if not candidates:
        return None

    # 重投影验证：横杆上一点投影到直线的距离最小者为正确解
    P_verify = np.array([0.0, h_half_w * 0.8, 0.0])
    best, _ = select_correct_pose(candidates, P_verify, l_hat, K)
    return best


# ═════════════════════════════════════════════════════════════════════════════
# 主测试流程
# ═════════════════════════════════════════════════════════════════════════════

def run(dataset_id: str = '01', sample_step: int = 5, mode: str = 'mask'):
    """
    在指定数据集上运行完整 PCL 测试。

    Parameters
    ----------
    dataset_id  : 数据集编号（'01' 或 '02'）
    sample_step : 每隔 N 帧处理一帧（减少计算量）
    mode        : 检测模式 —— 'mask'（掩码精确检测）或 'rgb'（RGB图像检测）

    Returns
    -------
    res        : dict，各帧误差列表
    vis_frames : list，关键帧可视化数据
    """
    if mode not in ('mask', 'rgb'):
        raise ValueError(f"mode 必须为 'mask' 或 'rgb'，当前值：{mode}")

    # ── 路径构建
    datasets_root = os.path.join(_HERE, '..', 'datasets')
    dataset_dir   = os.path.join(datasets_root, f'rocket_render_{dataset_id}')
    rgb_dir       = os.path.join(dataset_dir, 'rgb')
    mask_dir      = os.path.join(dataset_dir, 'mask')
    csv_path      = os.path.join(dataset_dir, f'trajectory_{dataset_id}.csv')

    print("=" * 64)
    print(f"  PCL 位姿估计 —— 数据集 rocket_render_{dataset_id}  [{mode.upper()} 模式]")
    print("=" * 64)

    # ── 加载轨迹 CSV
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到轨迹文件：{csv_path}")
    traj = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            traj.append({k: float(v) for k, v in row.items()})
    print(f"  轨迹帧数：{len(traj)}")

    # ── 加载图像列表（按文件名排序）
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"未找到 rgb 目录：{rgb_dir}")
    img_files = sorted([
        os.path.join(rgb_dir, fn)
        for fn in os.listdir(rgb_dir)
        if fn.lower().endswith('.png')
    ])
    if not img_files:
        raise FileNotFoundError(f"rgb 目录中未找到 PNG 图像：{rgb_dir}")
    print(f"  图像数量：{len(img_files)}")

    # ── mask 模式：获取掩码文件列表（与 rgb 同名）
    mask_files = []
    if mode == 'mask':
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"未找到 mask 目录（mask 模式需要）：{mask_dir}")
        mask_files = sorted([
            os.path.join(mask_dir, fn)
            for fn in os.listdir(mask_dir)
            if fn.lower().endswith('.png')
        ])
        if not mask_files:
            raise FileNotFoundError(f"mask 目录中未找到 PNG 图像：{mask_dir}")
        print(f"  掩码数量：{len(mask_files)}")

    # ── 图像与轨迹帧对齐（直接按索引对应，截断多余部分）
    n_common = min(len(traj), len(img_files))
    if mode == 'mask' and mask_files:
        n_common = min(n_common, len(mask_files))
    traj       = traj[:n_common]
    img_files  = img_files[:n_common]
    if mask_files:
        mask_files = mask_files[:n_common]

    # ── 自动检测图像分辨率
    first_img = imread_unicode(img_files[0])
    if first_img is None:
        raise IOError(f"无法读取图像：{img_files[0]}")
    img_h, img_w = first_img.shape[:2]
    K_mat = build_K(img_w, img_h)
    print(f"  图像分辨率：{img_w}×{img_h}")

    # ── 降落区中心（轨迹终点 xy，地面 z=0）
    lz = np.array([traj[-1]['x'], traj[-1]['y'], 0.0])
    h_start, h_end = traj[0]['z'], traj[-1]['z']
    print(f"  降落区：({lz[0]:.2f}, {lz[1]:.2f}) m  高度范围：{h_end:.1f}~{h_start:.0f} m")

    # ── 采样帧索引
    frame_indices = list(range(0, n_common, sample_step))
    print(f"  处理帧数：{len(frame_indices)}（每 {sample_step} 帧取 1）")

    # ── 结果容器
    res = {k: [] for k in ['h', 'pos_err', 'ang_err', 'Oz_err',
                             'reproj', 'det', 'O_est', 'O_gt',
                             'nZB_est', 'nZB_gt']}
    vis_frames = []
    n_det_fail, n_pcl_fail = 0, 0

    print(f"\n  {'帧':>5} {'高度(m)':>9} {'位置误差':>9} {'法向误差':>9} "
          f"{'Reproj':>8}  状态")
    print("  " + "-" * 57)

    for fi, idx in enumerate(frame_indices):
        pt       = traj[idx]
        img_path = img_files[idx]
        h        = pt['z']

        # ── 计算相机位置和朝向（世界坐标系）
        pos_w   = np.array([pt['x'], pt['y'], pt['z']])
        R_body  = Rotation.from_euler(
            'xyz', [pt['roll'], pt['pitch'], pt['yaw']]).as_matrix()
        cam_pos = pos_w + R_body @ CAM_OFFSET   # 相机在世界坐标系中的位置
        R_wc    = body_to_world_cam(R_body)      # 世界→OpenCV相机旋转（精确，基于Blender参数）

        # ── 读图 & 检测
        img = imread_unicode(img_path)
        if mode == 'mask':
            # 直接从语义掩码提取
            mask_path = mask_files[idx]
            mask      = imread_unicode(mask_path)
            if mask is not None and mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            ell, l_hat, lseg = (detect_from_mask(mask, img_w)
                                 if mask is not None else (None, None, None))
        else:
            # RGB 图像：预处理 → 椭圆检测 → 横杆检测
            gray  = preprocess_for_detection(img) if img is not None else None
            ell   = detect_ellipse_rgb(gray)      if gray is not None else None
            l_hat, lseg = (detect_crossbar_rgb(gray, ell, img_w)
                           if ell is not None else (None, None))
        det_ok = ell is not None and l_hat is not None

        res['det'].append(int(det_ok))
        res['h'].append(h)

        if not det_ok:
            n_det_fail += 1
            for k in ('pos_err', 'ang_err', 'Oz_err', 'reproj'):
                res[k].append(np.nan)
            res['O_est'].append(None)
            res['O_gt'].append(None)
            res['nZB_est'].append(None)
            res['nZB_gt'].append(None)
            continue

        # ── PCL 位姿估计
        pose = run_pcl_single(ell, l_hat, K_mat, CIRCLE_R, H_HALF_W)
        if pose is None:
            n_pcl_fail += 1
            for k in ('pos_err', 'ang_err', 'Oz_err', 'reproj'):
                res[k].append(np.nan)
            res['O_est'].append(None)
            res['O_gt'].append(None)
            res['nZB_est'].append(None)
            res['nZB_gt'].append(None)
            continue

        # ── 误差计算
        O_gt, nZB_gt = compute_ground_truth(R_wc, cam_pos, lz)
        O_est        = pose['O_B']
        nZB_est      = pose['n_ZB']

        pos_err = np.linalg.norm(O_est - O_gt)

        dot     = np.clip(
            np.dot(nZB_est / np.linalg.norm(nZB_est),
                   nZB_gt  / np.linalg.norm(nZB_gt)), -1.0, 1.0)
        ang_err = math.degrees(math.acos(abs(dot)))
        Oz_err  = abs(O_est[2] - O_gt[2])

        # 验证点重投影到直线的距离（衡量内部一致性）
        P_v = np.array([0.0, H_HALF_W * 0.8, 0.0])
        pc  = pose['R_BC'].T @ P_v + O_est
        if pc[2] > 0.1:
            q      = K_mat @ pc
            p      = q[:2] / q[2]
            ph     = np.array([p[0], p[1], 1.0])
            reproj = abs(l_hat @ ph) / (math.hypot(l_hat[0], l_hat[1]) + 1e-12)
        else:
            reproj = np.nan

        res['pos_err'].append(pos_err)
        res['ang_err'].append(ang_err)
        res['Oz_err'].append(Oz_err)
        res['reproj'].append(reproj)
        res['O_est'].append(O_est)
        res['O_gt'].append(O_gt)
        res['nZB_est'].append(nZB_est)
        res['nZB_gt'].append(nZB_gt)

        # 收集关键帧用于可视化（每个目标高度取一帧）
        for target_h in [2500, 1500, 800, 300, 100, 50, 15, 3]:
            if (abs(h - target_h) < 35 and
                    not any(abs(vf['h'] - target_h) < 25 for vf in vis_frames)):
                vis_frames.append({
                    'h': h, 'img': img.copy(),
                    'ell': ell, 'line': lseg,
                    'pos_err': pos_err, 'ang_err': ang_err,
                })

        if fi % 40 == 0 or fi < 3:
            print(f"  {fi:5d} {h:9.1f}m {pos_err:9.3f}m "
                  f"{ang_err:9.3f}° {reproj:8.2f}px  OK")

    # ── 汇总统计
    pe = np.array([v for v in res['pos_err'] if not np.isnan(v)])
    ae = np.array([v for v in res['ang_err'] if not np.isnan(v)])
    oz = np.array([v for v in res['Oz_err']  if not np.isnan(v)])
    dr = np.mean(res['det']) * 100

    print("\n" + "=" * 64)
    print("  汇总统计")
    print("=" * 64)
    print(f"  总帧数：{len(frame_indices)}  检测成功率：{dr:.1f}%")
    print(f"  检测失败：{n_det_fail}帧  PCL 失败：{n_pcl_fail}帧")

    if len(pe):
        print(f"  位置误差：均值={pe.mean():.3f}m  中位={np.median(pe):.3f}m"
              f"  std={pe.std():.3f}m  最大={pe.max():.3f}m")
        print(f"  深度误差：均值={oz.mean():.3f}m  中位={np.median(oz):.3f}m")
        print(f"  法向误差：均值={ae.mean():.3f}°  中位={np.median(ae):.3f}°"
              f"  最大={ae.max():.3f}°")
        print(f"  位置误差 <1m：{np.mean(pe < 1) * 100:.1f}%  "
              f"<5m：{np.mean(pe < 5) * 100:.1f}%  "
              f"法向误差 <5°：{np.mean(ae < 5) * 100:.1f}%")

        print(f"\n  {'高度区间':^14} {'帧数':>5} {'检测率':>7} "
              f"{'位置(m)':>9} {'深度(m)':>9} {'法向(°)':>9}")
        for lo, hi in [(2000, 3500), (1000, 2000), (500, 1000),
                       (100, 500), (10, 100), (0, 10)]:
            idx_h = [i for i, hv in enumerate(res['h']) if lo <= hv < hi]
            if not idx_h:
                continue
            dv = np.mean([res['det'][i] for i in idx_h]) * 100
            pv = [res['pos_err'][i] for i in idx_h
                  if not np.isnan(res['pos_err'][i])]
            ov = [res['Oz_err'][i]  for i in idx_h
                  if not np.isnan(res['Oz_err'][i])]
            av = [res['ang_err'][i] for i in idx_h
                  if not np.isnan(res['ang_err'][i])]
            print(f"  {lo:4d}~{hi:4d}m    {len(idx_h):5d} {dv:7.1f}% "
                  f"{np.mean(pv) if pv else float('nan'):9.3f} "
                  f"{np.mean(ov) if ov else float('nan'):9.3f} "
                  f"{np.mean(av) if av else float('nan'):9.3f}")

    return res, vis_frames


# ═════════════════════════════════════════════════════════════════════════════
# 绘图与保存
# ═════════════════════════════════════════════════════════════════════════════

def make_plots(res: dict, vis_frames: list,
               output_dir: str, dataset_id: str = '01',
               mode: str = 'mask') -> None:
    """生成三张统计图表并保存到 output_dir。"""
    os.makedirs(output_dir, exist_ok=True)

    heights = np.array(res['h'])
    pe      = np.array(res['pos_err'])
    ae      = np.array(res['ang_err'])
    rp      = np.array(res['reproj'])
    det     = np.array(res['det'], bool)
    valid   = ~np.isnan(pe)
    tag     = f"rocket_render_{dataset_id} [{mode.upper()}]"
    suffix  = f"{dataset_id}_{mode}"

    # ── 图1：4 面板总览
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'PCL Method — Pose Estimation on {tag}',
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(heights[valid], pe[valid], s=8,
               c=heights[valid], cmap='plasma_r', alpha=0.7)
    ax.axhline(1, color='red',    ls='--', lw=1.5, label='1m target')
    ax.axhline(5, color='orange', ls=':',  lw=1.2, label='5m target')
    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error vs Height')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(heights[valid], ae[valid], s=8,
               c=heights[valid], cmap='plasma_r', alpha=0.7)
    ax.axhline(5, color='red', ls='--', lw=1.5, label='5° target')
    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Normal Angle Error (°)')
    ax.set_title('Circle Normal Direction Error vs Height')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    data = pe[valid]
    if len(data) > 0:
        bins = np.linspace(0, min(np.percentile(data, 97), 100), 70)
        ax.hist(data, bins=bins, color='#2196F3', alpha=0.7,
                edgecolor='white', label='Histogram')
        med = np.median(data)
        ax.axvline(med, color='navy',   lw=2,    label=f'Median={med:.3f}m')
        ax.axvline(1.0, color='red',    ls='--', lw=1.5, label='1m target')
        ax2 = ax.twinx()
        sd  = np.sort(data)
        cdf = np.arange(1, len(sd) + 1) / len(sd) * 100
        ax2.plot(sd, cdf, 'r-', lw=2, label='CDF')
        ax2.axhline(90, color='orange', ls=':', lw=1)
        ax2.set_ylabel('CDF (%)')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_xlabel('Position Error (m)')
    ax.set_ylabel('Count')
    ax.set_title('Position Error Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    bnds = np.logspace(np.log10(max(0.5, heights.min())),
                        np.log10(heights.max()), 15)
    bdr, brp, bcx = [], [], []
    for bi in range(len(bnds) - 1):
        m = (heights >= bnds[bi]) & (heights < bnds[bi + 1])
        if m.sum() == 0:
            continue
        bdr.append(det[m].mean() * 100)
        rv = rp[m & ~np.isnan(rp)]
        brp.append(rv.mean() if len(rv) else 0.0)
        bcx.append((bnds[bi] + bnds[bi + 1]) / 2)
    x = range(len(bdr))
    ax.bar(x, bdr, color='#7B1FA2', alpha=0.7, label='Det rate')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c:.0f}' for c in bcx], rotation=45, fontsize=7)
    ax.set_xlabel('Height bin (m)')
    ax.set_ylabel('Detection Rate (%)', color='#7B1FA2')
    ax.set_ylim(0, 108)
    ax.grid(True, alpha=0.3, axis='y')
    ax3 = ax.twinx()
    ax3.plot(x, brp, 'o-', color='#F44336', lw=2, ms=5, label='Reproj err')
    ax3.set_ylabel('Reproj Error (px)', color='#F44336')
    ax.set_title('Detection Rate & Reprojection Error')

    plt.tight_layout()
    out1 = os.path.join(output_dir, f'pcl_overview_{suffix}.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存：{out1}")

    # ── 图2：关键帧检测可视化
    vfs = sorted(vis_frames, key=lambda f: f['h'], reverse=True)[:8]
    if vfs:
        nc = 4
        nr = math.ceil(len(vfs) / nc)
        fig2, axes2 = plt.subplots(nr, nc, figsize=(16, nr * 3.5))
        axes2 = axes2.flatten()
        fig2.suptitle(
            f'PCL Detection at Different Heights — {tag} '
            f'(Cyan=Ellipse, Red=Crossbar)',
            fontsize=12, fontweight='bold')
        for ai, vf in enumerate(vfs):
            ax = axes2[ai]
            rgb = cv2.cvtColor(vf['img'], cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            if vf['ell']:
                (ex, ey), (MA, ma), ang = vf['ell']
                ax.add_patch(MEllipse((ex, ey), MA, ma, angle=ang,
                                      fill=False, edgecolor='cyan',
                                      lw=2, ls='--'))
                ax.plot(ex, ey, 'c+', ms=12, mew=2)
            if vf['line']:
                lx1, ly1, lx2, ly2 = vf['line']
                ax.plot([lx1, lx2], [ly1, ly2], 'r-', lw=2.5)
            pe_v = vf['pos_err']
            col  = 'lime' if pe_v < 1 else ('orange' if pe_v < 5 else 'red')
            ax.set_title(f"h={vf['h']:.0f}m\n"
                         f"PosErr={pe_v:.3f}m  Ang={vf['ang_err']:.2f}°",
                         fontsize=8, color=col)
            ax.axis('off')
        for ai in range(len(vfs), len(axes2)):
            axes2[ai].axis('off')
        plt.tight_layout()
        out2 = os.path.join(output_dir, f'pcl_detection_{suffix}.png')
        plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  已保存：{out2}")

    # ── 图3：时序误差曲线
    fig3, axes3 = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig3.suptitle(f'PCL Error Profile Along Descent Trajectory — {tag}',
                  fontsize=13, fontweight='bold')
    fids = np.arange(len(heights))

    ax = axes3[0]
    ax.semilogy(fids, heights, '-', color='#FF9800', lw=1.5)
    ax.fill_between(fids, 1, heights, alpha=0.15, color='#FF9800')
    ax.set_ylabel('Height (m)')
    ax.set_title('Flight Height')
    ax.grid(True, alpha=0.3)

    ax = axes3[1]
    if valid.any():
        ax.plot(fids[valid], pe[valid], '-', color='#2196F3', lw=1, alpha=0.85)
        ax.fill_between(fids[valid], 0, pe[valid], alpha=0.18, color='#2196F3')
        ax.set_title(f'Position Error  mean={pe[valid].mean():.3f}m  '
                     f'{np.mean(pe[valid] < 1) * 100:.1f}%<1m')
    else:
        ax.set_title('Position Error')
    ax.axhline(1, color='red',    ls='--', lw=1.5, label='1m target')
    ax.axhline(5, color='orange', ls=':',  lw=1.2, label='5m target')
    ax.set_ylabel('Pos Error (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes3[2]
    if valid.any():
        ae_arr = np.array(res['ang_err'])
        ax.plot(fids[valid], ae_arr[valid], '-', color='#4CAF50', lw=1, alpha=0.85)
        ax.fill_between(fids[valid], 0, ae_arr[valid],
                        alpha=0.18, color='#4CAF50')
        ax.set_title(f'Circle Normal Direction Error  '
                     f'mean={ae_arr[valid].mean():.3f}°  '
                     f'{np.mean(ae_arr[valid] < 5) * 100:.1f}%<5°')
    else:
        ax.set_title('Circle Normal Direction Error')
    ax.axhline(5, color='red', ls='--', lw=1.5, label='5° target')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Normal Angle Err (°)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out3 = os.path.join(output_dir, f'pcl_timeseries_{suffix}.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存：{out3}")


# ═════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PCL 方法在仿真数据集上的端到端测试')
    parser.add_argument(
        '--dataset', default='01', choices=['01', '02'],
        help='数据集编号（默认 01）')
    parser.add_argument(
        '--sample', type=int, default=5,
        help='帧采样步长，每 N 帧处理 1 帧（默认 5）')
    parser.add_argument(
        '--mode', default='mask', choices=['mask', 'rgb'],
        help='检测模式：mask=掩码精确检测（默认），rgb=RGB图像检测')
    args = parser.parse_args()

    output_dir = os.path.join(_HERE, 'outputs')

    res, vis = run(dataset_id=args.dataset,
                   sample_step=args.sample,
                   mode=args.mode)

    print("\n  正在生成图表...")
    make_plots(res, vis, output_dir,
               dataset_id=args.dataset, mode=args.mode)
    print("\n[完成]")
