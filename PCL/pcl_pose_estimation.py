"""
PCL (Perspective Circle and Line) Pose Estimation
==================================================
基于论文：
  Meng et al., "Satellite Pose Estimation via Single Perspective Circle and Line,"
  IEEE Transactions on Aerospace and Electronic Systems, Vol. 54, No. 6, Dec. 2018.

本代码实现了完整的 PCL 方法，包括：
  1. 从单目图像的椭圆投影恢复圆的双义位姿（P1C 问题）
  2. 利用空间直线的图像投影消除双义性并恢复滚转角
  3. 支持三种直线与圆所在平面的位置关系：
       - Case 1: 直线平行于局部坐标轴 YB
       - Case 2: 直线平行于圆所在平面但不平行于 YB
       - Case 3: 直线不平行于圆所在平面

依赖：numpy, opencv-python, scipy
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation


# =============================================================================
# 第一部分：P1C —— 从单张图像中的椭圆恢复圆的双义位姿
# =============================================================================

def fit_ellipse_from_image(gray_image):
    """
    从灰度图像中自动提取最大的椭圆轮廓并返回椭圆参数矩阵 C_q。

    Parameters
    ----------
    gray_image : np.ndarray (H, W), uint8
        输入灰度图像。

    Returns
    -------
    C_q : np.ndarray (3, 3)
        椭圆的二次型矩阵（像素坐标系下）。
    ellipse : tuple
        OpenCV fitEllipse 返回的元组 ((cx,cy), (a,b), angle_deg)，
        其中 a, b 为长/短轴全长（像素），angle_deg 为倾斜角（度）。
    """
    # 二值化 + Canny 边缘
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 选取点数最多且足够拟合椭圆的轮廓
    best = max((c for c in contours if len(c) >= 5), key=len, default=None)
    if best is None:
        raise ValueError("图像中未找到有效轮廓，无法拟合椭圆。")

    ellipse = cv2.fitEllipse(best)  # ((cx,cy), (MA,ma), angle)
    C_q = ellipse_to_matrix(ellipse)
    return C_q, ellipse


def ellipse_to_matrix(ellipse):
    """
    将 OpenCV ellipse 元组转换为 3×3 齐次二次型矩阵 C_q，
    满足 [u v 1] C_q [u v 1]^T = 0。

    Parameters
    ----------
    ellipse : tuple
        OpenCV fitEllipse 的返回值：((cx, cy), (MA, ma), angle_deg)
        MA = 长轴全长，ma = 短轴全长，angle_deg = 长轴与水平方向夹角（度）。

    Returns
    -------
    C_q : np.ndarray (3, 3)，对称矩阵
    """
    (cx, cy), (MA, ma), angle_deg = ellipse
    a = MA / 2.0   # 长半轴
    b = ma / 2.0   # 短半轴
    theta = np.deg2rad(angle_deg)

    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # 一般二次曲线系数：Au^2 + Buv + Cv^2 + Du + Ev + F = 0
    # 椭圆标准形转一般形
    A = (cos_t / a) ** 2 + (sin_t / b) ** 2
    B = 2 * cos_t * sin_t * (1 / a ** 2 - 1 / b ** 2)
    C = (sin_t / a) ** 2 + (cos_t / b) ** 2
    D = -2 * A * cx - B * cy
    E = -B * cx - 2 * C * cy
    F = A * cx ** 2 + B * cx * cy + C * cy ** 2 - 1.0

    # 论文式 (2): C_q 的对称矩阵形式
    C_q = np.array([
        [A,     B / 2,  D / 2],
        [B / 2, C,      E / 2],
        [D / 2, E / 2,  F    ]
    ])
    return C_q


def ellipse_params_to_matrix(a_coef, b_coef, c_coef, d_coef, e_coef, f_coef):
    """
    直接由一般二次曲线系数构造 C_q 矩阵。
    对应论文式 (1): au^2 + bv^2 + cuv + du + ev + f = 0

    Parameters
    ----------
    a_coef, b_coef, c_coef, d_coef, e_coef, f_coef : float
        椭圆的一般二次曲线系数。

    Returns
    -------
    C_q : np.ndarray (3, 3)
    """
    C_q = np.array([
        [a_coef,       c_coef / 2,  d_coef / 2],
        [c_coef / 2,   b_coef,      e_coef / 2],
        [d_coef / 2,   e_coef / 2,  f_coef     ]
    ])
    return C_q


def recover_dual_poses_p1c(C_q, K, R):
    """
    P1C：由已知半径 R 的圆在标定相机中的椭圆投影，
    恢复圆心位置和法向量的两组双义解。

    对应论文 Section II-B，式 (5)(6)。

    Parameters
    ----------
    C_q : np.ndarray (3, 3)
        椭圆二次型矩阵（像素坐标系）。
    K   : np.ndarray (3, 3)
        相机内参矩阵。
    R   : float
        圆的已知半径（与场景单位一致，如 mm）。

    Returns
    -------
    solutions : list of dict，长度为 2
        每个 dict 包含：
            'O_B'   : np.ndarray (3,)  圆心在相机坐标系下的位置
            'n_ZB'  : np.ndarray (3,)  圆法向量（单位向量），朝向相机（与 ZC 夹角为锐角）
    """
    # 论文式 (4): Q_tilde = K^T * C_q * K  —— 相机坐标系下椭圆锥的二次型
    Q_tilde = K.T @ C_q @ K   # shape (3,3)

    # 对 Q_tilde 做特征值分解，得到旋转矩阵 U 使 Q_tilde 对角化
    eigenvalues, eigenvectors = np.linalg.eigh(Q_tilde)

    # 按升序排列特征值：应满足 λ1 ≥ λ2 > 0 > λ3
    idx = np.argsort(eigenvalues)[::-1]          # 降序
    lam = eigenvalues[idx]                        # [λ1, λ2, λ3]
    vecs = eigenvectors[:, idx]                  # 对应特征向量列

    lam1, lam2, lam3 = lam[0], lam[1], lam[2]

    # 验证特征值符号：λ1 ≥ λ2 > 0 > λ3
    if not (lam1 >= lam2 > 0 > lam3):
        raise ValueError(
            f"特征值符号不满足 λ1≥λ2>0>λ3，请检查椭圆参数或相机内参。"
            f"\n特征值：{lam}"
        )

    # 构造旋转矩阵 U，使椭圆锥轴对齐坐标轴
    # 论文约定：e3 对应 λ3（最小，负值），方向使 [e3]_3 > 0
    e3 = vecs[:, 2]
    if e3[2] < 0:
        e3 = -e3
    e2 = vecs[:, 1]
    e1 = np.cross(e2, e3)
    e1 /= np.linalg.norm(e1)
    U = np.column_stack([e1, e2, e3])   # (3,3)

    # 论文式 (5)：在 O_C-X'Y'Z' 坐标系中两组圆心和法向量（±号对应两解）
    # 为书写简洁，定义中间量
    abs1, abs2, abs3 = abs(lam1), abs(lam2), abs(lam3)

    # 圆心 Z' 坐标（深度，恒为正）
    Oz_prime = R * np.sqrt(abs1 * (abs2 + abs3) / (abs3 * (abs1 + abs3)))

    # 圆心 X' 坐标（±对应两解）
    Ox_prime = R * np.sqrt(abs3 * (abs1 - abs2) / (abs1 * (abs1 - abs3)))

    # 法向量分量
    nz_prime = -np.sqrt((abs2 + abs3) / (abs1 + abs3))
    nx_prime =  np.sqrt((abs1 - abs2) / (abs1 + abs3))

    solutions = []
    for sign in [+1, -1]:
        # 在 X'Y'Z' 坐标系中的圆心和法向量
        O_prime = np.array([sign * Ox_prime, 0.0, Oz_prime])
        n_prime = np.array([sign * nx_prime, 0.0, nz_prime])

        # 论文式 (6)：转换回相机坐标系 {C}
        O_B = U @ O_prime
        n_ZB = U @ n_prime
        n_ZB /= np.linalg.norm(n_ZB)

        # 确保法向量与光轴夹角为锐角（θ < 90°）
        if n_ZB[2] < 0:
            n_ZB = -n_ZB

        solutions.append({'O_B': O_B, 'n_ZB': n_ZB})

    return solutions


# =============================================================================
# 第二部分：利用直线消除双义性并恢复滚转角
# =============================================================================

def compute_plane_normal_from_line_image(l_hat, K):
    """
    由直线图像坐标计算平面 π1（由 OC 和直线图像 ˜l 确定）的法向量。

    对应论文 Section II-C-1-a，πˆ1 = M^T * ˆl。

    Parameters
    ----------
    l_hat : np.ndarray (3,)
        直线的齐次参数向量 [a, b, c]（图像直线方程 au+bv+c=0）。
    K     : np.ndarray (3, 3)
        相机内参矩阵。

    Returns
    -------
    n_pi1 : np.ndarray (3,)
        平面 π1 的单位法向量（约定与 XC 轴夹角为锐角）。
    """
    # 投影矩阵 M = [K | 0]（世界坐标系 = 相机坐标系时）
    M = np.hstack([K, np.zeros((3, 1))])   # (3,4)

    # πˆ1 = M^T * ˆl，取前三个分量作为法向量
    pi1 = M.T @ l_hat        # (4,)
    n_pi1 = pi1[:3].copy()   # (3,)

    # 约定：n_pi1 与 XC（[1,0,0]）夹角为锐角
    if n_pi1[0] < 0:
        n_pi1 = -n_pi1

    n_pi1 /= np.linalg.norm(n_pi1)
    return n_pi1


def recover_full_pose_case1(solution, n_pi1):
    """
    Case 1: 直线 L̃ 平行于局部坐标轴 YB（即 nL̃ = nYB）。

    对应论文 Section II-C-1，式 (7)(8)。

    Parameters
    ----------
    solution : dict
        P1C 的一组解：{'O_B': ..., 'n_ZB': ...}
    n_pi1    : np.ndarray (3,)
        平面 π1 的单位法向量。

    Returns
    -------
    pose : dict
        完整位姿：
            'O_B'   圆心位置（相机坐标系）
            'n_XB'  局部 X 轴方向
            'n_YB'  局部 Y 轴方向（= 直线方向）
            'n_ZB'  圆法向量
            'R_BC'  旋转矩阵（从 {C} 到 {B}），3×3
    """
    O_B  = solution['O_B']
    n_ZB = solution['n_ZB']

    # 论文式 (7)：直线方向 = ZB × π1 的法向
    n_L  = np.cross(n_ZB, n_pi1)
    n_YB = n_L / np.linalg.norm(n_L)

    # 论文式 (8)：XB = YB × ZB
    n_XB = np.cross(n_YB, n_ZB)
    n_XB /= np.linalg.norm(n_XB)

    # 旋转矩阵 R_BC（{B}R{C}）：每行为 {B} 的轴在 {C} 中的坐标
    R_BC = np.row_stack([n_XB, n_YB, n_ZB])

    return {'O_B': O_B, 'n_XB': n_XB, 'n_YB': n_YB, 'n_ZB': n_ZB, 'R_BC': R_BC}


def recover_full_pose_case2(solution, n_pi1, n_L_body):
    """
    Case 2: 直线 L̃ 平行于圆所在平面 π2 但不平行于 YB 轴。

    对应论文 Section II-C-2，式 (12)(13)(14)(15)。

    分析：π1 与 π2 的交线 n_L_cam 有两个方向（±），绕 n_ZB 旋转 α
    后得到 n_YB，但旋转方向（±α）与 n_L_cam 方向存在四种组合。
    论文用 n_L_body 的分量符号来确定唯一组合，但该符号约定在不同
    位姿下可能出现歧义。
    本实现采用更稳健的策略：为每组 P1C 解生成两个 n_YB 候选
    （对应旋转角 +α 和 -α），最终由重投影验证统一选最优解。

    Parameters
    ----------
    solution   : dict
        P1C 的一组解：{'O_B': ..., 'n_ZB': ...}
    n_pi1      : np.ndarray (3,)
        平面 π1 的单位法向量。
    n_L_body   : np.ndarray (3,)
        直线在局部坐标系 {B} 中的方向向量（已知先验）。

    Returns
    -------
    poses : list of dict，长度为 2
        两个候选位姿（旋转角 +α 和 -α），由调用方通过重投影选最优。
    """
    O_B  = solution['O_B']
    n_ZB = solution['n_ZB']

    # 论文式 (13)：α 为直线与 YB 的夹角（在局部坐标系中）
    # nYB_body = [0,1,0]，所以 α = arccos(|n_L_body[1]|)
    alpha = np.arccos(np.clip(abs(n_L_body[1]), -1.0, 1.0))

    # 计算相机坐标系中直线方向 n_L（π1 ∩ π2 的交线方向）
    # 与 Case 1 式(7) 一致：n_L = n_ZB × n_π1（注意叉积顺序）
    n_L_cam = np.cross(n_ZB, n_pi1)
    if np.linalg.norm(n_L_cam) < 1e-9:
        raise ValueError("π1 与 π2 近似平行，无法确定直线方向。")
    n_L_cam /= np.linalg.norm(n_L_cam)

    # 生成两个候选：绕 n_ZB 分别旋转 +α 和 -α（论文式 14）
    # 正确的旋转角由重投影验证在上层筛选
    poses = []
    for rot_sign in [+1, -1]:
        n_YB = rodrigues_rotate(n_L_cam, n_ZB, rot_sign * alpha)
        n_YB /= np.linalg.norm(n_YB)
        n_XB = np.cross(n_YB, n_ZB)
        n_XB /= np.linalg.norm(n_XB)
        R_BC = np.vstack([n_XB, n_YB, n_ZB])
        poses.append({'O_B': O_B, 'n_XB': n_XB, 'n_YB': n_YB,
                      'n_ZB': n_ZB, 'R_BC': R_BC})
    return poses


def recover_full_pose_case3(solution, n_pi1, n_L_body):
    """
    Case 3: 直线 L̃ 不平行于圆所在平面 π2。

    对应论文 Section II-C-3，式 (16)(17)。

    直线不在 π2 内，故无法直接从图像中获取 n_L 在 {C} 中的方向。
    但可将 n_L_body 投影到 π2 的 XY 平面，再按同样思路旋转求 n_YB。
    与 Case 2 同理，旋转角 ±α 的符号由重投影验证在上层确定。

    Parameters
    ----------
    solution   : dict
        P1C 的一组解：{'O_B': ..., 'n_ZB': ...}
    n_pi1      : np.ndarray (3,)
        平面 π1 的单位法向量。
    n_L_body   : np.ndarray (3,)
        直线在局部坐标系 {B} 中的方向向量（已知先验）。

    Returns
    -------
    poses : list of dict，长度为 2
        两个候选位姿（旋转角 +α 和 -α）。
    """
    O_B  = solution['O_B']
    n_ZB = solution['n_ZB']

    # 将 n_L_body 投影到平面 O_B-XBYB（忽略 ZB 分量），论文式 (16)
    n_L_proj = np.array([n_L_body[0], n_L_body[1], 0.0])
    norm_proj = np.linalg.norm(n_L_proj)
    if norm_proj < 1e-9:
        raise ValueError("n_L_body 在 XY 平面的投影接近零向量。")
    n_L_proj /= norm_proj

    # α = 投影向量与 YB 轴的夹角（论文式 16），取绝对值后 arccos
    # 注意：n_L_proj[1] 可能为负（直线方向指向 -YB），应取绝对值
    # 旋转方向的歧义由重投影验证在上层解决（枚举 ±α）
    alpha = np.arccos(np.clip(abs(n_L_proj[1]), -1.0, 1.0))

    # n_L_cam：π2 平面内（⊥n_ZB）且由 π1 确定的方向（与 Case1/2 保持叉积顺序一致）
    n_L_cam = np.cross(n_ZB, n_pi1)
    if np.linalg.norm(n_L_cam) < 1e-9:
        n_L_cam = np.array([1.0, 0.0, 0.0])
    n_L_cam /= np.linalg.norm(n_L_cam)

    # 生成两个候选（±α），由重投影验证选最优（论文式 17）
    poses = []
    for rot_sign in [+1, -1]:
        n_YB = rodrigues_rotate(n_L_cam, n_ZB, rot_sign * alpha)
        n_YB /= np.linalg.norm(n_YB)
        n_XB = np.cross(n_YB, n_ZB)
        n_XB /= np.linalg.norm(n_XB)
        R_BC = np.vstack([n_XB, n_YB, n_ZB])
        poses.append({'O_B': O_B, 'n_XB': n_XB, 'n_YB': n_YB,
                      'n_ZB': n_ZB, 'R_BC': R_BC})
    return poses


def rodrigues_rotate(v, k, angle):
    """
    Rodrigues 旋转公式：将向量 v 绕单位轴 k 旋转 angle 弧度。

    Parameters
    ----------
    v     : np.ndarray (3,)  待旋转向量
    k     : np.ndarray (3,)  旋转轴（单位向量）
    angle : float            旋转角度（弧度）

    Returns
    -------
    v_rot : np.ndarray (3,)  旋转后的向量
    """
    k = k / np.linalg.norm(k)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    v_rot = v * cos_a + np.cross(k, v) * sin_a + k * (np.dot(k, v)) * (1 - cos_a)
    return v_rot


# =============================================================================
# 第三部分：重投影验证 —— 选出正确的位姿解
# =============================================================================

def reproject_point(P_body, pose, K):
    """
    将局部坐标系中的三维点 P_body 重投影到图像平面，
    对应论文式 (10)：λp̂ = K * P_C。

    Parameters
    ----------
    P_body : np.ndarray (3,)  点在 {B} 中的坐标
    pose   : dict             包含 'O_B'（相机坐标系）和 'R_BC'（旋转矩阵）
    K      : np.ndarray (3,3) 相机内参矩阵

    Returns
    -------
    p_img : np.ndarray (2,)   图像像素坐标 [u, v]
    """
    R_BC = pose['R_BC']   # {B}R{C}：将 {C} 中坐标变换到 {B}
    O_B  = pose['O_B']    # 圆心在 {C} 中的坐标

    # P 在相机坐标系中的坐标（论文式 9）
    # {B}T{C}: P_C = R_BC^T * (P_B - 0) + O_B
    # 注意：{B}T{C} 是从 {B} 到 {C} 的变换，故 P_C = R_BC^T * P_body + O_B
    P_cam = R_BC.T @ P_body + O_B   # (3,)

    # 投影到图像平面
    p_hom = K @ P_cam               # (3,)
    if abs(p_hom[2]) < 1e-9:
        raise ValueError("投影点在相机后方或位于无穷远处。")
    p_img = p_hom[:2] / p_hom[2]   # [u, v]
    return p_img


def point_to_line_distance(p, l_hat):
    """
    计算图像点 p 到图像直线 l_hat 的距离。

    对应论文式 (11)：l̂^T · p̂ = 0（理想情况）。

    Parameters
    ----------
    p     : np.ndarray (2,)  图像坐标 [u, v]
    l_hat : np.ndarray (3,)  直线齐次参数 [a, b, c]（au + bv + c = 0）

    Returns
    -------
    dist : float  点到直线的距离（像素）
    """
    p_hom = np.array([p[0], p[1], 1.0])
    val = abs(l_hat @ p_hom)
    denom = np.sqrt(l_hat[0] ** 2 + l_hat[1] ** 2) + 1e-12
    return val / denom


def select_correct_pose(poses, P_body_on_line, l_hat, K,
                        extra_verify_points=None):
    """
    通过重投影验证，从多组候选位姿中选出正确的解。

    对应论文 Section II-C-1-b 及 Section II-D。

    Parameters
    ----------
    poses              : list of dict
        候选位姿列表（来自 recover_full_pose_case*）。
    P_body_on_line     : np.ndarray (3,)
        直线上任意一点在局部坐标系 {B} 中的坐标。
    l_hat              : np.ndarray (3,)
        图像直线的齐次参数 [a, b, c]。
    K                  : np.ndarray (3, 3)
        相机内参矩阵。
    extra_verify_points : list of np.ndarray (3,) or None
        额外的验证点（{B} 坐标），用于提高验证可靠性（Case3 使用）。

    Returns
    -------
    correct_pose : dict
        重投影总距离最小的位姿（即正确解）。
    distances    : list of float
        每组候选解的主验证点重投影距离（像素）。
    """
    # 构建验证点列表
    verify_pts = [P_body_on_line]
    if extra_verify_points:
        verify_pts.extend(extra_verify_points)

    distances = []       # 主验证点距离（用于返回，保持接口兼容）
    total_distances = [] # 所有验证点总距离（用于选择最优）
    for pose in poses:
        try:
            # 主验证点
            p_img = reproject_point(verify_pts[0], pose, K)
            d0    = point_to_line_distance(p_img, l_hat)
            distances.append(d0)
            # 所有验证点总距离
            total = d0
            for vp in verify_pts[1:]:
                p_extra = reproject_point(vp, pose, K)
                total  += point_to_line_distance(p_extra, l_hat)
            total_distances.append(total)
        except Exception:
            distances.append(float('inf'))
            total_distances.append(float('inf'))

    best_idx = int(np.argmin(total_distances))
    return poses[best_idx], distances


# =============================================================================
# 第四部分：位姿优化（可选，需要已知点对应关系）
# =============================================================================

def optimize_pose(K, R_init, T_init, object_points, image_points):
    """
    以 PCL 封闭解为初值，用重投影误差优化最终位姿。

    对应论文式 (18)：argmin_{R,T} Σ||p_i - p̂(K,R,T,P_i)||^2

    Parameters
    ----------
    K             : np.ndarray (3,3)  相机内参
    R_init        : np.ndarray (3,3)  初始旋转矩阵（{C} → {B}）
    T_init        : np.ndarray (3,)   初始平移向量（圆心在 {C} 中）
    object_points : np.ndarray (N,3)  三维点（局部坐标系 {B}）
    image_points  : np.ndarray (N,2)  对应图像点（像素坐标）

    Returns
    -------
    rvec : np.ndarray (3,)   优化后的旋转向量（Rodrigues）
    tvec : np.ndarray (3,)   优化后的平移向量
    """
    # 将旋转矩阵转为旋转向量作为初始值
    rvec_init, _ = cv2.Rodrigues(R_init.T)   # R_CB = R_BC^T（从 {B} 到 {C}）
    tvec_init    = T_init.reshape(3, 1)

    obj_pts = object_points.astype(np.float32).reshape(-1, 1, 3)
    img_pts = image_points.astype(np.float32).reshape(-1, 1, 2)

    _, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K.astype(np.float32),
        None,                          # 假设已去畸变
        rvec_init.astype(np.float32),
        tvec_init.astype(np.float32),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    return rvec.flatten(), tvec.flatten()


# =============================================================================
# 第五部分：PCL 主接口
# =============================================================================

def pcl_pose_estimation(
    C_q,
    K,
    R_circle,
    l_hat,
    line_case,
    n_L_body=None,
    P_verify_body=None
):
    """
    PCL（Perspective Circle and Line）位姿估计主函数。

    Parameters
    ----------
    C_q           : np.ndarray (3,3)
        椭圆二次型矩阵（像素坐标系）。
    K             : np.ndarray (3,3)
        相机内参矩阵。
    R_circle      : float
        圆的已知半径（场景单位，如 mm）。
    l_hat         : np.ndarray (3,)
        图像直线的齐次参数 [a, b, c]（au+bv+c=0）。
    line_case     : int
        直线与圆平面的位置关系：
          1 → 直线平行于局部坐标轴 YB
          2 → 直线平行于 π2 但不平行于 YB
          3 → 直线不平行于 π2
    n_L_body      : np.ndarray (3,) or None
        直线在 {B} 中的方向向量（Case 2 和 Case 3 必须提供）。
    P_verify_body : np.ndarray (3,) or None
        直线上用于验证的点（{B} 坐标）；None 时使用 [0,1,0]^T 附近默认点。

    Returns
    -------
    correct_pose : dict
        正确的完整位姿，包含：
            'O_B'   : 圆心位置（相机坐标系），np.ndarray (3,)
            'n_XB'  : 局部 X 轴，np.ndarray (3,)
            'n_YB'  : 局部 Y 轴（滚转恢复），np.ndarray (3,)
            'n_ZB'  : 圆法向量，np.ndarray (3,)
            'R_BC'  : 旋转矩阵（{C}→{B}），np.ndarray (3,3)
            'euler' : Euler 角（ZYX，度）[ψ, θ, φ]
    distances    : list of float
        两候选解的重投影验证距离（像素）。
    dual_solutions : list of dict
        P1C 阶段的两组双义解（含 'O_B' 和 'n_ZB'）。
    """
    # ---- Step 1: 椭圆→圆的双义位姿 P1C ----
    dual_solutions = recover_dual_poses_p1c(C_q, K, R_circle)

    # ---- Step 2: 计算 π1 的法向量 ----
    n_pi1 = compute_plane_normal_from_line_image(l_hat, K)

    # ---- Step 3: 针对两组解分别恢复完整位姿 ----
    # Case 1 每个 P1C 解产生 1 个候选，Case 2/3 产生 2 个候选（±α）
    # 最终所有候选统一由重投影验证选最优。
    full_poses = []
    for sol in dual_solutions:
        if line_case == 1:
            pose = recover_full_pose_case1(sol, n_pi1)
            full_poses.append(pose)
        elif line_case == 2:
            if n_L_body is None:
                raise ValueError("Case 2 需要提供 n_L_body（直线在 {B} 中的方向）。")
            candidates = recover_full_pose_case2(sol, n_pi1, n_L_body)
            full_poses.extend(candidates)
        elif line_case == 3:
            if n_L_body is None:
                raise ValueError("Case 3 需要提供 n_L_body（直线在 {B} 中的方向）。")
            candidates = recover_full_pose_case3(sol, n_pi1, n_L_body)
            full_poses.extend(candidates)
        else:
            raise ValueError(f"line_case 必须为 1、2 或 3，当前值：{line_case}")

    # ---- Step 4: 重投影验证，选出正确解 ----
    if P_verify_body is None:
        P_verify_body = np.array([0.0, R_circle, 0.0])

    # Case 2/3 产生 4 个候选，用额外验证点提高区分度
    # Case 3 额外传入第二个验证点（直线另一端点）以增强约束
    extra_pts = None
    if line_case == 3 and n_L_body is not None:
        # 用与 P_verify_body 相对的方向取第二个验证点（估算）
        # 这里简单取 P_verify_body 沿 n_L_body 方向的对称点
        span = 2 * R_circle  # 约为直线可见长度
        extra_pt = P_verify_body - n_L_body * span
        extra_pts = [extra_pt]

    correct_pose, distances = select_correct_pose(
        full_poses, P_verify_body, l_hat, K,
        extra_verify_points=extra_pts
    )

    # ---- Step 5: 附加 Euler 角（ZYX 顺序，对应论文中的 ψ, θ, φ）----
    # R_BC 是从 {C} 到 {B} 的旋转，R_CB = R_BC^T 是从 {B} 到 {C}
    rot = Rotation.from_matrix(correct_pose['R_BC'].T)
    euler_zyx = rot.as_euler('zyx', degrees=True)   # [ψ, θ, φ]
    correct_pose['euler'] = euler_zyx

    return correct_pose, distances, dual_solutions


# =============================================================================
# 第六部分：辅助工具函数
# =============================================================================

def build_camera_matrix(fx, fy, cx, cy):
    """
    构造相机内参矩阵 K。

    Parameters
    ----------
    fx, fy : float  x、y 方向焦距（像素）
    cx, cy : float  主点坐标（像素）

    Returns
    -------
    K : np.ndarray (3,3)
    """
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)


def line_from_two_points(p1, p2):
    """
    由图像上两点计算直线的齐次参数 [a, b, c]（au+bv+c=0）。

    Parameters
    ----------
    p1, p2 : array-like (2,)  图像坐标 [u, v]

    Returns
    -------
    l_hat : np.ndarray (3,)，已归一化使 sqrt(a^2+b^2)=1
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    # 叉积得到齐次直线参数
    l = np.cross(np.append(p1, 1), np.append(p2, 1))
    # 归一化（方便计算点到直线距离）
    l /= np.sqrt(l[0] ** 2 + l[1] ** 2 + 1e-12)
    return l


def euler_zyx_to_rotation_matrix(psi_deg, theta_deg, phi_deg):
    """
    将 ZYX Euler 角（ψ, θ, φ，度）转换为旋转矩阵。

    Parameters
    ----------
    psi_deg   : float  绕 Z 轴旋转角（偏航，度）
    theta_deg : float  绕 Y 轴旋转角（俯仰，度）
    phi_deg   : float  绕 X 轴旋转角（滚转，度）

    Returns
    -------
    R : np.ndarray (3,3)
    """
    return Rotation.from_euler('zyx', [psi_deg, theta_deg, phi_deg],
                                degrees=True).as_matrix()


def print_pose_result(pose, distances):
    """格式化打印位姿估计结果。"""
    O  = pose['O_B']
    eu = pose['euler']
    print("=" * 55)
    print("  PCL 位姿估计结果")
    print("=" * 55)
    print(f"  圆心位置 (相机坐标系) [mm]:")
    print(f"    X = {O[0]:10.3f}")
    print(f"    Y = {O[1]:10.3f}")
    print(f"    Z = {O[2]:10.3f}")
    print(f"  Euler 角 (ZYX, 度):")
    print(f"    ψ (yaw)   = {eu[0]:8.3f}°")
    print(f"    θ (pitch) = {eu[1]:8.3f}°")
    print(f"    φ (roll)  = {eu[2]:8.3f}°")
    print(f"  重投影验证距离 (像素):")
    print(f"    解 1: {distances[0]:.4f}  解 2: {distances[1]:.4f}")
    print(f"  正确解为解 {'1' if distances[0] < distances[1] else '2'}")
    print("=" * 55)


# =============================================================================
# 第七部分：合成数据仿真示例（对应论文 Section III-A）
# =============================================================================

def run_synthetic_demo():
    """
    使用论文中的合成参数演示 PCL 方法（Case 1：直线平行于 YB 轴）。

    论文参数：
        f = 16.1 mm，像素尺寸 14μm，分辨率 1024×1024
        圆半径 R = 235 mm
        圆心位于 O_B = [Tz*tan(-10°), Tz*tan(-5°), Tz]，Tz = 5000 mm
        Euler 角：ψ=-30°, θ=0°, φ=-15°
        Case 1 直线端点：P0=[-384,600,214]^T，P1=[-384,-600,214]^T（{B} 坐标）
    """
    print("\n" + "=" * 55)
    print("  PCL 合成数据仿真（Case 1）")
    print("=" * 55)

    # ---- 相机内参 ----
    pixel_size = 14e-3   # mm/pixel（14μm）
    f_mm       = 16.1    # 焦距 mm
    fx = fy    = f_mm / pixel_size   # 像素焦距
    cx = cy    = 512.0               # 主点
    K = build_camera_matrix(fx, fy, cx, cy)
    print(f"  相机内参：fx=fy={fx:.1f} px，cx=cy={cx:.1f} px")

    # ---- 目标位姿（真值）----
    Tz = 5000.0   # mm
    Tx = Tz * np.tan(np.deg2rad(-10))
    Ty = Tz * np.tan(np.deg2rad(-5))
    O_B_gt = np.array([Tx, Ty, Tz])   # 圆心真值（相机坐标系）

    psi_gt, theta_gt, phi_gt = -30.0, 0.0, -15.0   # Euler 角真值（度）
    R_gt = euler_zyx_to_rotation_matrix(psi_gt, theta_gt, phi_gt)
    # R_gt 是从 {B} 到 {C} 的旋转，因此 R_BC = R_gt.T
    R_BC_gt = R_gt.T

    print(f"  真值圆心：X={Tx:.1f}, Y={Ty:.1f}, Z={Tz:.1f} mm")
    print(f"  真值 Euler：ψ={psi_gt}°, θ={theta_gt}°, φ={phi_gt}°")

    # ---- 圆的参数 ----
    R_circle = 235.0   # mm

    # ---- 生成投影椭圆（将圆上若干点投影后拟合）----
    n_pts = 360
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    # 圆上的点在 {B} 坐标系中
    circle_pts_B = np.column_stack([
        R_circle * np.cos(angles),
        R_circle * np.sin(angles),
        np.zeros(n_pts)
    ])
    # 转换到相机坐标系：P_C = R_gt * P_B + O_B
    circle_pts_C = (R_gt @ circle_pts_B.T).T + O_B_gt   # (N,3)
    # 投影到图像
    circle_pts_img = (K @ circle_pts_C.T).T
    circle_pts_img = circle_pts_img[:, :2] / circle_pts_img[:, 2:3]   # (N,2)

    # 用 OpenCV 拟合椭圆
    pts_for_fit = circle_pts_img.astype(np.float32).reshape(-1, 1, 2)
    ellipse_cv = cv2.fitEllipse(pts_for_fit)
    C_q = ellipse_to_matrix(ellipse_cv)
    (ecx, ecy), (eMA, ema), eangle = ellipse_cv
    print(f"\n  投影椭圆：中心=({ecx:.1f},{ecy:.1f})，"
          f"长轴={eMA:.1f}，短轴={ema:.1f}，角度={eangle:.1f}°")

    # ---- 生成直线投影（Case 1：L̃ ∥ YB）----
    P0_B = np.array([-384.0,  600.0, 214.0])
    P1_B = np.array([-384.0, -600.0, 214.0])

    # 转换到相机坐标系并投影
    def project_point(P_B, R, O, K_mat):
        P_C   = R @ P_B + O
        p_hom = K_mat @ P_C
        return p_hom[:2] / p_hom[2]

    p0_img = project_point(P0_B, R_gt, O_B_gt, K)
    p1_img = project_point(P1_B, R_gt, O_B_gt, K)
    l_hat  = line_from_two_points(p0_img, p1_img)
    print(f"  直线图像端点：({p0_img[0]:.1f},{p0_img[1]:.1f}) → "
          f"({p1_img[0]:.1f},{p1_img[1]:.1f})")

    # ---- 运行 PCL ----
    print("\n  运行 PCL 位姿估计...")
    correct_pose, distances, dual_sols = pcl_pose_estimation(
        C_q=C_q,
        K=K,
        R_circle=R_circle,
        l_hat=l_hat,
        line_case=1,
        P_verify_body=P0_B   # 用 P0 作为验证点
    )

    print_pose_result(correct_pose, distances)

    # ---- 计算误差 ----
    O_est  = correct_pose['O_B']
    eu_est = correct_pose['euler']
    pos_err = np.linalg.norm(O_est - O_B_gt)
    rel_err = pos_err / Tz * 100
    ang_err = np.array([eu_est[0]-psi_gt, eu_est[1]-theta_gt, eu_est[2]-phi_gt])

    print("\n  误差分析：")
    print(f"    位置绝对误差：{pos_err:.3f} mm")
    print(f"    位置相对误差：{rel_err:.4f} %")
    print(f"    Euler 角误差：Δψ={ang_err[0]:.4f}°, Δθ={ang_err[1]:.4f}°, Δφ={ang_err[2]:.4f}°")

    return correct_pose


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    run_synthetic_demo()
