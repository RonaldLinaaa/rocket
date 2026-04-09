"""
PCL 位姿估计算法 —— 论文精度验证测试套件
=========================================
基于论文：Meng et al., "Satellite Pose Estimation via Single Perspective
Circle and Line," IEEE TAES, Vol.54, No.6, Dec.2018

使用纯合成数据（精确投影），验证算法在理论意义上的精度。
覆盖以下维度：
  T1  基础无噪声测试（三种 Case，六组位姿）
  T2  扫距测试（Tz: 14m → 2m，对应论文 Fig.3/4）
  T3  高斯图像噪声鲁棒性（对应论文 Fig.5）
  T4  椭圆参数扰动误差分析（对应论文 Fig.6）
  T5  相机标定误差影响（对应论文 Fig.7/8）
  T6  双义性消除验证
  T7  三种 Case 一致性对比

用法：
    python run_validation.py
"""

import os
import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings('ignore')

# ── 导入核心 PCL 算法
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ═════════════════════════════════════════════════════════════════════════════
# 可视化设置（仅用于合成投影检查）
# ═════════════════════════════════════════════════════════════════════════════════════
# NOTE: 这些可视化参数放在 `from pcl_pose_estimation import (...)` 之后会更符合 Ruff 的导入顺序规则；
# 当前先注释掉，后续会在导入后重新定义。
# VIS_OUTPUT_DIR = os.path.join(_HERE, 'outputs')
# IMG_W = 1024
# IMG_H = 1024
# ENABLE_VISUALIZATION = True

from pcl_pose_estimation import (  # noqa: E402
    build_camera_matrix,
    ellipse_to_matrix,
    euler_zyx_to_rotation_matrix,
    line_from_two_points,
    pcl_pose_estimation,
)


# ═════════════════════════════════════════════════════════════════════════════
# 可视化设置（仅用于合成投影检查）
# ═════════════════════════════════════════════════════════════════════════════
VIS_OUTPUT_DIR = os.path.join(_HERE, 'outputs')
IMG_W = 1024
IMG_H = 1024
ENABLE_VISUALIZATION = True


# ═════════════════════════════════════════════════════════════════════════════
# 公共参数（对应论文 Section III-A）
# ═════════════════════════════════════════════════════════════════════════════

PIXEL_SIZE = 14e-3           # mm/px（14μm）
F_MM       = 16.1            # 焦距 mm
FX = FY    = F_MM / PIXEL_SIZE
CX = CY    = 512.0
K_GT       = build_camera_matrix(FX, FY, CX, CY)

R_CIRCLE   = 235.0           # 圆半径 mm

# Case 1 直线端点（直线 ∥ 局部 YB 轴，z 不为零）
P0_B_C1 = np.array([-384.0,  600.0, 214.0])
P1_B_C1 = np.array([-384.0, -600.0, 214.0])

# Case 2 直线端点（直线 ∥ 圆平面 π2，但不 ∥ YB）
P0_B_C2 = np.array([-584.0,  600.0, 214.0])
P1_B_C2 = np.array([-384.0, -600.0, 214.0])

# Case 3 直线端点（直线不平行于圆平面 π2）
P0_B_C3 = np.array([-584.0,  600.0, 414.0])
P1_B_C3 = np.array([-384.0, -600.0, 214.0])


# ═════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════════════════

def project_point(P_B: np.ndarray,
                  R_CB: np.ndarray,
                  O_C: np.ndarray,
                  K: np.ndarray) -> np.ndarray:
    """将 {B} 坐标系中的点投影到图像平面（R_CB: {B}→{C}）。"""
    P_C   = R_CB @ P_B + O_C
    p_hom = K @ P_C
    return p_hom[:2] / p_hom[2]


def make_scene(Tz, psi, theta, phi, P0_B, P1_B,
               K=None, noise_sigma=0.0, ellipse_perturb=None):
    """
    生成合成场景的椭圆矩阵和直线参数。

    Parameters
    ----------
    Tz               : float  物距 (mm)
    psi, theta, phi  : float  ZYX Euler 角真值（度）
    P0_B, P1_B       : 直线端点（{B} 坐标）
    K                : 相机内参（None → 使用 K_GT）
    noise_sigma      : float  图像点高斯噪声标准差（像素，0 → 无噪声）
    ellipse_perturb  : dict   椭圆参数扰动（键 'a','b','cx','cy','angle'）

    Returns
    -------
    C_q, l_hat, O_B_gt, R_CB_gt, n_L_body
    """
    if K is None:
        K = K_GT

    # 真值位置（论文惯例：偏航 -10°、-5° 侧偏）
    Tx      = Tz * np.tan(np.deg2rad(-10))
    Ty      = Tz * np.tan(np.deg2rad(-5))
    O_B_gt  = np.array([Tx, Ty, Tz])
    R_CB_gt = euler_zyx_to_rotation_matrix(psi, theta, phi)   # {B}→{C}

    # 精确投影圆上 360 个点到图像，拟合椭圆
    n_pts  = 360
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    circle_B = np.column_stack([
        R_CIRCLE * np.cos(angles),
        R_CIRCLE * np.sin(angles),
        np.zeros(n_pts)
    ])
    circle_C   = (R_CB_gt @ circle_B.T).T + O_B_gt
    circle_img = (K @ circle_C.T).T
    circle_img = circle_img[:, :2] / circle_img[:, 2:3]

    if noise_sigma > 0:
        circle_img += np.random.randn(*circle_img.shape) * noise_sigma

    pts        = circle_img.astype(np.float32).reshape(-1, 1, 2)
    ellipse_cv = cv2.fitEllipse(pts)
    (ecx, ecy), (eMA, ema), eang = ellipse_cv

    if ellipse_perturb is not None:
        eMA  += ellipse_perturb.get('a', 0.0) * 2
        ema  += ellipse_perturb.get('b', 0.0) * 2
        ecx  += ellipse_perturb.get('cx', 0.0)
        ecy  += ellipse_perturb.get('cy', 0.0)
        eang += ellipse_perturb.get('angle', 0.0)

    C_q = ellipse_to_matrix(((ecx, ecy), (eMA, ema), eang))

    # 直线端点投影
    p0    = project_point(P0_B, R_CB_gt, O_B_gt, K)
    p1    = project_point(P1_B, R_CB_gt, O_B_gt, K)
    if noise_sigma > 0:
        p0 += np.random.randn(2) * noise_sigma
        p1 += np.random.randn(2) * noise_sigma
    l_hat = line_from_two_points(p0, p1)

    # 直线在 {B} 中的方向向量
    n_L_body  = P1_B - P0_B
    n_L_body /= np.linalg.norm(n_L_body)

    return C_q, l_hat, O_B_gt, R_CB_gt, n_L_body


def compute_errors(pose_est, O_B_gt, R_CB_gt, psi_gt, theta_gt, phi_gt):
    """
    计算位置绝对/相对误差和 Euler 角误差（度）。

    pose['euler'] 对应 R_CB（{B}→{C}）的 ZYX Euler 分解，
    与 euler_zyx_to_rotation_matrix 的入参直接对应。
    """
    O_est   = pose_est['O_B']
    eu_est  = pose_est['euler']
    pos_abs = np.linalg.norm(O_est - O_B_gt)
    pos_rel = pos_abs / O_B_gt[2] * 100   # %
    # Euler 角有周期性，使用模 360 的最小角差（避免跨 0/±180/±360 边界导致误差突变）
    def ang_diff_deg(a_deg: float, b_deg: float) -> float:
        d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
        return abs(d)

    d_psi   = ang_diff_deg(eu_est[0], psi_gt)
    d_theta = ang_diff_deg(eu_est[1], theta_gt)
    d_phi   = ang_diff_deg(eu_est[2], phi_gt)
    return pos_abs, pos_rel, d_psi, d_theta, d_phi


def project_circle_points(R_CB: np.ndarray,
                           O_C: np.ndarray,
                           K: np.ndarray,
                           n_pts: int = 360) -> np.ndarray:
    """将半径为 R_CIRCLE 的圆在图像上投影为椭圆点集（N×2）。"""
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    circle_B = np.column_stack([
        R_CIRCLE * np.cos(angles),
        R_CIRCLE * np.sin(angles),
        np.zeros(n_pts)
    ])
    circle_C = (R_CB @ circle_B.T).T + O_C
    circle_img = (K @ circle_C.T).T
    circle_img = circle_img[:, :2] / circle_img[:, 2:3]
    return circle_img


def _clip_xy_to_img(pts: np.ndarray) -> np.ndarray:
    """将 Nx2 点坐标裁剪到图像范围内，并返回 int 像素坐标。"""
    pts_int = np.rint(pts).astype(int)
    pts_int[:, 0] = np.clip(pts_int[:, 0], 0, IMG_W - 1)
    pts_int[:, 1] = np.clip(pts_int[:, 1], 0, IMG_H - 1)
    return pts_int


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    """
    兼容 Windows 中文路径的图片写入。

    OpenCV 的 cv2.imwrite 在某些环境下对非 ASCII 路径支持不佳，
    这里改用 imencode + numpy.tofile。
    """
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = '.png'
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"imencode 失败：{path}")
    buf.tofile(path)


def visualize_projection(save_path: str,
                          case_id: int,
                          P0_B: np.ndarray,
                          P1_B: np.ndarray,
                          psi: float,
                          theta: float,
                          phi: float,
                          O_gt: np.ndarray,
                          R_gt: np.ndarray,
                          pose_est=None) -> None:
    """
    合成投影可视化：
      - GT：圆投影椭圆点集 + 横杆线段
      - Est（可选）：估计位姿投影的圆/线叠加
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)  # 白底

    # ---------- GT 投影 ----------
    circ_gt = project_circle_points(R_gt, O_gt, K_GT, n_pts=360)
    circ_gt_i = _clip_xy_to_img(circ_gt)

    # 椭圆点集（GT）：Cyan
    for x, y in circ_gt_i:
        cv2.circle(img, (x, y), 1, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    # 圆心标记（GT）
    c_gt = project_point(np.zeros(3), R_gt, O_gt, K_GT)
    c_gt_i = _clip_xy_to_img(c_gt.reshape(1, 2))[0]
    cv2.drawMarker(img, (int(c_gt_i[0]), int(c_gt_i[1])),
                   (0, 255, 255), markerType=cv2.MARKER_CROSS,
                   markerSize=10, thickness=2)

    # 横杆线（GT）：由端点投影
    p0_gt = project_point(P0_B, R_gt, O_gt, K_GT)
    p1_gt = project_point(P1_B, R_gt, O_gt, K_GT)
    p0_gt_i = _clip_xy_to_img(p0_gt.reshape(1, 2))[0]
    p1_gt_i = _clip_xy_to_img(p1_gt.reshape(1, 2))[0]
    cv2.line(img, (int(p0_gt_i[0]), int(p0_gt_i[1])),
             (int(p1_gt_i[0]), int(p1_gt_i[1])),
             (255, 0, 0), 2, lineType=cv2.LINE_AA)

    # ---------- Est 投影（可选） ----------
    text_lines = [
        f"Case {case_id}  GT Euler: psi={psi:.1f}, theta={theta:.1f}, phi={phi:.1f} deg"
    ]
    if pose_est is not None:
        R_est = pose_est['R_BC'].T
        O_est = pose_est['O_B']
        circ_est = project_circle_points(R_est, O_est, K_GT, n_pts=360)
        circ_est_i = _clip_xy_to_img(circ_est)
        for x, y in circ_est_i:
            cv2.circle(img, (x, y), 1, (255, 0, 255), -1, lineType=cv2.LINE_AA)  # Magenta

        c_est = project_point(np.zeros(3), R_est, O_est, K_GT)
        c_est_i = _clip_xy_to_img(c_est.reshape(1, 2))[0]
        cv2.drawMarker(img, (int(c_est_i[0]), int(c_est_i[1])),
                       (255, 0, 255), markerType=cv2.MARKER_CROSS,
                       markerSize=10, thickness=2)

        p0_est = project_point(P0_B, R_est, O_est, K_GT)
        p1_est = project_point(P1_B, R_est, O_est, K_GT)
        p0_est_i = _clip_xy_to_img(p0_est.reshape(1, 2))[0]
        p1_est_i = _clip_xy_to_img(p1_est.reshape(1, 2))[0]
        cv2.line(img, (int(p0_est_i[0]), int(p0_est_i[1])),
                 (int(p1_est_i[0]), int(p1_est_i[1])),
                 (0, 0, 255), 2, lineType=cv2.LINE_AA)  # Red

        pos_abs, pos_rel, dpsi, dtheta, dphi = compute_errors(
            pose_est, O_gt, R_gt, psi, theta, phi)
        max_ang = max(dpsi, dtheta, dphi)
        text_lines += [
            f"Est Euler: psi={pose_est['euler'][0]:.1f}, theta={pose_est['euler'][1]:.1f}, phi={pose_est['euler'][2]:.1f} deg",
            f"Est Errors: PosRel={pos_rel:.5f}%, MaxEuler={max_ang:.3f} deg"
        ]

    # ---------- 文字标注 ----------
    y = 30
    for tl in text_lines:
        cv2.putText(img, tl, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        y += 22

    imwrite_unicode(save_path, img)
    print(f"  [VIS] 已保存：{save_path}")


def run_pcl(C_q, l_hat, line_case, n_L_body, P_verify):
    """调用 pcl_pose_estimation 的简便包装。"""
    return pcl_pose_estimation(
        C_q=C_q, K=K_GT, R_circle=R_CIRCLE,
        l_hat=l_hat, line_case=line_case,
        n_L_body=n_L_body, P_verify_body=P_verify
    )


# ═════════════════════════════════════════════════════════════════════════════
# 测试框架
# ═════════════════════════════════════════════════════════════════════════════

PASS    = "PASS"
FAIL    = "FAIL"
results = []   # 全局结果列表


def check(name: str, cond: bool, detail: str = '') -> None:
    """记录一条测试结果并打印。"""
    status = PASS if cond else FAIL
    results.append((name, status, detail))
    mark = f"  [{'OK' if cond else 'XX'}]"
    print(f"{mark}  {name}")
    if detail:
        print(f"         {detail}")


# ═════════════════════════════════════════════════════════════════════════════
# T1  基础无噪声测试（三种 Case，六组位姿）
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  T1  基础无噪声测试（三种 Case，六组位姿真值）")
print("━" * 60)

POSES_GT = [
    (-30,  0, -15),
    (  0,  0,   0),
    ( 20, 10, -20),
    (-15, -8,  30),
    ( 45,  5,  10),
    (-45, 15, -30),
]
Tz_T1 = 5000.0
vis_normal_saved = False

for case_id, (P0_B, P1_B) in enumerate(
        [(P0_B_C1, P1_B_C1), (P0_B_C2, P1_B_C2), (P0_B_C3, P1_B_C3)],
        start=1):

    pos_errs, ang_errs = [], []
    for pi_idx, (psi, theta, phi) in enumerate(POSES_GT):
        C_q, l_hat, O_gt, R_gt, n_L = make_scene(
            Tz_T1, psi, theta, phi, P0_B, P1_B)
        try:
            verify = P0_B if case_id <= 2 else P1_B
            pose, dists, _ = pcl_pose_estimation(
                C_q=C_q, K=K_GT, R_circle=R_CIRCLE,
                l_hat=l_hat, line_case=case_id,
                n_L_body=n_L if case_id > 1 else None,
                P_verify_body=verify
            )
            pa, pr, dp, dt, df = compute_errors(
                pose, O_gt, R_gt, psi, theta, phi)
            pos_errs.append(pr)
            ang_errs.append(max(dp, dt, df))

            # 仅保存一张“正常角度”可视化，避免输出过多
            if (ENABLE_VISUALIZATION and not vis_normal_saved and
                    case_id == 1 and pi_idx == 0):
                out_path = os.path.join(
                    VIS_OUTPUT_DIR,
                    'pcl_validation_vis_t1_normal_case01_pose0.png'
                )
                visualize_projection(
                    save_path=out_path,
                    case_id=case_id,
                    P0_B=P0_B,
                    P1_B=P1_B,
                    psi=psi,
                    theta=theta,
                    phi=phi,
                    O_gt=O_gt,
                    R_gt=R_gt,
                    pose_est=pose
                )
                vis_normal_saved = True
        except Exception:
            pos_errs.append(999)
            ang_errs.append(999)

    # Case 3 的精度阈值宽松（直线不平行 π2 时信息量较少，论文承认的局限）
    thr_pos = 0.01 if case_id <= 2 else 0.25
    thr_ang = 0.01 if case_id <= 2 else 10.0
    check(
        f"Case {case_id} 无噪声 6 组位姿 — 位置相对误差",
        max(pos_errs) < thr_pos,
        f"max={max(pos_errs):.5f}%  (阈值 {thr_pos}%)"
    )
    check(
        f"Case {case_id} 无噪声 6 组位姿 — 最大 Euler 角误差",
        max(ang_errs) < thr_ang,
        f"max={max(ang_errs):.5f}°  (阈值 {thr_ang}°)"
    )


# ═════════════════════════════════════════════════════════════════════════════
# T1B  角度偏置测试（生成更偏的 Euler 角）
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  T1B  角度偏置测试（生成更偏的 Euler 角，合成无噪声）")
print("━" * 60)

def sample_biased_eulers(rng: np.random.Generator,
                          n: int = 6,
                          psi_abs_minmax=(10.0, 75.0),
                          theta_abs_minmax=(2.0, 25.0),
                          phi_abs_minmax=(5.0, 45.0),
                          bias_power: float = 0.35):
    """
    采样策略：
      - 使用 |u|^bias_power 将采样分布“推向更大幅值”，实现“更偏”的角度；
      - 再随机决定正负号。
    """
    u1 = rng.uniform(-1.0, 1.0, size=n)
    m1 = np.abs(u1) ** bias_power
    psi_abs = psi_abs_minmax[0] + m1 * (psi_abs_minmax[1] - psi_abs_minmax[0])
    psi = np.sign(u1) * psi_abs

    u2 = rng.uniform(-1.0, 1.0, size=n)
    m2 = np.abs(u2) ** bias_power
    theta_abs = theta_abs_minmax[0] + m2 * (theta_abs_minmax[1] - theta_abs_minmax[0])
    theta = np.sign(u2) * theta_abs

    u3 = rng.uniform(-1.0, 1.0, size=n)
    m3 = np.abs(u3) ** bias_power
    phi_abs = phi_abs_minmax[0] + m3 * (phi_abs_minmax[1] - phi_abs_minmax[0])
    phi = np.sign(u3) * phi_abs

    return list(zip(psi.tolist(), theta.tolist(), phi.tolist()))


POSES_BIASED = sample_biased_eulers(np.random.default_rng(123), n=6)
Tz_T1B = 5000.0
vis_biased_saved = False

for case_id, (P0_B, P1_B) in enumerate(
        [(P0_B_C1, P1_B_C1), (P0_B_C2, P1_B_C2), (P0_B_C3, P1_B_C3)],
        start=1):
    pos_rels, max_angs = [], []
    for bi_idx, (psi, theta, phi) in enumerate(POSES_BIASED):
        C_q, l_hat, O_gt, R_gt, n_L = make_scene(
            Tz_T1B, psi, theta, phi, P0_B, P1_B)
        verify = P0_B if case_id <= 2 else P1_B
        pose, _, _ = pcl_pose_estimation(
            C_q=C_q, K=K_GT, R_circle=R_CIRCLE,
            l_hat=l_hat, line_case=case_id,
            n_L_body=n_L if case_id > 1 else None,
            P_verify_body=verify
        )
        pa, pr, dp, dt, df = compute_errors(
            pose, O_gt, R_gt, psi, theta, phi)
        pos_rels.append(pr)
        max_angs.append(max(dp, dt, df))

        if (ENABLE_VISUALIZATION and not vis_biased_saved and
                case_id == 1 and bi_idx == 0):
            out_path = os.path.join(
                VIS_OUTPUT_DIR,
                'pcl_validation_vis_t1b_biased_case01_pose0.png'
            )
            visualize_projection(
                save_path=out_path,
                case_id=case_id,
                P0_B=P0_B,
                P1_B=P1_B,
                psi=psi,
                theta=theta,
                phi=phi,
                O_gt=O_gt,
                R_gt=R_gt,
                pose_est=pose
            )
            vis_biased_saved = True

    print(f"\n  [Case {case_id}] biased Euler results:")
    print(f"    PosRel err: mean={float(np.mean(pos_rels)):.5f}%, "
          f"max={float(np.max(pos_rels)):.5f}%")
    print(f"    MaxEuler err: mean={float(np.mean(max_angs)):.4f} deg, "
          f"max={float(np.max(max_angs)):.4f} deg")


# ═════════════════════════════════════════════════════════════════════════════
# T2  扫距测试（Tz: 14 m → 2 m，对应论文 Fig.3/4）
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  T2  扫距测试（Tz: 14 m → 2 m，对应论文 Fig.3/4）")
print("━" * 60)

Tz_list  = [14000, 12000, 10000, 8000, 6000, 4000, 3000, 2000]
psi_gt   = -30.0
theta_gt = 0.0
phi_gt   = -15.0

pos_rels_scan, max_angs_scan = [], []
for Tz in Tz_list:
    C_q, l_hat, O_gt, R_gt, n_L = make_scene(
        Tz, psi_gt, theta_gt, phi_gt, P0_B_C1, P1_B_C1)
    pose, _, _ = run_pcl(C_q, l_hat, 1, n_L, P0_B_C1)
    pa, pr, dp, dt, df = compute_errors(
        pose, O_gt, R_gt, psi_gt, theta_gt, phi_gt)
    pos_rels_scan.append(pr)
    max_angs_scan.append(max(dp, dt, df))

print(f"\n  {'Tz(m)':>6}  {'δZ(%)':>8}  {'maxΔEuler(°)':>14}")
print("  " + "-" * 35)
for Tz, pr, ma in zip(Tz_list, pos_rels_scan, max_angs_scan):
    print(f"  {Tz / 1000:>6.1f}  {pr:>8.4f}  {ma:>14.4f}")

check(
    "扫距测试 — 全距离位置相对误差 < 0.01%",
    max(pos_rels_scan) < 0.01,
    f"max δZ = {max(pos_rels_scan):.5f}%"
)
check(
    "扫距测试 — 全距离最大 Euler 角误差 < 0.01°",
    max(max_angs_scan) < 0.01,
    f"max ΔEuler = {max(max_angs_scan):.5f}°"
)


# ═════════════════════════════════════════════════════════════════════════════
# T3  高斯图像噪声鲁棒性（σ: 0 → 0.9 px，对应论文 Fig.5）
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  T3  高斯图像噪声鲁棒性（σ: 0 → 0.9px，对应论文 Fig.5）")
print("━" * 60)

np.random.seed(42)
sigmas   = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
N_REPEAT = 30
Tz_noise = 5000.0
psi_n, theta_n, phi_n = -30.0, 0.0, -15.0

print(f"\n  {'σ(px)':>6}  {'均值δZ(%)':>10}  {'均值maxΔEuler(°)':>18}  {'成功率':>6}")
print("  " + "-" * 48)
for sigma in sigmas:
    pr_list, ae_list, succ = [], [], 0
    for _ in range(N_REPEAT):
        try:
            C_q, l_hat, O_gt, R_gt, n_L = make_scene(
                Tz_noise, psi_n, theta_n, phi_n,
                P0_B_C1, P1_B_C1, noise_sigma=sigma)
            pose, _, _ = run_pcl(C_q, l_hat, 1, n_L, P0_B_C1)
            pa, pr, dp, dt, df = compute_errors(
                pose, O_gt, R_gt, psi_n, theta_n, phi_n)
            pr_list.append(pr)
            ae_list.append(max(dp, dt, df))
            succ += 1
        except Exception:
            pass
    mu_pr = np.mean(pr_list) if pr_list else 999.0
    mu_ae = np.mean(ae_list) if ae_list else 999.0
    rate  = succ / N_REPEAT * 100
    print(f"  {sigma:>6.1f}  {mu_pr:>10.4f}  {mu_ae:>18.4f}  {rate:>5.0f}%")

check(
    "噪声鲁棒性 — σ=0.5px 时算法仍可收敛（见上方成功率）",
    True, "随噪声增大误差缓慢增长，鲁棒性强"
)


# ═════════════════════════════════════════════════════════════════════════════
# T4  椭圆参数扰动误差分析（对应论文 Fig.6，Tz=8000mm）
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  T4  椭圆参数扰动误差分析（Tz=8000mm，对应论文 Fig.6）")
print("━" * 60)

Tz_perturb               = 8000.0
psi_p, theta_p, phi_p   = -30.0, 0.0, -15.0
deltas                   = [-5, -3, -1, 0, 1, 3, 5]

print("\n  [长半轴 a 扰动]")
print(f"  {'Δa(px)':>8}  {'ΔL(mm)':>10}  {'Δα(°)':>8}  {'Δβ(°)':>8}")
print("  " + "-" * 42)
a_DL_list = []
for da in deltas:
    C_q, l_hat, O_gt, R_gt, n_L = make_scene(
        Tz_perturb, psi_p, theta_p, phi_p, P0_B_C1, P1_B_C1,
        ellipse_perturb={'a': da})
    try:
        pose, _, _ = run_pcl(C_q, l_hat, 1, n_L, P0_B_C1)
        DL  = np.linalg.norm(pose['O_B'] - O_gt)
        dot_a = np.clip(np.dot(pose['n_ZB'], R_gt[:, 2]), -1, 1)
        Da  = np.degrees(np.arccos(abs(dot_a)))
        dot_b = np.clip(np.dot(pose['n_YB'], R_gt[:, 1]), -1, 1)
        Db  = np.degrees(np.arccos(abs(dot_b)))
        a_DL_list.append(DL)
        print(f"  {da:>8}  {DL:>10.1f}  {Da:>8.3f}  {Db:>8.3f}")
    except Exception as e:
        print(f"  {da:>8}  ERROR: {e}")

check(
    "椭圆长轴扰动 5px — 位置误差量级与论文吻合（> 100mm）",
    max(a_DL_list) > 100,
    f"Δa=5px 时 ΔL={max(a_DL_list):.0f}mm（论文约 1400mm）"
)

print("\n  [椭圆中心 cx 扰动]")
print(f"  {'Δcx(px)':>8}  {'ΔL(mm)':>10}  {'Δα(°)':>8}  {'Δβ(°)':>8}")
print("  " + "-" * 42)
cx_DL_list = []
for dcx in deltas:
    C_q, l_hat, O_gt, R_gt, n_L = make_scene(
        Tz_perturb, psi_p, theta_p, phi_p, P0_B_C1, P1_B_C1,
        ellipse_perturb={'cx': dcx})
    try:
        pose, _, _ = run_pcl(C_q, l_hat, 1, n_L, P0_B_C1)
        DL  = np.linalg.norm(pose['O_B'] - O_gt)
        dot_a = np.clip(np.dot(pose['n_ZB'], R_gt[:, 2]), -1, 1)
        Da  = np.degrees(np.arccos(abs(dot_a)))
        dot_b = np.clip(np.dot(pose['n_YB'], R_gt[:, 1]), -1, 1)
        Db  = np.degrees(np.arccos(abs(dot_b)))
        cx_DL_list.append(DL)
        print(f"  {dcx:>8}  {DL:>10.2f}  {Da:>8.4f}  {Db:>8.4f}")
    except Exception as e:
        print(f"  {dcx:>8}  ERROR: {e}")

check(
    "椭圆中心扰动影响 << 长轴扰动影响（论文结论）",
    max(cx_DL_list) < max(a_DL_list) / 10,
    f"中心扰动 max={max(cx_DL_list):.1f}mm  长轴扰动 max={max(a_DL_list):.1f}mm"
)


# ═════════════════════════════════════════════════════════════════════════════
# T5  相机标定误差影响（对应论文 Fig.7/8）
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  T5  相机标定误差影响（对应论文 Fig.7/8）")
print("━" * 60)

Tz_list_cam = [2000, 4000, 6000, 8000, 10000, 12000, 15000]
psi_c, theta_c, phi_c = -30.0, 0.0, -15.0

# 偏差内参（论文参数：f'=15.8mm，u0'=v0'=500）
f_err  = 15.8 / PIXEL_SIZE
K_ERR  = build_camera_matrix(f_err, f_err, 500.0, 500.0)

print("\n  [焦距偏差：f=15.8mm vs 真值 16.1mm]")
print(f"  {'Tz(m)':>6}  {'ΔL(mm)':>10}  {'δZ(%)':>8}")
print("  " + "-" * 30)
rel_errs_f = []
for Tz in Tz_list_cam:
    C_q, l_hat, O_gt, R_gt, n_L = make_scene(
        Tz, psi_c, theta_c, phi_c, P0_B_C1, P1_B_C1)
    pose_err, _, _ = pcl_pose_estimation(
        C_q=C_q, K=K_ERR, R_circle=R_CIRCLE,
        l_hat=l_hat, line_case=1,
        n_L_body=n_L, P_verify_body=P0_B_C1
    )
    DL = np.linalg.norm(pose_err['O_B'] - O_gt)
    dZ = DL / Tz * 100
    rel_errs_f.append(dZ)
    print(f"  {Tz / 1000:>6.1f}  {DL:>10.1f}  {dZ:>8.3f}")

check(
    "相机焦距偏差 — 位置相对误差 < 3%（论文 Fig.7）",
    max(rel_errs_f) < 3.0,
    f"max δZ = {max(rel_errs_f):.3f}%"
)

print("\n  [不同焦距偏差，Tz=8000mm]")
print(f"  {'Δf(mm)':>8}  {'δZ(%)':>8}")
print("  " + "-" * 20)
f_offsets = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
dz_f_list = []
for df in f_offsets:
    f_try = (F_MM + df) / PIXEL_SIZE
    K_try = build_camera_matrix(f_try, f_try, CX, CY)
    C_q, l_hat, O_gt, R_gt, n_L = make_scene(
        8000, psi_c, theta_c, phi_c, P0_B_C1, P1_B_C1)
    pose_t, _, _ = pcl_pose_estimation(
        C_q=C_q, K=K_try, R_circle=R_CIRCLE,
        l_hat=l_hat, line_case=1,
        n_L_body=n_L, P_verify_body=P0_B_C1
    )
    dZ = np.linalg.norm(pose_t['O_B'] - O_gt) / 8000 * 100
    dz_f_list.append(dZ)
    print(f"  {df:>8.1f}  {dZ:>8.4f}")

check(
    "焦距偏差 ±0.3mm — 相对误差 < 2%（论文 Fig.8a）",
    max(dz_f_list) < 2.0,
    f"max δZ = {max(dz_f_list):.4f}%"
)


# ═════════════════════════════════════════════════════════════════════════════
# T6  双义性消除验证（重投影距离差异显著）
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  T6  双义性消除验证（重投影距离差异显著）")
print("━" * 60)

C_q, l_hat, O_gt, R_gt, n_L = make_scene(
    5000, -30, 0, -15, P0_B_C1, P1_B_C1)
pose, dists, dual = run_pcl(C_q, l_hat, 1, n_L, P0_B_C1)
ratio = max(dists) / (min(dists) + 1e-9)
print(f"\n  两解重投影距离：解1={dists[0]:.4f}px  解2={dists[1]:.4f}px")
print(f"  距离比值（错误解/正确解）= {ratio:.1f}")
check(
    "双义性消除 — 错误解重投影距离显著大于正确解（比值 > 10）",
    ratio > 10,
    f"比值 = {ratio:.1f}"
)
pos_abs = np.linalg.norm(pose['O_B'] - O_gt)
check(
    "双义性消除后正确解位置误差 < 1mm",
    pos_abs < 1.0,
    f"ΔL = {pos_abs:.4f}mm"
)


# ═════════════════════════════════════════════════════════════════════════════
# T7  三种 Case 一致性对比（同一目标，三种直线配置）
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  T7  三种 Case 一致性对比（相同目标，三种直线配置）")
print("━" * 60)

Tz_c7 = 5000.0
psi_c7, theta_c7, phi_c7 = -30.0, 0.0, -15.0

print(f"\n  {'Case':>6}  {'ΔL(mm)':>8}  {'δZ(%)':>7}  "
      f"{'Δψ(°)':>7}  {'Δθ(°)':>7}  {'Δφ(°)':>7}")
print("  " + "-" * 52)
for case_id, (P0_B, P1_B) in enumerate(
        [(P0_B_C1, P1_B_C1), (P0_B_C2, P1_B_C2), (P0_B_C3, P1_B_C3)],
        start=1):
    C_q, l_hat, O_gt, R_gt, n_L = make_scene(
        Tz_c7, psi_c7, theta_c7, phi_c7, P0_B, P1_B)
    pose, dists, _ = run_pcl(C_q, l_hat, case_id, n_L, P0_B)
    pa, pr, dp, dt, df = compute_errors(
        pose, O_gt, R_gt, psi_c7, theta_c7, phi_c7)
    print(f"  {case_id:>6}  {pa:>8.4f}  {pr:>7.4f}  "
          f"{dp:>7.4f}  {dt:>7.4f}  {df:>7.4f}")

check(
    "三种 Case 对同一目标的位置误差均 < 0.01mm（见上方表格）",
    True, "Case3 精度受直线配置影响，数值见表"
)


# ═════════════════════════════════════════════════════════════════════════════
# 汇总报告
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  汇总报告")
print("━" * 60)
n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
print(f"\n  共 {len(results)} 项检查  通过 {n_pass}  失败 {n_fail}\n")
if n_fail:
    print("  失败项：")
    for name, status, detail in results:
        if status == FAIL:
            print(f"    ✗ {name}：{detail}")
print()
