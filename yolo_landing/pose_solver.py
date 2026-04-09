"""
PnP 位姿解算器

基于检测到的 2D 关键点和已知的 3D 模型坐标,
使用 EPnP / IPPE 算法解算相机相对于着陆标志的 6-DoF 位姿。

数学模型:
  s * [u, v, 1]^T = K * [R | t] * [X, Y, Z, 1]^T

  其中:
    K: 相机内参矩阵 (已知)
    [R | t]: 着陆标志在相机坐标系下的位姿 (待求)
    (X, Y, Z): 关键点 3D 坐标 (来自 kp.json, 所有 Z=0)
    (u, v): 检测到的 2D 关键点像素坐标
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict


# 相机内参 (与 Blender 仿真一致)
# fx = fy = lens_mm / sensor_width_mm * img_w = 24/36*1080 = 720
CAMERA_MATRIX = np.array([
    [720.0,   0.0, 540.0],
    [  0.0, 720.0, 360.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

DIST_COEFFS = np.zeros(5, dtype=np.float64)  # 仿真数据无畸变

# 3D 关键点坐标 (kp.json, 单位: 与 Blender 场景一致)
WORLD_KEYPOINTS_3D = np.array([
    [ 0.0,   0.0,  0.0],   # kp0: 中心
    [21.5,   0.0,  0.0],   # kp1
    [ 0.0,  21.5,  0.0],   # kp2
    [-21.5,  0.0,  0.0],   # kp3
    [ 0.0, -21.5,  0.0],   # kp4
    [-3.9,   9.0,  0.0],   # kp5
    [-3.9,  -9.0,  0.0],   # kp6
    [ 3.9,   9.0,  0.0],   # kp7
    [ 3.9,  -9.0,  0.0],   # kp8
], dtype=np.float64)


class LandingPoseSolver:
    """着陆标志 PnP 位姿解算器"""

    def __init__(
        self,
        camera_matrix: np.ndarray = CAMERA_MATRIX,
        dist_coeffs: np.ndarray = DIST_COEFFS,
        world_points_3d: np.ndarray = WORLD_KEYPOINTS_3D,
        min_points: int = 4,
        confidence_threshold: float = 0.5,
    ):
        self.K = camera_matrix
        self.dist = dist_coeffs
        self.pts_3d = world_points_3d
        self.min_points = min_points
        self.conf_thresh = confidence_threshold

    def solve(
        self,
        keypoints_2d: np.ndarray,
        confidences: np.ndarray,
        method: str = "epnp",
    ) -> Optional[Dict]:
        """
        从 2D 关键点解算位姿

        Args:
            keypoints_2d: [K, 2] 像素坐标
            confidences:  [K] 每个关键点的置信度
            method: 'epnp' | 'ippe' | 'iterative' | 'ransac'

        Returns:
            dict with keys:
              rvec:     旋转向量 [3]
              tvec:     平移向量 [3]
              R:        旋转矩阵 [3, 3]
              reproj_error: 重投影误差 (像素)
              inlier_count: 使用的关键点数量
            or None if solve fails
        """
        # 筛选置信度足够的关键点
        mask = confidences >= self.conf_thresh
        if mask.sum() < self.min_points:
            return None

        pts_2d = keypoints_2d[mask].astype(np.float64)
        pts_3d = self.pts_3d[mask].astype(np.float64)

        # 选择 PnP 方法
        pnp_methods = {
            "epnp": cv2.SOLVEPNP_EPNP,
            "ippe": cv2.SOLVEPNP_IPPE,         # 专为共面点优化
            "iterative": cv2.SOLVEPNP_ITERATIVE,
            "sqpnp": cv2.SOLVEPNP_SQPNP,
        }

        if method == "ransac":
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, self.K, self.dist,
                iterationsCount=200,
                reprojectionError=5.0,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not success:
                return None
            n_inliers = len(inliers) if inliers is not None else 0
        else:
            flag = pnp_methods.get(method, cv2.SOLVEPNP_EPNP)
            success, rvec, tvec = cv2.solvePnP(
                pts_3d, pts_2d, self.K, self.dist, flags=flag)
            if not success:
                return None
            n_inliers = len(pts_2d)

        # 用 LM 优化精炼
        rvec, tvec = cv2.solvePnPRefineLM(
            pts_3d, pts_2d, self.K, self.dist, rvec, tvec)

        # 镜像歧义消解：着陆标志在相机正前方，tvec[2] 必须 > 0
        # 如果 PnP 解出 tvec[2] < 0，尝试 IPPE 返回两个解取正确的
        if tvec.flatten()[2] < 0:
            rvec, tvec = self._resolve_mirror(pts_3d, pts_2d, rvec, tvec)

        R, _ = cv2.Rodrigues(rvec)

        reproj_pts, _ = cv2.projectPoints(
            pts_3d, rvec, tvec, self.K, self.dist)
        reproj_pts = reproj_pts.reshape(-1, 2)
        reproj_error = np.linalg.norm(reproj_pts - pts_2d, axis=1).mean()

        return {
            "rvec": rvec.flatten(),
            "tvec": tvec.flatten(),
            "R": R,
            "reproj_error": reproj_error,
            "inlier_count": n_inliers,
        }

    def _resolve_mirror(
        self,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        镜像歧义消解：共面点 PnP 可能返回相机在平面"背面"的解。
        使用 IPPE 获取两个候选解，选择 tvec[2]>0（标志在相机前方）的那个。
        如果都不满足，翻转当前解。
        """
        try:
            n_sol, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
                pts_3d, pts_2d, self.K, self.dist,
                flags=cv2.SOLVEPNP_IPPE)
            best_rvec, best_tvec = rvec, tvec
            best_err = float("inf")
            for i in range(n_sol):
                rv, tv = rvecs[i], tvecs[i]
                rv, tv = cv2.solvePnPRefineLM(
                    pts_3d, pts_2d, self.K, self.dist, rv, tv)
                if tv.flatten()[2] > 0:
                    rp, _ = cv2.projectPoints(pts_3d, rv, tv, self.K, self.dist)
                    err = np.linalg.norm(rp.reshape(-1, 2) - pts_2d, axis=1).mean()
                    if err < best_err:
                        best_err = err
                        best_rvec, best_tvec = rv, tv
            if best_tvec.flatten()[2] > 0:
                return best_rvec, best_tvec
        except Exception:
            pass

        # 最后兜底：对 tvec 取反并翻转旋转 180°
        R, _ = cv2.Rodrigues(rvec)
        R_flip = np.diag([1.0, -1.0, -1.0]) @ R
        tvec_flip = np.diag([1.0, -1.0, -1.0]) @ tvec.reshape(3, 1)
        rvec_flip, _ = cv2.Rodrigues(R_flip)
        return rvec_flip, tvec_flip

    def get_camera_pose_in_world(self, result: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 solvePnP 结果转换为相机在世界坐标系中的位姿

        solvePnP 输出: 世界->相机变换 (R_cw, t_cw)
        相机在世界中的位置: t_wc = -R_cw^T @ t_cw
        相机在世界中的朝向: R_wc = R_cw^T
        """
        R_cw = result["R"]
        t_cw = result["tvec"]
        R_wc = R_cw.T
        t_wc = -R_cw.T @ t_cw
        return R_wc, t_wc

    def compute_euler_angles(self, R: np.ndarray) -> np.ndarray:
        """旋转矩阵 -> 欧拉角 (度), 按 XYZ 顺序"""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.degrees(np.array([x, y, z]))

    def visualize(
        self,
        image: np.ndarray,
        keypoints_2d: np.ndarray,
        confidences: np.ndarray,
        result: Optional[Dict],
    ) -> np.ndarray:
        """在图像上绘制关键点和位姿信息"""
        vis = image.copy()

        # 绘制关键点
        for i, (pt, conf) in enumerate(zip(keypoints_2d, confidences)):
            x, y = int(pt[0]), int(pt[1])
            if conf >= self.conf_thresh:
                color = (0, 255, 0)   # 可用: 绿色
            else:
                color = (128, 128, 128)  # 低置信: 灰色
            cv2.circle(vis, (x, y), 4, color, -1)
            cv2.putText(vis, f"kp{i}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # 绘制位姿信息
        if result is not None:
            R_wc, t_wc = self.get_camera_pose_in_world(result)
            euler = self.compute_euler_angles(R_wc)
            info_lines = [
                f"Position: ({t_wc[0]:.1f}, {t_wc[1]:.1f}, {t_wc[2]:.1f})",
                f"Euler XYZ: ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg",
                f"Reproj err: {result['reproj_error']:.2f} px",
                f"Inliers: {result['inlier_count']}",
            ]
            y0 = 25
            for line in info_lines:
                cv2.putText(vis, line, (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y0 += 22

            # 绘制重投影点 (红色 x)
            reproj_pts, _ = cv2.projectPoints(
                self.pts_3d, result["rvec"], result["tvec"],
                self.K, self.dist)
            for pt in reproj_pts.reshape(-1, 2):
                x, y = int(pt[0]), int(pt[1])
                cv2.drawMarker(vis, (x, y), (0, 0, 255),
                               cv2.MARKER_CROSS, 6, 1)

            # 绘制坐标轴
            axis_pts = np.float64([
                [0, 0, 0], [15, 0, 0], [0, 15, 0], [0, 0, -15]
            ])
            axis_2d, _ = cv2.projectPoints(
                axis_pts, result["rvec"], result["tvec"],
                self.K, self.dist)
            axis_2d = axis_2d.reshape(-1, 2).astype(int)
            origin = tuple(axis_2d[0])
            cv2.arrowedLine(vis, origin, tuple(axis_2d[1]), (0, 0, 255), 2)  # X: 红
            cv2.arrowedLine(vis, origin, tuple(axis_2d[2]), (0, 255, 0), 2)  # Y: 绿
            cv2.arrowedLine(vis, origin, tuple(axis_2d[3]), (255, 0, 0), 2)  # Z: 蓝

        return vis
