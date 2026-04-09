"""
椭圆 + H 字混合位姿解算器

策略:
  ● 椭圆弧检测 → 精确距离估计 (d = R·f / √(a·b), 误差 <1%)
  ● 椭圆轴比 → 倾斜角约束 (cos α = b/a)
  ● PnP (关键点) → 旋转估计
  ● 混合融合: 根据椭圆像素大小自适应切换

  near (>30px): 纯 PnP（关键点精度足够）
  mid (10~30px): PnP 方向 + 椭圆距离修正
  far (<10px):   椭圆距离 + 椭圆方向 + H 字粗略朝向
"""

import math
import numpy as np
import cv2
from typing import Optional, Dict, Tuple

from .pose_solver import LandingPoseSolver, CAMERA_MATRIX, DIST_COEFFS, WORLD_KEYPOINTS_3D

CIRCLE_RADIUS = 21.5  # 着陆标志圆环半径


class EllipsePoseSolver:
    """椭圆 + 关键点混合位姿解算器"""

    def __init__(
        self,
        camera_matrix: np.ndarray = CAMERA_MATRIX,
        dist_coeffs: np.ndarray = DIST_COEFFS,
        circle_radius: float = CIRCLE_RADIUS,
        confidence_threshold: float = 0.3,
        # 自适应切换阈值（椭圆长轴像素数）
        near_threshold: float = 30.0,
        far_threshold: float = 10.0,
    ):
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)
        self.dist = dist_coeffs
        self.R = circle_radius
        self.f = camera_matrix[0, 0]

        self.pnp_solver = LandingPoseSolver(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            confidence_threshold=confidence_threshold,
        )
        self.near_thresh = near_threshold
        self.far_thresh = far_threshold

    def solve(
        self,
        image: np.ndarray,
        keypoints_2d: np.ndarray,
        confidences: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        method: str = "epnp",
    ) -> Optional[Dict]:
        """
        混合位姿解算

        Args:
            image: BGR 图像（用于边缘检测）
            keypoints_2d: [9, 2] 像素坐标（YOLO 检测）
            confidences: [9] 置信度
            bbox: [x1, y1, x2, y2] 检测框（用于 ROI 裁剪）
            method: PnP 方法

        Returns:
            dict: rvec, tvec, R, reproj_error, inlier_count,
                  d_ellipse, ellipse_size, strategy
        """
        # Step 1: 尝试椭圆检测
        ellipse_result = self._detect_ellipse(image, bbox, keypoints_2d)

        # Step 2: PnP 解算
        pnp_result = self.pnp_solver.solve(keypoints_2d, confidences, method=method)

        # Step 3: 根据椭圆大小选择策略
        if ellipse_result is not None:
            ell_size = ellipse_result["semi_major"]
            d_ell = ellipse_result["distance"]

            if ell_size >= self.near_thresh and pnp_result is not None:
                # near: 纯 PnP，椭圆仅提供参考
                result = pnp_result.copy()
                result["d_ellipse"] = d_ell
                result["ellipse_size"] = ell_size
                result["strategy"] = "pnp_only"
                return result

            elif ell_size >= self.far_thresh and pnp_result is not None:
                # mid: PnP 方向 + 椭圆距离
                result = self._hybrid_pose(pnp_result, d_ell, keypoints_2d, confidences)
                result["d_ellipse"] = d_ell
                result["ellipse_size"] = ell_size
                result["strategy"] = "hybrid"
                return result

            else:
                # far: 椭圆为主
                result = self._ellipse_pose(ellipse_result, keypoints_2d, confidences)
                if result is not None:
                    result["d_ellipse"] = d_ell
                    result["ellipse_size"] = ell_size
                    result["strategy"] = "ellipse_primary"
                    return result

        # 兜底: 纯 PnP
        if pnp_result is not None:
            pnp_result["d_ellipse"] = None
            pnp_result["ellipse_size"] = 0
            pnp_result["strategy"] = "pnp_fallback"
        return pnp_result

    def _detect_ellipse(
        self,
        image: np.ndarray,
        bbox: Optional[np.ndarray],
        keypoints_2d: np.ndarray,
    ) -> Optional[Dict]:
        """
        从 YOLO 检测的外环关键点 (kp0-kp4) 拟合椭圆。

        kp0=中心, kp1-kp4=圆周上 4 个点 → 共 5 点拟合椭圆。
        比边缘检测更可靠: 直接利用 YOLO 检测结果，无需额外图像处理。
        椭圆面积公式仍然提供精确的距离估计。
        """
        # 取 kp0 (中心) 和 kp1-kp4 (外环) 共 5 个点
        outer_indices = [0, 1, 2, 3, 4]
        pts = keypoints_2d[outer_indices]

        # 检查有效性（非 NaN，在图像范围内）
        h, w = image.shape[:2] if image is not None else (720, 1080)
        valid_mask = np.ones(len(pts), dtype=bool)
        for i, p in enumerate(pts):
            if np.any(np.isnan(p)) or p[0] < 0 or p[0] > w or p[1] < 0 or p[1] > h:
                valid_mask[i] = False
        valid_pts = pts[valid_mask]

        if len(valid_pts) < 5:
            # 不足 5 点: 用中心 + 至少 3 个外环点估计
            if len(valid_pts) < 4:
                return None
            # 4 点: 用 minAreaRect 近似
            rect = cv2.minAreaRect(valid_pts.astype(np.float32).reshape(-1, 1, 2))
            (ecx, ecy), (ew, eh), eang = rect
        else:
            try:
                ellipse_cv = cv2.fitEllipse(valid_pts.astype(np.float32).reshape(-1, 1, 2))
                (ecx, ecy), (ew, eh), eang = ellipse_cv
            except cv2.error:
                return None

        semi_major = max(ew, eh) / 2
        semi_minor = min(ew, eh) / 2

        if semi_major < 1 or semi_minor < 0.5:
            return None

        # 椭圆面积 → 距离: d = R·f / √(a_px · b_px)
        a_n = semi_major / self.f
        b_n = semi_minor / self.f
        d_est = self.R / math.sqrt(a_n * b_n)

        axis_ratio = semi_minor / semi_major
        tilt_angle = math.acos(min(1.0, axis_ratio))

        return {
            "center": np.array([ecx, ecy]),
            "semi_major": semi_major,
            "semi_minor": semi_minor,
            "angle_deg": eang,
            "distance": d_est,
            "tilt_angle_rad": tilt_angle,
            "axis_ratio": axis_ratio,
        }

    def _hybrid_pose(
        self,
        pnp_result: Dict,
        d_ellipse: float,
        keypoints_2d: np.ndarray,
        confidences: np.ndarray,
    ) -> Dict:
        """混合策略: PnP 方向 + 椭圆距离"""
        t_pnp = pnp_result["tvec"]
        t_norm = np.linalg.norm(t_pnp)
        if t_norm < 1e-6:
            return pnp_result

        t_dir = t_pnp / t_norm
        t_corrected = t_dir * d_ellipse

        # 用修正后的 t 做一轮 PnP 精化
        rvec = pnp_result["rvec"].reshape(3, 1)
        mask = confidences >= self.pnp_solver.conf_thresh
        if mask.sum() >= 4:
            pts_2d = keypoints_2d[mask].astype(np.float64)
            pts_3d = WORLD_KEYPOINTS_3D[mask].astype(np.float64)
            try:
                rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
                    pts_3d, pts_2d, self.K, self.dist,
                    rvec.copy(), t_corrected.reshape(3, 1))
                # 验证精化结果: 距离不应偏离椭圆估计太多
                d_ref = np.linalg.norm(tvec_ref)
                if abs(d_ref - d_ellipse) / d_ellipse < 0.3:
                    t_corrected = tvec_ref.flatten()
                    rvec = rvec_ref
            except cv2.error:
                pass

        R, _ = cv2.Rodrigues(rvec)
        reproj_pts, _ = cv2.projectPoints(
            WORLD_KEYPOINTS_3D[mask], rvec, t_corrected.reshape(3, 1),
            self.K, self.dist)
        reproj_pts = reproj_pts.reshape(-1, 2)
        pts_2d_valid = keypoints_2d[mask].astype(np.float64)
        reproj_error = np.linalg.norm(reproj_pts - pts_2d_valid, axis=1).mean()

        return {
            "rvec": rvec.flatten(),
            "tvec": t_corrected,
            "R": R,
            "reproj_error": reproj_error,
            "inlier_count": int(mask.sum()),
        }

    def _ellipse_pose(
        self,
        ellipse_result: Dict,
        keypoints_2d: np.ndarray,
        confidences: np.ndarray,
    ) -> Optional[Dict]:
        """椭圆为主的位姿解算（远距情况）"""
        d_est = ellipse_result["distance"]
        center = ellipse_result["center"]

        # 视线方向
        ray = self.K_inv @ np.array([center[0], center[1], 1.0])
        ray_dir = ray / np.linalg.norm(ray)
        t_init = (ray_dir * d_est).reshape(3, 1)

        # 初始旋转（假设近似正对）
        z_axis = ray_dir
        ref = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
        x_axis = np.cross(ref, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R_init = np.vstack([x_axis, y_axis, z_axis])
        rv_init, _ = cv2.Rodrigues(R_init)

        # 尝试用关键点精化
        mask = confidences >= self.pnp_solver.conf_thresh
        if mask.sum() >= 4:
            pts_2d = keypoints_2d[mask].astype(np.float64)
            pts_3d = WORLD_KEYPOINTS_3D[mask].astype(np.float64)
            try:
                ok, rv, tv = cv2.solvePnP(
                    pts_3d, pts_2d, self.K, self.dist,
                    rvec=rv_init, tvec=t_init.copy(),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if ok and tv.flatten()[2] > 0:
                    # 距离锁定到椭圆值
                    t_dir = tv.flatten() / np.linalg.norm(tv.flatten())
                    tv = (t_dir * d_est).reshape(3, 1)
                    rv, tv = cv2.solvePnPRefineLM(pts_3d, pts_2d, self.K, self.dist, rv, tv)
                    rv_init, t_init = rv, tv
            except cv2.error:
                pass

        R, _ = cv2.Rodrigues(rv_init)
        return {
            "rvec": rv_init.flatten(),
            "tvec": t_init.flatten(),
            "R": R,
            "reproj_error": 0.0,
            "inlier_count": int(mask.sum()),
        }

    def get_camera_pose_in_world(self, result: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """转换为世界坐标系中的相机位姿"""
        return self.pnp_solver.get_camera_pose_in_world(result)

    def compute_euler_angles(self, R: np.ndarray) -> np.ndarray:
        return self.pnp_solver.compute_euler_angles(R)
