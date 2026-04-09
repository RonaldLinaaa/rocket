"""
Module 2 — Synthetic Scene Renderer
可复用火箭着陆场景合成图像渲染器

替代Blender的纯Python/OpenCV渲染器，生成火箭下降过程的
箭载相机视角合成图像数据集。

渲染内容:
  - 地面纹理 (模拟着陆场地)
  - 合作标志 (外圆环 + H型符号), 直径 ≈ 40m
  - 透视投影 (60° FOV, 1920×1080)
  - 光照变化、噪声、运动模糊等增广效果

输入: trajectory.csv
输出:
  dataset/rgb/frame_XXXXX.png
  dataset/depth/frame_XXXXX.png
  dataset/mask/frame_XXXXX.png
  dataset/pose.csv
  dataset/velocity.csv
  dataset/acceleration.csv
  dataset/annotations/labels/frame_XXXXX.txt  (YOLO格式bbox)
"""

import numpy as np
import cv2
import csv
import os
import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import json

# ------------------------------------------------------------------ #
#  配置参数
# ------------------------------------------------------------------ #
IMG_W, IMG_H = 1920, 1080
FOV_DEG      = 60.0
MARKER_DIAM  = 40.0   # m  着陆标志直径
GROUND_SIZE  = 5000   # m  地面场景范围
RING_THICK   = 3    # m  外环宽度

# 相机内参
FOV_RAD   = math.radians(FOV_DEG)
FX        = IMG_W / (2 * math.tan(FOV_RAD / 2))
FY        = FX
CX        = IMG_W / 2.0
CY        = IMG_H / 2.0

# 地面纹理尺寸 (像素/米)
TEX_SCALE = 4   # 用于地面纹理分辨率

# ------------------------------------------------------------------ #
#  工具函数
# ------------------------------------------------------------------ #

def euler_to_rotation_matrix(roll, pitch, yaw):
    """ZYX Euler → 旋转矩阵 (body→world)"""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    Rz = np.array([[cy, -sy, 0],[sy,  cy, 0],[0,   0,  1]])
    Ry = np.array([[cp, 0, sp],[0,  1,  0],[-sp,0, cp]])
    Rx = np.array([[1,  0,  0],[0,  cr,-sr],[0, sr,  cr]])
    return Rz @ Ry @ Rx


def project_3d_to_2d(pts_world, cam_pos, R_body2world):
    """
    将世界坐标系中的3D点投影到图像平面。
    相机固定在火箭底部,朝下看 (即相机Z轴 = 世界-Z轴经姿态旋转后)。

    pts_world: (N,3)
    cam_pos:   (3,)
    R_body2world: (3,3)
    """
    # 相机坐标系: body frame中相机朝-Z (即朝下)
    # R_cam2world = R_body2world @ R_cam_offset
    # R_cam_offset: 相机朝下 → cam_z = body_-z → 绕X轴旋转180°
    R_cam_offset = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)
    R_cam2world  = R_body2world @ R_cam_offset

    # world → camera
    R_world2cam = R_cam2world.T
    pts_cam = (R_world2cam @ (pts_world - cam_pos).T).T   # (N,3)

    # 只保留相机前方 (z > 0.1m)
    valid = pts_cam[:,2] > 0.1
    u = np.full(len(pts_cam), -1.0)
    v = np.full(len(pts_cam), -1.0)
    u[valid] = FX * pts_cam[valid,0] / pts_cam[valid,2] + CX
    v[valid] = FY * pts_cam[valid,1] / pts_cam[valid,2] + CY
    z_cam    = pts_cam[:,2]
    return u, v, z_cam, valid


def generate_ground_texture(size_px):
    """生成逼真的地面纹理 (混凝土+标线模拟)"""
    rng = np.random.default_rng(42)
    # 基础混凝土颜色
    base = rng.integers(90, 130, (size_px, size_px, 3), dtype=np.uint8)
    # 随机裂缝纹理
    for _ in range(80):
        x1 = rng.integers(0, size_px)
        y1 = rng.integers(0, size_px)
        ang = rng.uniform(0, 2*math.pi)
        length = rng.integers(30, 200)
        x2 = int(x1 + length * math.cos(ang))
        y2 = int(y1 + length * math.sin(ang))
        cv2.line(base, (x1,y1), (x2,y2),
                 (int(base[y1,x1,0])-20,)*3, 1)
    # 网格线 (模拟地面标记)
    for i in range(0, size_px, size_px//10):
        cv2.line(base, (i,0), (i,size_px), (70,70,70), 1)
        cv2.line(base, (0,i), (size_px,i), (70,70,70), 1)
    return base


def draw_landing_marker_on_texture(tex, tex_res_per_m, center_world_m):
    """
    在地面纹理上绘制着陆合作标志 (外圆环 + H符号)
    tex: numpy RGB图像
    tex_res_per_m: 像素/米
    center_world_m: 标志中心世界坐标 (x,y) 单位米, 相对纹理原点
    """
    cx_px = int(center_world_m[0] * tex_res_per_m)
    cy_px = int(center_world_m[1] * tex_res_per_m)
    r_outer = int(MARKER_DIAM / 2 * tex_res_per_m)
    r_inner = int((MARKER_DIAM / 2 - RING_THICK) * tex_res_per_m)
    thick   = int(RING_THICK * tex_res_per_m)

    img_pil = Image.fromarray(tex)
    draw = ImageDraw.Draw(img_pil)

    # 白色填充外圆
    draw.ellipse([(cx_px-r_outer, cy_px-r_outer),
                  (cx_px+r_outer, cy_px+r_outer)],
                 fill=(255,255,255), outline=(255,255,255))
    # 灰色内圆 (圆环效果)
    draw.ellipse([(cx_px-r_inner, cy_px-r_inner),
                  (cx_px+r_inner, cy_px+r_inner)],
                 fill=(160,160,160), outline=(160,160,160))

    # H 符号  (白色, 粗线)
    h_size = r_inner * 0.65
    h_thick = max(2, int(r_inner * 0.15))
    # 左竖
    x_left  = int(cx_px - h_size * 0.45)
    x_right = int(cx_px + h_size * 0.45)
    y_top   = int(cy_px - h_size * 0.7)
    y_bot   = int(cy_px + h_size * 0.7)
    y_mid   = cy_px
    draw.rectangle([x_left - h_thick//2, y_top,
                    x_left + h_thick//2, y_bot], fill=(255,255,255))
    # 右竖
    draw.rectangle([x_right - h_thick//2, y_top,
                    x_right + h_thick//2, y_bot], fill=(255,255,255))
    # 横梁
    draw.rectangle([x_left - h_thick//2, y_mid - h_thick//2,
                    x_right + h_thick//2, y_mid + h_thick//2],
                   fill=(255,255,255))

    return np.array(img_pil)


def get_marker_bbox_in_image(cam_pos, R_body2world, altitude):
    """
    计算着陆标志在图像中的YOLO格式bounding box。
    标志中心在世界坐标 (0,0,0), 直径 MARKER_DIAM。
    返回 (cx_n, cy_n, w_n, h_n) 归一化坐标，若不可见返回None。
    """
    # 采样标志轮廓上的若干点
    N_pts = 64
    angles = np.linspace(0, 2*math.pi, N_pts, endpoint=False)
    r = MARKER_DIAM / 2
    pts = np.array([[r*math.cos(a), r*math.sin(a), 0.0] for a in angles])

    u, v, z_cam, valid = project_3d_to_2d(pts, cam_pos, R_body2world)

    if valid.sum() < 4:
        return None, None

    u_v = u[valid]; v_v = v[valid]
    # 裁剪到图像范围内但允许少量超出
    u_min, u_max = u_v.min(), u_v.max()
    v_min, v_max = v_v.min(), v_v.max()

    if u_max < 0 or u_min > IMG_W or v_max < 0 or v_min > IMG_H:
        return None, None

    # 裁剪
    u_min = max(0, u_min); u_max = min(IMG_W-1, u_max)
    v_min = max(0, v_min); v_max = min(IMG_H-1, v_max)

    bw = u_max - u_min
    bh = v_max - v_min
    if bw < 2 or bh < 2:
        return None, None

    # YOLO 归一化
    cx_n = ((u_min + u_max) / 2) / IMG_W
    cy_n = ((v_min + v_max) / 2) / IMG_H
    w_n  = bw / IMG_W
    h_n  = bh / IMG_H
    return (cx_n, cy_n, w_n, h_n), (int(u_min), int(v_min), int(u_max), int(v_max))


def render_frame(cam_pos, R_body2world, ground_tex, tex_res_per_m,
                 ground_half, frame_noise=0.0, motion_blur=False):
    """
    渲染单帧图像。
    cam_pos: (3,) 相机世界坐标 (等同于火箭质心)
    R_body2world: (3,3)
    ground_tex: 地面纹理 numpy 图
    tex_res_per_m: 像素/米
    ground_half: 地面半尺寸(m)

    返回 (rgb_img, depth_img, mask_img)
    """
    altitude = cam_pos[2]
    img     = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    depth   = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    mask    = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    # --- 背景天空 (高度>0时才有天空) ---
    # 为降低计算量，使用光线投射到地面平面的方式
    # 构建地面点网格 (稀疏采样再插值)

    # 对每个像素，反投影到地面平面 z=0
    # 优化: 先计算图像四角+边缘，然后用透视变形填充

    SAMPLE = 4  # 降采样因子
    rows = np.arange(0, IMG_H, SAMPLE)
    cols = np.arange(0, IMG_W, SAMPLE)
    uu, vv = np.meshgrid(cols, rows)  # (nh, nw)

    # 相机射线方向 (相机坐标系)
    ray_x = (uu - CX) / FX
    ray_y = (vv - CY) / FY
    ray_z = np.ones_like(ray_x)

    # 相机到世界坐标系
    R_cam_offset = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)
    R_cam2world  = R_body2world @ R_cam_offset

    rays_flat = np.stack([ray_x.ravel(), ray_y.ravel(), ray_z.ravel()], axis=1)  # (N,3)
    rays_world = (R_cam2world @ rays_flat.T).T  # (N,3)

    # 与 z=0 平面求交: cam_pos[2] + t*ray_z_world = 0 → t = -cam_pos[2]/ray_z_world
    rz_world = rays_world[:,2]
    valid_mask = rz_world < -1e-6   # 射线朝下才能打到地面
    t = np.where(valid_mask, -cam_pos[2] / rz_world, np.inf)

    hit_x = cam_pos[0] + t * rays_world[:,0]
    hit_y = cam_pos[1] + t * rays_world[:,1]

    nh, nw = uu.shape
    hit_x = hit_x.reshape(nh, nw)
    hit_y = hit_y.reshape(nh, nw)
    valid_2d = valid_mask.reshape(nh, nw)
    t_2d     = t.reshape(nh, nw)

    # 纹理采样 (双线性插值)
    tex_h, tex_w = ground_tex.shape[:2]
    # 世界坐标→纹理坐标
    tex_u = ((hit_x + ground_half) * tex_res_per_m).astype(np.float32)
    tex_v = ((ground_half - hit_y) * tex_res_per_m).astype(np.float32)
    tex_u = np.clip(tex_u, 0, tex_w-1)
    tex_v = np.clip(tex_v, 0, tex_h-1)

    # 采样纹理
    sampled = cv2.remap(ground_tex.astype(np.float32),
                        tex_u.astype(np.float32),
                        tex_v.astype(np.float32),
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)

    # 天空颜色 (蓝色渐变)
    sky_grad = np.zeros((nh, nw, 3), dtype=np.float32)
    sky_grad[:,:,0] = np.linspace(100, 60, nh)[:,None]   # B
    sky_grad[:,:,1] = np.linspace(130, 90, nh)[:,None]   # G
    sky_grad[:,:,2] = np.linspace(160, 120, nh)[:,None]  # R (BGR order)
    sky_grad += np.random.normal(0, 3, sky_grad.shape)

    # 合并地面/天空
    small_img = np.where(valid_2d[:,:,None], sampled, sky_grad).astype(np.uint8)

    # 光照效果: 距离越远越暗
    if altitude > 100:
        dist_norm = np.clip(t_2d / (altitude * 1.5), 0, 1)
        atmos = 1.0 - 0.3 * dist_norm
        small_img = (small_img * atmos[:,:,None]).astype(np.uint8)

    # 上采样到全分辨率
    img = cv2.resize(small_img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    # 深度图 (上采样)
    depth_small = np.where(valid_2d, t_2d, 0).astype(np.float32)
    depth = cv2.resize(depth_small, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    # --- mask: 着陆标志区域 ---
    # 将地面标志轮廓投影，填充
    N_pts = 128
    angles = np.linspace(0, 2*math.pi, N_pts, endpoint=False)
    r = MARKER_DIAM / 2
    pts = np.array([[r*math.cos(a), r*math.sin(a), 0.0] for a in angles])
    u_m, v_m, _, val = project_3d_to_2d(pts, cam_pos, R_body2world)
    if val.sum() >= 3:
        contour = np.array([[int(u_m[i]), int(v_m[i])]
                             for i in range(len(u_m)) if val[i]
                             and 0<=u_m[i]<IMG_W and 0<=v_m[i]<IMG_H],
                            dtype=np.int32)
        if len(contour) >= 3:
            cv2.fillPoly(mask, [contour], 255)

    # --- 图像增广 ---
    if frame_noise > 0:
        noise = np.random.normal(0, frame_noise, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if motion_blur:
        k = random.choice([3, 5])
        kernel = np.zeros((k, k))
        kernel[k//2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)

    return img, depth, mask


# ------------------------------------------------------------------ #
#  主渲染循环
# ------------------------------------------------------------------ #

def main(traj_csv='trajectory.csv', out_dir='dataset',
         max_frames=None, skip=1):
    """
    读取轨迹CSV，逐帧渲染合成图像。

    skip: 帧间隔 (降低数据量时使用，默认1=全帧)
    max_frames: 最多渲染帧数 (None=全部)
    """
    print(f"[INFO] 读取轨迹: {traj_csv}")
    traj = []
    with open(traj_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            traj.append({k: float(v) for k,v in row.items()})

    if max_frames:
        traj = traj[::skip][:max_frames]
    else:
        traj = traj[::skip]

    print(f"[INFO] 共 {len(traj)} 帧待渲染")

    # 创建目录
    for d in ['rgb','depth','mask','annotations/labels']:
        Path(f"{out_dir}/{d}").mkdir(parents=True, exist_ok=True)

    # 生成地面纹理
    print("[INFO] 生成地面纹理...")
    ground_half = GROUND_SIZE / 2
    tex_res_per_m = 2   # 2像素/米，地面纹理
    tex_size_px   = int(GROUND_SIZE * tex_res_per_m)
    # 限制纹理大小防止OOM
    tex_size_px   = min(tex_size_px, 4096)
    actual_res    = tex_size_px / GROUND_SIZE

    ground_tex = generate_ground_texture(tex_size_px)
    # 在地面纹理中央绘制着陆标志
    marker_center = (ground_half, ground_half)   # 纹理坐标(像素)
    ground_tex = draw_landing_marker_on_texture(
        ground_tex, actual_res, (ground_half, ground_half))
    print(f"[INFO] 地面纹理尺寸: {tex_size_px}×{tex_size_px}px  ({GROUND_SIZE}m×{GROUND_SIZE}m)")

    # 输出CSV文件句柄
    pose_f = open(f"{out_dir}/pose.csv", 'w', newline='')
    vel_f  = open(f"{out_dir}/velocity.csv", 'w', newline='')
    acc_f  = open(f"{out_dir}/acceleration.csv", 'w', newline='')

    pose_w = csv.writer(pose_f)
    vel_w  = csv.writer(vel_f)
    acc_w  = csv.writer(acc_f)

    pose_w.writerow(['frame','time','x','y','z','roll','pitch','yaw'])
    vel_w.writerow( ['frame','time','vx','vy','vz'])
    acc_w.writerow( ['frame','time','ax','ay','az'])

    # 每100帧报告一次
    n_total   = len(traj)
    n_visible = 0

    for idx, row in enumerate(traj):
        frame_id = f"{idx:05d}"
        alt = row['z']

        cam_pos  = np.array([row['x'], row['y'], row['z']])
        R = euler_to_rotation_matrix(row['roll'], row['pitch'], row['yaw'])

        # 动态增广参数
        noise_sigma = max(0, min(15, (3000 - alt) / 200))
        do_blur = (alt < 500) and (random.random() < 0.3)

        # 渲染
        rgb, depth_img, mask_img = render_frame(
            cam_pos, R, ground_tex, actual_res, ground_half,
            frame_noise=noise_sigma, motion_blur=do_blur
        )

        # 计算bbox
        bbox, bbox_px = get_marker_bbox_in_image(cam_pos, R, alt)

        # 在RGB图上叠加bbox (调试可视化, 可注释掉)
        if bbox_px is not None:
            x1,y1,x2,y2 = bbox_px
            cv2.rectangle(rgb, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(rgb, f"H:{alt:.0f}m", (x1,max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            n_visible += 1

        # 高度信息水印
        cv2.putText(rgb, f"Alt={alt:.1f}m  t={row['time']:.1f}s  "
                         f"Vz={row['vz']:.1f}m/s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # 保存图像
        cv2.imwrite(f"{out_dir}/rgb/frame_{frame_id}.png", rgb)

        # 深度图 (归一化到uint16)
        d_max = max(alt * 1.5, 10)
        depth_uint16 = (np.clip(depth_img, 0, d_max) / d_max * 65535).astype(np.uint16)
        cv2.imwrite(f"{out_dir}/depth/frame_{frame_id}.png", depth_uint16)

        # 掩码
        cv2.imwrite(f"{out_dir}/mask/frame_{frame_id}.png", mask_img)

        # YOLO标注
        label_path = f"{out_dir}/annotations/labels/frame_{frame_id}.txt"
        with open(label_path, 'w') as lf:
            if bbox is not None:
                cx_n, cy_n, w_n, h_n = bbox
                lf.write(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

        # CSV写入
        pose_w.writerow([idx, row['time'], row['x'], row['y'], row['z'],
                         row['roll'], row['pitch'], row['yaw']])
        vel_w.writerow( [idx, row['time'], row['vx'], row['vy'], row['vz']])
        acc_w.writerow( [idx, row['time'], row['ax'], row['ay'], row['az']])

        if (idx+1) % 100 == 0 or idx == n_total-1:
            print(f"  [{idx+1}/{n_total}] Alt={alt:.0f}m  "
                  f"bbox={'YES' if bbox else 'NO'}")

    pose_f.close(); vel_f.close(); acc_f.close()

    # 写数据集元信息
    meta = {
        "total_frames": n_total,
        "frames_with_marker": n_visible,
        "image_resolution": [IMG_W, IMG_H],
        "fov_deg": FOV_DEG,
        "focal_length_px": FX,
        "principal_point": [CX, CY],
        "marker_diameter_m": MARKER_DIAM,
        "class_names": ["landing_marker"],
        "annotation_format": "YOLO",
        "source_trajectory": traj_csv
    }
    with open(f"{out_dir}/dataset_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] 渲染完成!")
    print(f"  总帧数:       {n_total}")
    print(f"  标志可见帧:   {n_visible} ({100*n_visible/max(n_total,1):.1f}%)")
    print(f"  输出目录:     {out_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='火箭着陆场景合成图像渲染器')
    parser.add_argument('--traj',       default='trajectory.csv')
    parser.add_argument('--out',        default='dataset')
    parser.add_argument('--skip',       type=int, default=1,
                        help='帧间隔 (节省时间)')
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()

    main(args.traj, args.out, args.max_frames, args.skip)
