from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# 与 rocket_trajectory_blender.py 保持一致
IMG_W = 1080
IMG_H = 720
LENS_MM = 24.0
SENSOR_WIDTH_MM = 36.0
ROCKET_RADIUS = 1.85
ROCKET_LENGTH = 42.0

# 相机局部位姿（相对火箭）
CAM_LOCAL_T = np.array([ROCKET_RADIUS + 0.15, 0.0, ROCKET_LENGTH * 0.85], dtype=float)
CAM_LOCAL_EULER_XYZ = np.array([math.radians(15.0), 0.0, math.radians(-90.0)], dtype=float)

# 内参（sensor_fit='HORIZONTAL'，此设置下 fx=fy=720）
FX = LENS_MM / SENSOR_WIDTH_MM * IMG_W
FY = FX
CX = IMG_W / 2.0
CY = IMG_H / 2.0
RANDOM_SEED = 42

# 与 rocket_trajectory_blender.py 一致：圆环外半径 (m)
RING_R_OUT = 21.5
# 圆环薄板在着陆区局部 Z（LZ_Z=0）：MARK_Z=0.02，几何中心 z_center=MARK_Z+SLAB_TH/2，外缘顶/底 Z
RING_Z_BOT = 0.02
RING_Z_TOP = 0.10
# 圆周采样点数（越多投影包络越紧）
RING_SAMPLE_SEGS = 128


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler_xyz_to_matrix(x: float, y: float, z: float) -> np.ndarray:
    # 与 Blender Euler('XYZ') 对齐：R = Rz @ Ry @ Rx
    return rot_z(z) @ rot_y(y) @ rot_x(x)


def project_world_to_image(world_pt: np.ndarray, cam_pos_world: np.ndarray, cam_rot_world: np.ndarray):
    # 世界 -> 相机坐标
    p_cam = cam_rot_world.T @ (world_pt - cam_pos_world)
    z_forward = -p_cam[2]  # Blender 相机前向为 -Z
    if z_forward <= 1e-6:
        return None

    u = FX * (p_cam[0] / z_forward) + CX
    v = CY - FY * (p_cam[1] / z_forward)
    return float(u), float(v)


def load_kps(kp_path: Path):
    data = json.loads(kp_path.read_text(encoding="utf-8"))
    # 保证 kp0..kp8 顺序
    kps = []
    for i in range(9):
        p = data[f"kp{i}"]
        kps.append(np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float))
    return kps


def load_trajectory(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "time": float(row["time"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "roll": float(row["roll"]),
                    "pitch": float(row["pitch"]),
                    "yaw": float(row["yaw"]),
                }
            )
    return rows


def landing_center_from_trajectory(traj_rows: list) -> Tuple[float, float]:
    """着陆区中心：与 Blender 脚本一致，取轨迹末行 (x,y)。"""
    end = traj_rows[-1]
    return float(end["x"]), float(end["y"])


def ring_outer_world_points(lz_x: float, lz_y: float) -> list[np.ndarray]:
    """圆环外圆在世界系下的采样点（底面+顶面两圈，透视下轴对齐包络更稳）。"""
    pts = []
    for z in (RING_Z_BOT, RING_Z_TOP):
        for i in range(RING_SAMPLE_SEGS):
            ang = 2 * math.pi * i / RING_SAMPLE_SEGS
            cx = lz_x + RING_R_OUT * math.cos(ang)
            cy = lz_y + RING_R_OUT * math.sin(ang)
            pts.append(np.array([cx, cy, z], dtype=float))
    return pts


def bbox_from_ring_projection(
    cam_pos_world: np.ndarray,
    cam_rot_world: np.ndarray,
    lz_x: float,
    lz_y: float,
) -> Tuple[list[float], float]:
    """由圆环外圆投影得到 COCO 格式 bbox [x,y,w,h] 与 area。"""
    us, vs = [], []
    for p in ring_outer_world_points(lz_x, lz_y):
        uv = project_world_to_image(p, cam_pos_world, cam_rot_world)
        if uv is None:
            continue
        us.append(uv[0])
        vs.append(uv[1])
    if not us:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)
    # 裁剪到图像内（训练用检测框应在画布范围内）
    u_min = max(0.0, min(u_min, IMG_W - 1e-6))
    u_max = max(0.0, min(u_max, IMG_W - 1e-6))
    v_min = max(0.0, min(v_min, IMG_H - 1e-6))
    v_max = max(0.0, min(v_max, IMG_H - 1e-6))
    w = u_max - u_min
    h = v_max - v_min
    if w <= 0 or h <= 0:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    return [u_min, v_min, w, h], float(w * h)


def try_bbox_from_white_mask(img_path: Path, white_threshold: int = 250) -> Optional[Tuple[list[float], float]]:
    """
    由渲染/分割 mask 估计标志物 bbox：假设标志物为高亮（近白）。
    若图中无足够白像素则返回 None。
    """
    if not img_path.exists():
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    arr = np.array(Image.open(img_path).convert("RGB"))
    if arr.size == 0:
        return None
    # 简单亮度阈值（可与 Otsu 等替换）
    gray = arr.astype(np.float32).mean(axis=-1)
    mask = gray >= white_threshold
    ys, xs = np.where(mask)
    if len(xs) < 50:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    w = float(x2 - x1 + 1)
    h = float(y2 - y1 + 1)
    return [float(x1), float(y1), w, h], w * h


def resolve_mask_path(base: Path, seq_name: str, frame_idx: int) -> Optional[Path]:
    """若存在序列目录下 masks，则返回 mask 路径（用于白区 bbox）。"""
    d = base / seq_name / "masks"
    for name in (f"{frame_idx:04d}.png", f"rocket_render_{frame_idx:04d}.png"):
        p = d / name
        if p.exists():
            return p
    return None


def resolve_rgb_path(base: Path, file_name: str) -> Path:
    """
    与 yolo_landing/convert_dataset.resolve_image_path 一致：
    标注 file_name 可为规范路径；实际 RGB 往往在 <序列>/rgb/<帧>.png。
    """
    direct = base / file_name
    if direct.exists():
        return direct
    parts = file_name.split("/")
    if len(parts) == 2:
        seq, fname = parts
        frame_str = fname.replace("rocket_render_", "")
        for sub in ("rgb", "images"):
            alt = base / seq / sub / frame_str
            if alt.exists():
                return alt
    return direct


# 自动发现序列时忽略的目录名
_DISCOVER_SKIP_DIRS = frozenset({"visualizations", "__pycache__", ".git"})


def discover_sequences(base: Path) -> List[Tuple[str, Path]]:
    """
    扫描 datasets 下子目录：含 trajectory*.csv 的视为一条序列。
    同一目录多个 trajectory 时取第一个并告警（应用 sequences.json 明确指定）。
    """
    out: List[Tuple[str, Path]] = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if d.name in _DISCOVER_SKIP_DIRS:
            continue
        csvs = sorted(d.glob("trajectory*.csv"))
        if not csvs:
            continue
        if len(csvs) > 1:
            print(
                f"[Warning] {d.name} 下有多个 trajectory*.csv，将使用 {csvs[0].name}；"
                f"多轨迹请使用 sequences.json 分列配置。"
            )
        out.append((d.name, csvs[0]))
    if not out:
        raise FileNotFoundError(
            f"未在 {base} 下发现任何 trajectory*.csv。"
            f"请添加数据子目录或创建 sequences.json。"
        )
    return out


def load_sequence_specs(base: Path) -> List[Tuple[str, Path]]:
    """
    加载序列列表（目录名, 轨迹 CSV 绝对路径）。
    若存在 datasets/sequences.json 则优先使用；否则自动发现。
    """
    cfg = base / "sequences.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        specs: List[Tuple[str, Path]] = []
        for item in data.get("sequences", []):
            dir_name = item["dir"]
            traj_rel = item["trajectory"]
            p = base / dir_name / traj_rel
            if not p.exists():
                raise FileNotFoundError(f"sequences.json 中轨迹不存在: {p}")
            specs.append((dir_name, p))
        if not specs:
            print("[Warning] sequences.json 中 sequences 为空，改为自动发现。")
            return discover_sequences(base)
        return specs
    return discover_sequences(base)


def get_range_label(height_z: float) -> str:
    """按高度划分远中近。"""
    if height_z > 2000.0:
        return "far"
    if height_z < 500.0:
        return "near"
    return "mid"


def make_subset_coco(base_coco: dict, image_ids: list[int]) -> dict:
    """根据 image_id 子集生成独立 COCO。"""
    image_id_set = set(image_ids)
    sub_images = [im for im in base_coco["images"] if im["id"] in image_id_set]
    sub_anns = [ann for ann in base_coco["annotations"] if ann["image_id"] in image_id_set]
    return {
        "info": base_coco["info"],
        "licenses": base_coco["licenses"],
        "images": sub_images,
        "annotations": sub_anns,
        "categories": base_coco["categories"],
    }


def _scale_kps_bbox_for_display(
    keypoints_flat: list,
    bbox: list[float],
    ann_w: int,
    ann_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[list[float], list[float]]:
    """将 COCO 中基于 ann_w×ann_h 的像素坐标缩放到实际 RGB 尺寸。"""
    sx = dst_w / float(ann_w) if ann_w else 1.0
    sy = dst_h / float(ann_h) if ann_h else 1.0
    new_kps: list[float] = []
    for i in range(0, len(keypoints_flat), 3):
        x, y, v = keypoints_flat[i], keypoints_flat[i + 1], int(keypoints_flat[i + 2])
        if v == 0:
            new_kps.extend([0.0, 0.0, 0.0])
        else:
            new_kps.extend([x * sx, y * sy, float(v)])
    nb = [
        bbox[0] * sx,
        bbox[1] * sy,
        bbox[2] * sx,
        bbox[3] * sy,
    ]
    return new_kps, nb


def visualize_samples(base: Path, coco: dict, sample_count: int = 12):
    """
    随机可视化：优先把关键点与 bbox 叠在真实 RGB 上。
    使用与训练相同的 RGB 路径解析（根目录 或 <序列>/rgb/）。
    若 RGB 分辨率与标注 width/height 不一致，会按比例缩放绘制坐标。
    若无任何 RGB，则退化为白底画布（与标注同尺寸）。
    """
    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        print(f"[Warning] Pillow 不可用，跳过可视化导出: {e}")
        return

    vis_dir = base / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    anns_by_img = {ann["image_id"]: ann for ann in coco["annotations"]}
    all_images = coco["images"]
    n = min(sample_count, len(all_images))
    rng = random.Random(RANDOM_SEED + 1)
    chosen = rng.sample(all_images, n)

    for item in chosen:
        image_id = item["id"]
        ann = anns_by_img[image_id]
        file_name = item["file_name"]
        img_path = resolve_rgb_path(base, file_name)

        ann_w, ann_h = int(item["width"]), int(item["height"])

        if img_path.exists():
            canvas = Image.open(img_path).convert("RGB")
            dst_w, dst_h = canvas.size
            rgb_ok = True
        else:
            canvas = Image.new("RGB", (ann_w, ann_h), (255, 255, 255))
            dst_w, dst_h = ann_w, ann_h
            rgb_ok = False

        draw_kps, draw_bbox = _scale_kps_bbox_for_display(
            ann["keypoints"], ann["bbox"], ann_w, ann_h, dst_w, dst_h
        )

        draw = ImageDraw.Draw(canvas)

        for i in range(0, len(draw_kps), 3):
            x, y, v = draw_kps[i], draw_kps[i + 1], int(draw_kps[i + 2])
            if v <= 0:
                continue
            xi, yi = int(round(x)), int(round(y))
            color = (0, 255, 0) if v == 2 else (255, 165, 0)
            r = max(3, min(8, dst_w // 160))
            draw.ellipse((xi - r, yi - r, xi + r, yi + r), fill=color, outline=(0, 0, 0), width=1)
            draw.text((xi + r + 2, yi - r), f"kp{i // 3}", fill=(255, 0, 0))

        b = draw_bbox
        if b[2] > 0 and b[3] > 0:
            x0, y0 = int(round(b[0])), int(round(b[1]))
            x1, y1 = int(round(b[0] + b[2])), int(round(b[1] + b[3]))
            draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 255), width=max(1, dst_w // 540))

        src_note = Path(file_name).name
        if rgb_ok:
            try:
                rel_rgb = img_path.relative_to(base)
            except ValueError:
                rel_rgb = img_path
            src_note = f"rgb={rel_rgb}"
            if (dst_w, dst_h) != (ann_w, ann_h):
                src_note += f" | scaled {ann_w}x{ann_h}->{dst_w}x{dst_h}"
        else:
            src_note = "no_rgb_white_bg"
        draw.text(
            (12, 12),
            f"id={image_id} {item['sequence']} fr={item['frame_id']} {item['split']} {item['range_label']}",
            fill=(0, 0, 255),
        )
        draw.text((12, 32), src_note, fill=(200, 0, 200))

        out_name = f"vis_{item['sequence']}_{item['frame_id']:04d}.png"
        out_path = vis_dir / out_name
        canvas.save(out_path)

    print(f"[Info] 可视化已导出: {vis_dir} ({n} 张)，叠加规则与 convert_dataset 路径一致")



def main():
    parser = argparse.ArgumentParser(description="生成 COCO 关键点标注（可扩展多序列）")
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="不导出随机可视化（仅写 JSON）",
    )
    parser.add_argument(
        "--vis-count",
        type=int,
        default=12,
        help="随机可视化样本数量",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    kp_path = base / "kp.json"
    seqs = load_sequence_specs(base)
    print(f"[Info] 将处理 {len(seqs)} 条序列: " + ", ".join(f"{n}:{p.name}" for n, p in seqs))

    kp_world = load_kps(kp_path)
    cam_local_rot = euler_xyz_to_matrix(*CAM_LOCAL_EULER_XYZ)

    coco = {
        "info": {
            "description": "Visible Light Visual Positioning keypoints",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "landing_marker",
                "supercategory": "marker",
                "keypoints": [f"kp{i}" for i in range(9)],
                "skeleton": [],
            }
        ],
    }

    image_id = 1
    ann_id = 1

    for seq_name, traj_path in seqs:
        traj = load_trajectory(traj_path)
        lz_x, lz_y = landing_center_from_trajectory(traj)
        lz_offset = np.array([lz_x, lz_y, 0.0], dtype=float)

        for frame_idx, row in enumerate(traj, start=1):
            rocket_t = np.array([row["x"], row["y"], row["z"]], dtype=float)
            rocket_r = euler_xyz_to_matrix(row["roll"], row["pitch"], row["yaw"])

            cam_pos_world = rocket_t + rocket_r @ CAM_LOCAL_T
            cam_rot_world = rocket_r @ cam_local_rot

            keypoints_flat = []
            visible_cnt = 0

            for kp in kp_world:
                world_pt = kp + lz_offset
                uv = project_world_to_image(world_pt, cam_pos_world, cam_rot_world)
                if uv is None:
                    keypoints_flat.extend([0.0, 0.0, 0])
                    continue

                u, v = uv
                in_img = (0.0 <= u < IMG_W) and (0.0 <= v < IMG_H)
                if in_img:
                    keypoints_flat.extend([u, v, 2])
                    visible_cnt += 1
                else:
                    # 落在图像外，按 COCO 规范可记为 v=1（标注但不可见）
                    keypoints_flat.extend([u, v, 1])

            # bbox：优先 mask 白区外接矩形；否则用圆环外圆投影轴对齐包络
            bbox_src = "ring_projection"
            mask_p = resolve_mask_path(base, seq_name, frame_idx)
            bbox_area: Optional[Tuple[list[float], float]] = None
            if mask_p is not None:
                bbox_area = try_bbox_from_white_mask(mask_p)
                if bbox_area is not None:
                    bbox_src = "mask_white"
            if bbox_area is None:
                bbox, area = bbox_from_ring_projection(
                    cam_pos_world, cam_rot_world, lz_x, lz_y
                )
            else:
                bbox, area = bbox_area

            file_name = f"{seq_name}/rocket_render_{frame_idx:04d}.png"
            coco["images"].append(
                {
                    "id": image_id,
                    "width": IMG_W,
                    "height": IMG_H,
                    "file_name": file_name,
                    "frame_id": frame_idx,
                    "sequence": seq_name,
                    "height_z": row["z"],
                    "range_label": get_range_label(row["z"]),
                    "landing_center_xy": [lz_x, lz_y],
                }
            )

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": keypoints_flat,
                    "num_keypoints": visible_cnt,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "bbox_source": bbox_src,
                }
            )

            image_id += 1
            ann_id += 1

    # 7:2:1 划分（训练/测试/验证）
    rng = random.Random(RANDOM_SEED)
    all_image_ids = [im["id"] for im in coco["images"]]
    rng.shuffle(all_image_ids)

    total = len(all_image_ids)
    n_train = int(total * 0.7)
    n_test = int(total * 0.2)
    n_val = total - n_train - n_test

    train_ids = all_image_ids[:n_train]
    test_ids = all_image_ids[n_train : n_train + n_test]
    val_ids = all_image_ids[n_train + n_test :]

    split_map = {}
    for i in train_ids:
        split_map[i] = "train"
    for i in test_ids:
        split_map[i] = "test"
    for i in val_ids:
        split_map[i] = "val"

    for im in coco["images"]:
        im["split"] = split_map[im["id"]]

    out_path = base / "annotations_coco_keypoints.json"
    out_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成: {out_path}")

    train_coco = make_subset_coco(coco, train_ids)
    test_coco = make_subset_coco(coco, test_ids)
    val_coco = make_subset_coco(coco, val_ids)

    (base / "annotations_coco_train.json").write_text(
        json.dumps(train_coco, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (base / "annotations_coco_test.json").write_text(
        json.dumps(test_coco, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (base / "annotations_coco_val.json").write_text(
        json.dumps(val_coco, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.no_vis and args.vis_count > 0:
        visualize_samples(base, coco, sample_count=args.vis_count)

    print(
        "images="
        f"{len(coco['images'])}, annotations={len(coco['annotations'])}, "
        f"train/test/val={n_train}/{n_test}/{n_val}"
    )


if __name__ == "__main__":
    main()
