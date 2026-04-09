"""
COCO Keypoints -> YOLO Pose 格式转换

标注说明:
  ann['bbox'] 为 COCO 格式 [x,y,w,h]（像素），由圆环外径投影外接矩形得到；
  若存在 datasets/<序列>/masks/ 下对应帧的二值/近白 mask，则优先用白区外接框。
  更新 annotations_coco_keypoints.json 后需重新运行本脚本。

YOLO Pose 标签 (每行一个目标):
  class_id  cx  cy  w  h  x1 y1 v1  x2 y2 v2  ...  x9 y9 v9
  所有坐标归一化到 [0, 1]

用法:
  python convert_dataset.py
  python convert_dataset.py --copy   # 复制图片而非创建符号链接
"""

import json
import os
import shutil
import argparse
from pathlib import Path


DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
ANN_FILE = DATASETS_DIR / "annotations_coco_keypoints.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "yolo_dataset"
NUM_KEYPOINTS = 9


def resolve_image_path(file_name: str) -> Path:
    """处理标注中 file_name 到实际图片路径的映射"""
    # 直接路径
    direct = DATASETS_DIR / file_name
    if direct.exists():
        return direct

    # rocket_render_XX/rocket_render_YYYY.png -> rocket_render_XX/rgb/YYYY.png
    parts = file_name.split("/")
    if len(parts) == 2:
        seq = parts[0]
        fname = parts[1]
        # rocket_render_0001.png -> 0001.png
        frame_str = fname.replace("rocket_render_", "")
        for sub in ("rgb", "images"):
            alt = DATASETS_DIR / seq / sub / frame_str
            if alt.exists():
                return alt

    return direct  # 返回原始路径, 后续会跳过不存在的


def convert(use_copy: bool = False):
    print(f"读取标注: {ANN_FILE}")
    with open(ANN_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 建立索引
    img_dict = {img["id"]: img for img in coco["images"]}
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]] = ann

    # 创建输出目录
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "skip": 0}

    for img_info in coco["images"]:
        img_id = img_info["id"]
        split = img_info.get("split", "train")
        if split == "test":
            split = "val"  # YOLO 只需要 train/val

        W = img_info["width"]
        H = img_info["height"]

        ann = ann_by_img.get(img_id)
        if ann is None:
            stats["skip"] += 1
            continue

        # 图片
        src_path = resolve_image_path(img_info["file_name"])
        if not src_path.exists():
            stats["skip"] += 1
            continue

        dst_img = OUTPUT_DIR / "images" / split / f"{img_id:06d}.png"
        if not dst_img.exists():
            if use_copy:
                shutil.copy2(src_path, dst_img)
            else:
                try:
                    os.symlink(src_path, dst_img)
                except OSError:
                    shutil.copy2(src_path, dst_img)

        # 标签（裁剪 bbox 到图像范围）
        bbox = ann["bbox"]  # [x, y, w, h] 像素, 左上角
        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(W, bbox[0] + bbox[2])
        y2 = min(H, bbox[1] + bbox[3])
        if x2 <= x1 or y2 <= y1:
            stats["skip"] += 1
            continue
        cx = ((x1 + x2) / 2.0) / W
        cy = ((y1 + y2) / 2.0) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H

        kps_flat = ann["keypoints"]  # [x1, y1, v1, x2, y2, v2, ...]
        kp_parts = []
        for i in range(NUM_KEYPOINTS):
            kx = kps_flat[i * 3] / W
            ky = kps_flat[i * 3 + 1] / H
            kv = int(kps_flat[i * 3 + 2])
            # 越界关键点：裁剪坐标到 [0,1]，可见性设为 0
            if kv > 0 and (kx < 0 or kx > 1 or ky < 0 or ky > 1):
                kv = 0
            kx = max(0.0, min(1.0, kx))
            ky = max(0.0, min(1.0, ky))
            kp_parts.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])

        line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp_parts)

        label_file = OUTPUT_DIR / "labels" / split / f"{img_id:06d}.txt"
        with open(label_file, "w") as lf:
            lf.write(line + "\n")

        stats[split] += 1

    # 生成 dataset.yaml
    yaml_content = f"""# YOLOv8 Pose - Landing Marker Keypoint Detection
path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

nc: 1
names:
  0: landing_marker

kpt_shape: [{NUM_KEYPOINTS}, 3]
flip_idx: [{', '.join(str(i) for i in range(NUM_KEYPOINTS))}]
"""
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"转换完成: train={stats['train']}, val={stats['val']}, skip={stats['skip']}")
    print(f"数据集配置: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true",
                        help="复制图片 (默认使用符号链接)")
    args = parser.parse_args()
    convert(use_copy=args.copy)
