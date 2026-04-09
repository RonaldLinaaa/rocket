"""
根据 dataset/rgb 中的帧与 pose.csv 的仿真时间，导出与真实仿真时间同步的视频。

输出: dataset/landing_simulation.mp4
  - 帧顺序按 frame 索引 (即时间顺序)
  - 帧率 = 总帧数 / 总仿真时长，使 1 秒视频 = 1 秒仿真
  - 可选在画面上叠加仿真时间与高度
"""

import csv
from pathlib import Path

import cv2
import numpy as np


def export_video(
    dataset_dir: str = "dataset",
    output_name: str = "landing_simulation.mp4",
    show_overlay: bool = True,
    fps_override: float | None = None,
):
    """
    将 dataset/rgb 下按帧序排列的图片导出为与仿真时间同步的 MP4。

    参数:
      dataset_dir: 数据集根目录 (含 rgb/ 与 pose.csv)
      output_name: 输出文件名 (置于 dataset_dir 下)
      show_overlay: 是否在画面上叠加 time(s) 与 alt(m)
      fps_override: 若指定则固定该 fps；否则按 总帧数/仿真时长 使视频时长=仿真时长
    """
    dataset_dir = Path(dataset_dir)
    rgb_dir = dataset_dir / "rgb"
    pose_path = dataset_dir / "pose.csv"
    out_path = dataset_dir / output_name

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"未找到目录: {rgb_dir}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"未找到: {pose_path}")

    # 读取 pose.csv 获取每帧的仿真时间与高度
    frames_meta = []
    with open(pose_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames_meta.append({
                "frame": int(row["frame"]),
                "time": float(row["time"]),
                "z": float(row["z"]),
            })

    if not frames_meta:
        raise ValueError("pose.csv 为空")

    # 按 frame 索引排序，确保顺序正确
    frames_meta.sort(key=lambda x: x["frame"])

    total_sim_time = frames_meta[-1]["time"] - frames_meta[0]["time"]
    n_frames = len(frames_meta)
    if fps_override is not None:
        fps = fps_override
    else:
        fps = n_frames / total_sim_time if total_sim_time > 0 else 25.0
        fps = max(1.0, min(60.0, fps))

    # 取第一帧以获取尺寸
    first_frame_id = f"{frames_meta[0]['frame']:05d}"
    first_img_path = rgb_dir / f"frame_{first_frame_id}.png"
    if not first_img_path.exists():
        raise FileNotFoundError(f"未找到首帧: {first_img_path}")

    sample = cv2.imread(str(first_img_path))
    if sample is None:
        raise RuntimeError(f"无法读取图像: {first_img_path}")
    h, w = sample.shape[:2]

    # 视频编码 (MP4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频: {out_path}")

    print(f"[export_video] 总帧数={n_frames}, 仿真时长={total_sim_time:.2f}s, "
          f"输出 fps={fps:.2f}, 预计视频时长={n_frames/fps:.2f}s")
    print(f"[export_video] 输出: {out_path}")

    for i, meta in enumerate(frames_meta):
        frame_id = f"{meta['frame']:05d}"
        img_path = rgb_dir / f"frame_{frame_id}.png"
        if not img_path.exists():
            print(f"  [WARN] 跳过缺失帧: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] 无法读取: {img_path}")
            continue

        if show_overlay:
            t_str = f"t = {meta['time']:.2f} s"
            z_str = f"Alt = {meta['z']:.1f} m"
            cv2.putText(
                img, t_str, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2,
            )
            cv2.putText(
                img, z_str, (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2,
            )

        writer.write(img)
        if (i + 1) % 200 == 0:
            print(f"  已写入 {i + 1}/{n_frames} 帧")

    writer.release()
    print(f"[export_video] 完成: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="按仿真时间导出 dataset 视频")
    parser.add_argument("--data", default="dataset", help="数据集根目录")
    parser.add_argument("--out", default="landing_simulation.mp4", help="输出文件名")
    parser.add_argument("--no-overlay", action="store_true", help="不叠加时间/高度")
    parser.add_argument("--fps", type=float, default=None, help="固定帧率 (默认按仿真时长)")
    args = parser.parse_args()

    export_video(
        dataset_dir=args.data,
        output_name=args.out,
        show_overlay=not args.no_overlay,
        fps_override=args.fps,
    )
