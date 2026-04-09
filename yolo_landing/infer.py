"""
完整推理管线: YOLOv8-pose 关键点检测 -> PnP 位姿解算

流程:
  图像 -> YOLOv8-pose -> 9 个关键点 (u, v, conf)
       -> PnP(EPnP/IPPE) -> 相机 6-DoF 位姿 (R, t)

用法:
  # 单张图片
  python infer.py --source image.png --weights runs/landing_pose/weights/best.pt

  # 文件夹
  python infer.py --source datasets/rocket_render_01/rgb/ --weights best.pt

  # 视频
  python infer.py --source video.mp4 --weights best.pt --save-video
"""

import argparse
import time
import json
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请先安装 ultralytics: pip install ultralytics")

from pose_solver import LandingPoseSolver


def run_inference(
    weights: str,
    source: str,
    conf_thresh: float = 0.25,
    kpt_conf_thresh: float = 0.5,
    pnp_method: str = "epnp",
    save_vis: bool = True,
    save_video: bool = False,
    show: bool = False,
):
    model = YOLO(weights)
    solver = LandingPoseSolver(confidence_threshold=kpt_conf_thresh)

    source_path = Path(source)
    output_dir = Path("runs/infer_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集输入
    if source_path.is_dir():
        files = sorted(
            p for p in source_path.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
        )
    elif source_path.suffix.lower() in (".mp4", ".avi", ".mov"):
        files = [source_path]
    else:
        files = [source_path]

    pose_log = []
    video_writer = None

    for file_path in files:
        is_video = file_path.suffix.lower() in (".mp4", ".avi", ".mov")

        if is_video:
            cap = cv2.VideoCapture(str(file_path))
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        else:
            img = cv2.imread(str(file_path))
            if img is None:
                print(f"无法读取: {file_path}")
                continue
            frames = [img]

        for frame_idx, frame in enumerate(frames):
            t0 = time.time()

            # YOLOv8 推理
            results = model.predict(
                frame, conf=conf_thresh, verbose=False)

            det_time = time.time() - t0

            if len(results) == 0 or results[0].keypoints is None:
                if save_vis:
                    vis = frame.copy()
                    cv2.putText(vis, "No detection", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if show:
                        cv2.imshow("Landing Pose", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return
                continue

            # 取置信度最高的检测
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                best_idx = r.boxes.conf.argmax().item()
            else:
                best_idx = 0

            kpts_data = r.keypoints.data[best_idx].cpu().numpy()  # [K, 3]
            kpts_xy = kpts_data[:, :2]     # [K, 2] 像素坐标
            kpts_conf = kpts_data[:, 2]    # [K] 置信度

            # PnP 解算
            t1 = time.time()
            pose_result = solver.solve(kpts_xy, kpts_conf, method=pnp_method)
            pnp_time = time.time() - t1

            # 记录结果
            entry = {
                "file": str(file_path.name),
                "frame": frame_idx,
                "det_time_ms": det_time * 1000,
                "pnp_time_ms": pnp_time * 1000,
            }

            if pose_result is not None:
                R_wc, t_wc = solver.get_camera_pose_in_world(pose_result)
                euler = solver.compute_euler_angles(R_wc)
                entry.update({
                    "position": t_wc.tolist(),
                    "euler_deg": euler.tolist(),
                    "reproj_error": pose_result["reproj_error"],
                    "inliers": pose_result["inlier_count"],
                })
                status = (f"pos=({t_wc[0]:.1f},{t_wc[1]:.1f},{t_wc[2]:.1f}) "
                          f"reproj={pose_result['reproj_error']:.2f}px "
                          f"det={det_time*1000:.0f}ms pnp={pnp_time*1000:.1f}ms")
            else:
                status = "PnP failed"

            pose_log.append(entry)

            # 可视化
            if save_vis or show or save_video:
                vis = solver.visualize(frame, kpts_xy, kpts_conf, pose_result)

                if save_vis and not is_video:
                    out_path = output_dir / f"vis_{file_path.stem}.png"
                    cv2.imwrite(str(out_path), vis)

                if save_video and video_writer is None and is_video:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vout = output_dir / f"vis_{file_path.stem}.mp4"
                    video_writer = cv2.VideoWriter(str(vout), fourcc, 30, (w, h))

                if video_writer is not None:
                    video_writer.write(vis)

                if show:
                    cv2.imshow("Landing Pose", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if frame_idx % 100 == 0 or not is_video:
                print(f"[{file_path.name}:{frame_idx}] {status}")

    if video_writer is not None:
        video_writer.release()

    if show:
        cv2.destroyAllWindows()

    # 保存位姿日志
    log_path = output_dir / "pose_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(pose_log, f, indent=2, ensure_ascii=False)
    print(f"\n位姿日志: {log_path}")
    print(f"可视化结果: {output_dir}")

    # 统计
    valid = [e for e in pose_log if "reproj_error" in e]
    if valid:
        errors = [e["reproj_error"] for e in valid]
        print(f"\n--- 统计 ({len(valid)}/{len(pose_log)} 帧解算成功) ---")
        print(f"  重投影误差  mean={np.mean(errors):.2f}  "
              f"median={np.median(errors):.2f}  max={np.max(errors):.2f} px")


def main():
    parser = argparse.ArgumentParser(description="Landing Pose Inference Pipeline")
    parser.add_argument("--weights", type=str, required=True,
                        help="YOLOv8-pose 权重路径")
    parser.add_argument("--source", type=str, required=True,
                        help="图片/文件夹/视频路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="检测置信度阈值")
    parser.add_argument("--kpt-conf", type=float, default=0.5,
                        help="关键点置信度阈值 (PnP 只使用高于此值的关键点)")
    parser.add_argument("--pnp", type=str, default="epnp",
                        choices=["epnp", "ippe", "iterative", "ransac", "sqpnp"],
                        help="PnP 算法")
    parser.add_argument("--show", action="store_true",
                        help="实时显示")
    parser.add_argument("--save-video", action="store_true",
                        help="保存可视化视频")
    parser.add_argument("--no-vis", action="store_true",
                        help="不保存可视化图片")
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source=args.source,
        conf_thresh=args.conf,
        kpt_conf_thresh=args.kpt_conf,
        pnp_method=args.pnp,
        save_vis=not args.no_vis,
        save_video=args.save_video,
        show=args.show,
    )


if __name__ == "__main__":
    main()
