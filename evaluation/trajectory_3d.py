"""
轨迹解算与交互式 3D 可视化

从两段轨迹的图像中按固定间隔抽帧，使用 YOLO + PnP 解算相机位姿，
并与 GT 轨迹一起在交互式 3D 场景中呈现。

用法:
  python evaluation/trajectory_3d.py
  python evaluation/trajectory_3d.py --step 10 --pnp ippe
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_gt_trajectory(datasets_root: Path):
    """构建 GT 相机位姿序列（每帧），返回按序列分组的列表"""
    from evaluation.common_metrics import (
        CAM_LOCAL_T, CAM_LOCAL_EULER_XYZ,
        euler_xyz_to_matrix, load_sequence_specs, load_trajectory,
    )

    cam_local_rot = euler_xyz_to_matrix(*CAM_LOCAL_EULER_XYZ)
    sequences = {}

    for seq_name, traj_path in load_sequence_specs(datasets_root):
        traj_rows = load_trajectory(traj_path)
        frames = []
        for frame_idx, row in enumerate(traj_rows, start=1):
            rocket_t = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
            rocket_r = euler_xyz_to_matrix(row["roll"], row["pitch"], row["yaw"])
            cam_pos = rocket_t + rocket_r @ CAM_LOCAL_T
            frames.append({
                "frame": frame_idx,
                "x": float(cam_pos[0]),
                "y": float(cam_pos[1]),
                "z": float(cam_pos[2]),
                "height_z": float(row["z"]),
            })
        sequences[seq_name] = frames
    return sequences


def resolve_rgb(datasets_root: Path, seq_name: str, frame_idx: int) -> Path:
    """获取帧的实际图片路径"""
    for sub in ("rgb", "images"):
        p = datasets_root / seq_name / sub / f"{frame_idx:04d}.png"
        if p.exists():
            return p
    return datasets_root / seq_name / "rgb" / f"{frame_idx:04d}.png"


def run_trajectory_inference(args):
    from ultralytics import YOLO
    from yolo_landing.pose_solver import LandingPoseSolver

    datasets_root = Path(args.datasets_root).resolve()
    weights_path = Path(args.weights).resolve()
    step = args.step
    pnp_method = args.pnp

    model = YOLO(str(weights_path))
    solver = LandingPoseSolver(confidence_threshold=0.3)

    gt_sequences = build_gt_trajectory(datasets_root)
    pred_sequences = {}

    for seq_name, gt_frames in gt_sequences.items():
        print(f"\n处理序列: {seq_name} ({len(gt_frames)} 帧, 每 {step} 帧取 1 帧)")
        preds = []
        sampled = list(range(0, len(gt_frames), step))
        for count, idx in enumerate(sampled):
            frame = gt_frames[idx]
            frame_idx = frame["frame"]
            img_path = resolve_rgb(datasets_root, seq_name, frame_idx)
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            result = model.predict(img, conf=0.25, verbose=False, device=args.device)
            if len(result) == 0 or result[0].keypoints is None:
                continue

            r = result[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            best = int(r.boxes.conf.argmax().item())
            kpts = r.keypoints.data[best].detach().cpu().numpy()
            pred_xy = kpts[:, :2]
            pred_conf = kpts[:, 2]

            pose = solver.solve(pred_xy, pred_conf, method=pnp_method)
            if pose is None:
                continue

            R_wc, t_wc = solver.get_camera_pose_in_world(pose)
            preds.append({
                "frame": frame_idx,
                "x": float(t_wc[0]),
                "y": float(t_wc[1]),
                "z": float(t_wc[2]),
                "reproj_error": float(pose["reproj_error"]),
            })

            if (count + 1) % 50 == 0:
                print(f"  已处理 {count+1}/{len(sampled)} 帧")

        pred_sequences[seq_name] = preds
        print(f"  完成: {len(preds)}/{len(sampled)} 帧成功解算")

    return gt_sequences, pred_sequences


def create_3d_visualization(gt_sequences, pred_sequences, output_path: Path):
    """创建交互式 3D HTML 可视化"""
    import plotly.graph_objects as go

    fig = go.Figure()

    colors_gt = {"rocket_render_01": "#1f77b4", "rocket_render_02": "#2ca02c"}
    colors_pred = {"rocket_render_01": "#ff7f0e", "rocket_render_02": "#d62728"}

    for seq_name, gt_frames in gt_sequences.items():
        gx = [f["x"] for f in gt_frames]
        gy = [f["y"] for f in gt_frames]
        gz = [f["z"] for f in gt_frames]
        fig.add_trace(go.Scatter3d(
            x=gx, y=gy, z=gz,
            mode="lines",
            name=f"GT {seq_name}",
            line=dict(color=colors_gt.get(seq_name, "blue"), width=3),
            hovertemplate="GT<br>x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>",
        ))

        # GT 起点和终点标记
        fig.add_trace(go.Scatter3d(
            x=[gx[0]], y=[gy[0]], z=[gz[0]],
            mode="markers+text",
            name=f"Start {seq_name}",
            marker=dict(size=6, color="green", symbol="diamond"),
            text=["Start"], textposition="top center",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[gx[-1]], y=[gy[-1]], z=[gz[-1]],
            mode="markers+text",
            name=f"End {seq_name}",
            marker=dict(size=6, color="red", symbol="diamond"),
            text=["End"], textposition="top center",
            showlegend=False,
        ))

    for seq_name, preds in pred_sequences.items():
        if not preds:
            continue
        px = [f["x"] for f in preds]
        py = [f["y"] for f in preds]
        pz = [f["z"] for f in preds]
        reproj = [f["reproj_error"] for f in preds]
        frames = [f["frame"] for f in preds]

        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode="lines+markers",
            name=f"Pred {seq_name}",
            line=dict(color=colors_pred.get(seq_name, "orange"), width=2, dash="dash"),
            marker=dict(size=2, color=reproj, colorscale="YlOrRd",
                        cmin=0, cmax=max(5, max(reproj) if reproj else 5),
                        colorbar=dict(title="Reproj (px)", len=0.5, y=0.25)
                        if seq_name == list(pred_sequences.keys())[0] else None),
            hovertemplate=(
                "Pred<br>frame=%{customdata[0]}<br>"
                "x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<br>"
                "reproj=%{customdata[1]:.2f}px<extra></extra>"
            ),
            customdata=list(zip(frames, reproj)),
        ))

    # 着陆标志位置
    from yolo_landing.pose_solver import WORLD_KEYPOINTS_3D
    kp3d = WORLD_KEYPOINTS_3D
    fig.add_trace(go.Scatter3d(
        x=kp3d[:, 0], y=kp3d[:, 1], z=kp3d[:, 2],
        mode="markers",
        name="Landing Marker",
        marker=dict(size=5, color="black", symbol="cross"),
    ))

    fig.update_layout(
        title=dict(text="Rocket Landing Trajectory — Predicted vs Ground Truth",
                   font=dict(size=18)),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        width=1200, height=800,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"\n交互式 3D 可视化已保存: {output_path}")
    print("在浏览器中打开 HTML 文件即可旋转/缩放/平移查看")


def main():
    parser = argparse.ArgumentParser(description="轨迹解算与 3D 可视化")
    parser.add_argument("--weights", type=str,
                        default=str(ROOT / "yolo_landing" / "runs" / "landing_pose" / "weights" / "best.pt"))
    parser.add_argument("--datasets-root", type=str,
                        default=str(ROOT / "datasets"))
    parser.add_argument("--step", type=int, default=10,
                        help="采样间隔 (50Hz 下 step=10 即 5Hz)")
    parser.add_argument("--pnp", type=str, default="epnp",
                        choices=["epnp", "ippe", "iterative", "ransac", "sqpnp"])
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "visualization" / "trajectory_3d.html"))
    args = parser.parse_args()

    gt_seqs, pred_seqs = run_trajectory_inference(args)
    create_3d_visualization(gt_seqs, pred_seqs, Path(args.output))


if __name__ == "__main__":
    main()
