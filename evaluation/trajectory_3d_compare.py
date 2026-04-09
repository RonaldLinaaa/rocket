"""
轨迹对比: 纯 PnP vs 椭圆混合方法 — 交互式 3D 可视化

两条轨迹 × 两种方法 + GT = 共 6 条曲线在一个 3D 场景中。
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


def build_gt(datasets_root):
    from evaluation.common_metrics import (
        CAM_LOCAL_T, CAM_LOCAL_EULER_XYZ,
        euler_xyz_to_matrix, load_sequence_specs, load_trajectory,
    )
    cam_local_rot = euler_xyz_to_matrix(*CAM_LOCAL_EULER_XYZ)
    seqs = {}
    for seq_name, traj_path in load_sequence_specs(datasets_root):
        rows = load_trajectory(traj_path)
        frames = []
        for i, row in enumerate(rows, 1):
            rt = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
            rr = euler_xyz_to_matrix(row["roll"], row["pitch"], row["yaw"])
            cp = rt + rr @ CAM_LOCAL_T
            frames.append({"frame": i, "x": float(cp[0]), "y": float(cp[1]), "z": float(cp[2])})
        seqs[seq_name] = frames
    return seqs


def resolve_rgb(ds, seq, idx):
    for sub in ("rgb", "images"):
        p = ds / seq / sub / f"{idx:04d}.png"
        if p.exists():
            return p
    return ds / seq / "rgb" / f"{idx:04d}.png"


def run(args):
    from ultralytics import YOLO
    from yolo_landing.pose_solver import LandingPoseSolver
    from yolo_landing.ellipse_pose_solver import EllipsePoseSolver

    ds = Path(args.datasets_root).resolve()
    model = YOLO(str(Path(args.weights).resolve()))
    pnp_solver = LandingPoseSolver(confidence_threshold=0.3)
    ell_solver = EllipsePoseSolver(confidence_threshold=0.3)

    gt_seqs = build_gt(ds)
    pnp_seqs, ell_seqs = {}, {}

    for seq_name, gt_frames in gt_seqs.items():
        sampled = list(range(0, len(gt_frames), args.step))
        print(f"\n序列 {seq_name}: {len(gt_frames)} 帧, 采样 {len(sampled)} 帧 (step={args.step})")

        pnp_preds, ell_preds = [], []
        for count, idx in enumerate(sampled):
            frame = gt_frames[idx]
            fi = frame["frame"]
            img = cv2.imread(str(resolve_rgb(ds, seq_name, fi)))
            if img is None:
                continue

            res = model.predict(img, conf=0.25, verbose=False, device=args.device)
            if not res or res[0].keypoints is None or res[0].boxes is None or len(res[0].boxes) == 0:
                continue

            best = int(res[0].boxes.conf.argmax().item())
            kpts = res[0].keypoints.data[best].detach().cpu().numpy()
            xy, conf = kpts[:, :2], kpts[:, 2]
            bbox = res[0].boxes.xyxy[best].detach().cpu().numpy()

            # 纯 PnP
            pr = pnp_solver.solve(xy, conf, method=args.pnp)
            if pr is not None:
                Rw, tw = pnp_solver.get_camera_pose_in_world(pr)
                pnp_preds.append({"frame": fi, "x": float(tw[0]), "y": float(tw[1]),
                                  "z": float(tw[2]), "reproj": float(pr["reproj_error"])})

            # 椭圆混合
            er = ell_solver.solve(img, xy, conf, bbox=bbox, method=args.pnp)
            if er is not None:
                Rw2, tw2 = ell_solver.get_camera_pose_in_world(er)
                ell_preds.append({"frame": fi, "x": float(tw2[0]), "y": float(tw2[1]),
                                  "z": float(tw2[2]), "reproj": float(er["reproj_error"]),
                                  "strategy": er.get("strategy", ""),
                                  "d_ell": er.get("d_ellipse")})

            if (count + 1) % 50 == 0:
                print(f"  {count+1}/{len(sampled)}")

        pnp_seqs[seq_name] = pnp_preds
        ell_seqs[seq_name] = ell_preds
        print(f"  PnP: {len(pnp_preds)}/{len(sampled)}, Ellipse: {len(ell_preds)}/{len(sampled)}")

    return gt_seqs, pnp_seqs, ell_seqs


def make_html(gt_seqs, pnp_seqs, ell_seqs, output_path):
    import plotly.graph_objects as go
    from yolo_landing.pose_solver import WORLD_KEYPOINTS_3D

    fig = go.Figure()

    gt_colors = {"rocket_render_01": "#636EFA", "rocket_render_02": "#00CC96"}
    pnp_colors = {"rocket_render_01": "#EF553B", "rocket_render_02": "#FFA15A"}
    ell_colors = {"rocket_render_01": "#AB63FA", "rocket_render_02": "#19D3F3"}

    for seq, frames in gt_seqs.items():
        gx = [f["x"] for f in frames]
        gy = [f["y"] for f in frames]
        gz = [f["z"] for f in frames]
        fig.add_trace(go.Scatter3d(
            x=gx, y=gy, z=gz, mode="lines", name=f"GT {seq.replace('rocket_render_','Seq')}",
            line=dict(color=gt_colors.get(seq, "gray"), width=4),
            hovertemplate="GT<br>x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter3d(
            x=[gx[0]], y=[gy[0]], z=[gz[0]], mode="markers+text",
            marker=dict(size=5, color="green", symbol="diamond"),
            text=["Start"], textposition="top center", showlegend=False))
        fig.add_trace(go.Scatter3d(
            x=[gx[-1]], y=[gy[-1]], z=[gz[-1]], mode="markers+text",
            marker=dict(size=5, color="red", symbol="diamond"),
            text=["Land"], textposition="top center", showlegend=False))

    for seq, preds in pnp_seqs.items():
        if not preds:
            continue
        fig.add_trace(go.Scatter3d(
            x=[p["x"] for p in preds], y=[p["y"] for p in preds], z=[p["z"] for p in preds],
            mode="lines+markers", name=f"PnP {seq.replace('rocket_render_','Seq')}",
            line=dict(color=pnp_colors.get(seq, "red"), width=2, dash="dash"),
            marker=dict(size=2, color=[p["reproj"] for p in preds],
                        colorscale="YlOrRd", cmin=0, cmax=5),
            hovertemplate="PnP<br>f=%{customdata[0]}<br>x=%{x:.1f}<br>y=%{y:.1f}"
                          "<br>z=%{z:.1f}<br>reproj=%{customdata[1]:.2f}px<extra></extra>",
            customdata=[[p["frame"], p["reproj"]] for p in preds]))

    for seq, preds in ell_seqs.items():
        if not preds:
            continue
        fig.add_trace(go.Scatter3d(
            x=[p["x"] for p in preds], y=[p["y"] for p in preds], z=[p["z"] for p in preds],
            mode="lines+markers", name=f"Ellipse {seq.replace('rocket_render_','Seq')}",
            line=dict(color=ell_colors.get(seq, "purple"), width=2, dash="dot"),
            marker=dict(size=2, color=[p["reproj"] for p in preds],
                        colorscale="Purples", cmin=0, cmax=5),
            hovertemplate="Ellipse<br>f=%{customdata[0]}<br>x=%{x:.1f}<br>y=%{y:.1f}"
                          "<br>z=%{z:.1f}<br>strategy=%{customdata[2]}"
                          "<br>d_ell=%{customdata[3]}<extra></extra>",
            customdata=[[p["frame"], p["reproj"], p.get("strategy",""), p.get("d_ell","")] for p in preds]))

    # 着陆标志
    kp = WORLD_KEYPOINTS_3D
    fig.add_trace(go.Scatter3d(
        x=kp[:, 0], y=kp[:, 1], z=kp[:, 2], mode="markers",
        name="Landing Pad", marker=dict(size=5, color="black", symbol="cross")))

    fig.update_layout(
        title=dict(text="Trajectory Comparison — PnP vs Ellipse-Hybrid vs GT", font=dict(size=18)),
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)", font=dict(size=11)),
        width=1400, height=900, margin=dict(l=0, r=0, t=50, b=0))

    # 添加误差统计注释
    stats_lines = []
    for seq in gt_seqs:
        gt_f = {f["frame"]: np.array([f["x"], f["y"], f["z"]]) for f in gt_seqs[seq]}
        for label, preds in [("PnP", pnp_seqs.get(seq, [])), ("Ellipse", ell_seqs.get(seq, []))]:
            errs = []
            for p in preds:
                gt_pos = gt_f.get(p["frame"])
                if gt_pos is not None:
                    errs.append(np.linalg.norm(np.array([p["x"], p["y"], p["z"]]) - gt_pos))
            if errs:
                arr = np.array(errs)
                sn = seq.replace("rocket_render_", "S")
                stats_lines.append(f"{sn} {label}: mean={arr.mean():.1f}m, med={np.median(arr):.1f}m")

    fig.add_annotation(
        text="<br>".join(stats_lines), xref="paper", yref="paper",
        x=0.99, y=0.01, showarrow=False,
        font=dict(size=10, family="monospace"),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="gray",
        align="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"\n交互式 3D 对比已保存: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(ROOT / "yolo_landing/runs/landing_pose/weights/best.pt"))
    p.add_argument("--datasets-root", default=str(ROOT / "datasets"))
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--pnp", default="epnp")
    p.add_argument("--device", default="0")
    p.add_argument("--output", default=str(ROOT / "visualization/trajectory_3d_compare.html"))
    args = p.parse_args()

    gt, pnp, ell = run(args)
    make_html(gt, pnp, ell, Path(args.output))


if __name__ == "__main__":
    main()
