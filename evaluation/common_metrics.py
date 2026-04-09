import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMG_W = 1080
IMG_H = 720
ROCKET_RADIUS = 1.85
ROCKET_LENGTH = 42.0
CAM_LOCAL_T = np.array([ROCKET_RADIUS + 0.15, 0.0, ROCKET_LENGTH * 0.85], dtype=np.float64)
CAM_LOCAL_EULER_XYZ = np.array(
    [math.radians(15.0), 0.0, math.radians(-90.0)],
    dtype=np.float64,
)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_xyz_to_matrix(x: float, y: float, z: float) -> np.ndarray:
    # 与数据生成脚本保持一致：R = Rz @ Ry @ Rx
    return rot_z(z) @ rot_y(y) @ rot_x(x)


def load_trajectory(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "roll": float(row["roll"]),
                    "pitch": float(row["pitch"]),
                    "yaw": float(row["yaw"]),
                }
            )
    return rows


def load_sequence_specs(datasets_root: Path) -> List[Tuple[str, Path]]:
    cfg = datasets_root / "sequences.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        out: List[Tuple[str, Path]] = []
        for item in data.get("sequences", []):
            seq = item["dir"]
            traj = datasets_root / seq / item["trajectory"]
            if traj.exists():
                out.append((seq, traj))
        if out:
            return out

    # 兜底：自动发现 trajectory*.csv
    out = []
    for d in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
        csvs = sorted(d.glob("trajectory*.csv"))
        if csvs:
            out.append((d.name, csvs[0]))
    return out


# Blender 相机坐标系 → OpenCV 相机坐标系
# Blender: -Z 前方, Y 上; OpenCV: +Z 前方, Y 下 → 绕 X 旋转 180°
_BLENDER_TO_OPENCV = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)


def build_gt_camera_pose_map(datasets_root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    cam_local_rot = euler_xyz_to_matrix(*CAM_LOCAL_EULER_XYZ)
    pose_map: Dict[str, Dict[str, np.ndarray]] = {}
    for seq_name, traj_path in load_sequence_specs(datasets_root):
        traj_rows = load_trajectory(traj_path)
        for frame_idx, row in enumerate(traj_rows, start=1):
            rocket_t = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
            rocket_r = euler_xyz_to_matrix(row["roll"], row["pitch"], row["yaw"])
            cam_pos_world = rocket_t + rocket_r @ CAM_LOCAL_T
            # Blender 相机旋转 → OpenCV 约定
            cam_rot_world = rocket_r @ cam_local_rot @ _BLENDER_TO_OPENCV
            key = f"{seq_name}/rocket_render_{frame_idx:04d}.png"
            pose_map[key] = {"t_wc": cam_pos_world, "R_wc": cam_rot_world}
    return pose_map


def resolve_rgb_path(datasets_root: Path, file_name: str) -> Path:
    direct = datasets_root / file_name
    if direct.exists():
        return direct
    parts = file_name.split("/")
    if len(parts) == 2:
        seq, fname = parts
        frame = fname.replace("rocket_render_", "")
        for sub in ("rgb", "images"):
            alt = datasets_root / seq / sub / frame
            if alt.exists():
                return alt
    return direct


def load_coco_test(ann_path: Path) -> List[Dict]:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    anns_by_img = {ann["image_id"]: ann for ann in data["annotations"]}
    samples: List[Dict] = []
    for img in data["images"]:
        ann = anns_by_img.get(img["id"])
        if ann is None:
            continue
        samples.append({"image": img, "ann": ann})
    return samples


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    # 旋转误差使用 SO(3) 测地角
    r = R_pred @ R_gt.T
    trace = float(np.trace(r))
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_summary_metrics(records: List[Dict]) -> Dict:
    total = len(records)
    detected = [r for r in records if r.get("detected", False)]
    with_kp = [r for r in detected if len(r.get("kp_errors_px", [])) > 0]
    pose_ok = [r for r in records if r.get("pose_success", False)]

    kp_errors = np.array(
        [e for r in with_kp for e in r.get("kp_errors_px", [])],
        dtype=np.float64,
    )
    trans_errors = np.array([r["translation_error"] for r in pose_ok], dtype=np.float64)
    rot_errors = np.array([r["rotation_error_deg"] for r in pose_ok], dtype=np.float64)

    det_ms = np.array([r.get("det_time_ms", 0.0) for r in records], dtype=np.float64)
    pnp_ms = np.array([r.get("pnp_time_ms", 0.0) for r in records], dtype=np.float64)
    total_ms = det_ms + pnp_ms

    def safe_mean(arr: np.ndarray) -> Optional[float]:
        return None if arr.size == 0 else float(arr.mean())

    def safe_median(arr: np.ndarray) -> Optional[float]:
        return None if arr.size == 0 else float(np.median(arr))

    # 关键点评价参数（可用于论文/实验对比）
    metrics = {
        "num_samples": total,
        "detection_rate": float(len(detected) / total) if total > 0 else 0.0,
        "pose_success_rate": float(len(pose_ok) / total) if total > 0 else 0.0,
        "keypoint": {
            "mean_error_px": safe_mean(kp_errors),
            "median_error_px": safe_median(kp_errors),
            "pck@5px": None if kp_errors.size == 0 else float((kp_errors <= 5.0).mean()),
            "pck@10px": None if kp_errors.size == 0 else float((kp_errors <= 10.0).mean()),
        },
        "pose": {
            "translation_error_mean": safe_mean(trans_errors),
            "translation_error_median": safe_median(trans_errors),
            "rotation_error_deg_mean": safe_mean(rot_errors),
            "rotation_error_deg_median": safe_median(rot_errors),
        },
        "speed": {
            "det_ms_mean": safe_mean(det_ms),
            "pnp_ms_mean": safe_mean(pnp_ms),
            "total_ms_mean": safe_mean(total_ms),
            "fps": None if total_ms.size == 0 else float(1000.0 / np.mean(total_ms)),
        },
    }
    return metrics


def _save_histogram(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 40) -> None:
    if values.size == 0:
        return
    plt.figure(figsize=(8, 4.5))
    plt.hist(values, bins=bins, color="#4C78A8", alpha=0.9, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_speed_bar(metrics: Dict, out_path: Path) -> None:
    labels = ["Det", "PnP", "Total"]
    vals = [
        metrics["speed"]["det_ms_mean"] or 0.0,
        metrics["speed"]["pnp_ms_mean"] or 0.0,
        metrics["speed"]["total_ms_mean"] or 0.0,
    ]
    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(labels, vals, color=["#59A14F", "#F28E2B", "#E15759"])
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom")
    plt.title("Inference Speed Breakdown")
    plt.ylabel("Time (ms)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


RANGE_LABELS = ["near", "mid", "far"]
RANGE_COLORS = {"near": "#59A14F", "mid": "#4C78A8", "far": "#E15759"}


def _save_range_grouped_bar(per_range: Dict[str, Dict], metric_key: str,
                            title: str, ylabel: str, out_path: Path) -> None:
    labels, vals, colors = [], [], []
    for rl in RANGE_LABELS:
        m = per_range.get(rl)
        if m is None:
            continue
        v = m.get(metric_key)
        if v is None:
            continue
        labels.append(rl)
        vals.append(v)
        colors.append(RANGE_COLORS[rl])
    if not vals:
        return
    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(labels, vals, color=colors)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_range_comparison(per_range: Dict[str, Dict], method_name: str,
                           output_dir: Path) -> None:
    """per_range: {range_label: metrics_dict}"""
    _save_range_grouped_bar(
        per_range, "kp_mean_error_px",
        f"{method_name} Keypoint Error by Altitude",
        "Mean Error (px)", output_dir / "range_keypoint_error.png")
    _save_range_grouped_bar(
        per_range, "translation_error_mean",
        f"{method_name} Translation Error by Altitude",
        "Mean Translation Error", output_dir / "range_translation_error.png")
    _save_range_grouped_bar(
        per_range, "rotation_error_deg_mean",
        f"{method_name} Rotation Error by Altitude",
        "Mean Rotation Error (deg)", output_dir / "range_rotation_error.png")
    _save_range_grouped_bar(
        per_range, "pose_success_rate",
        f"{method_name} Pose Success Rate by Altitude",
        "Success Rate", output_dir / "range_pose_success.png")
    _save_range_grouped_bar(
        per_range, "pck_at_5",
        f"{method_name} PCK@5px by Altitude",
        "PCK@5px", output_dir / "range_pck5.png")


def _compute_range_metrics(records: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for r in records:
        rl = r.get("range_label", "unknown")
        grouped.setdefault(rl, []).append(r)
    per_range: Dict[str, Dict] = {}
    for rl in RANGE_LABELS:
        recs = grouped.get(rl, [])
        if not recs:
            continue
        kp_err = np.array(
            [e for r2 in recs for e in r2.get("kp_errors_px", [])],
            dtype=np.float64)
        pose_ok = [r2 for r2 in recs if r2.get("pose_success", False)]
        trans_err = np.array(
            [r2["translation_error"] for r2 in pose_ok], dtype=np.float64)
        rot_err = np.array(
            [r2["rotation_error_deg"] for r2 in pose_ok], dtype=np.float64)
        per_range[rl] = {
            "num_samples": len(recs),
            "detection_rate": sum(1 for r2 in recs if r2.get("detected")) / max(len(recs), 1),
            "pose_success_rate": len(pose_ok) / max(len(recs), 1),
            "kp_mean_error_px": float(kp_err.mean()) if kp_err.size else None,
            "kp_median_error_px": float(np.median(kp_err)) if kp_err.size else None,
            "pck_at_5": float((kp_err <= 5).mean()) if kp_err.size else None,
            "pck_at_10": float((kp_err <= 10).mean()) if kp_err.size else None,
            "translation_error_mean": float(trans_err.mean()) if trans_err.size else None,
            "rotation_error_deg_mean": float(rot_err.mean()) if rot_err.size else None,
        }
    return per_range


def dump_report_and_figures(method_name: str, records: List[Dict], output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_summary_metrics(records)

    per_range = _compute_range_metrics(records)

    report = {
        "method": method_name,
        "metrics": metrics,
        "per_range": per_range,
        "records": records,
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    kp_err = np.array([e for r in records for e in r.get("kp_errors_px", [])], dtype=np.float64)
    trans_err = np.array([r["translation_error"] for r in records if r.get("pose_success", False)], dtype=np.float64)
    rot_err = np.array([r["rotation_error_deg"] for r in records if r.get("pose_success", False)], dtype=np.float64)

    _save_histogram(
        kp_err,
        title=f"{method_name} Keypoint Error Distribution",
        xlabel="Pixel Error (px)",
        out_path=output_dir / "keypoint_error_hist.png",
    )
    _save_histogram(
        trans_err,
        title=f"{method_name} Translation Error Distribution",
        xlabel="Translation Error (same unit as dataset)",
        out_path=output_dir / "translation_error_hist.png",
    )
    _save_histogram(
        rot_err,
        title=f"{method_name} Rotation Error Distribution",
        xlabel="Rotation Error (deg)",
        out_path=output_dir / "rotation_error_hist.png",
    )
    _save_speed_bar(metrics, output_dir / "speed_breakdown.png")
    _save_range_comparison(per_range, method_name, output_dir)
    return report


def print_metrics_to_console(report: Dict) -> None:
    m = report["metrics"]
    print("\n========== Evaluation Summary ==========")
    print(f"Method: {report['method']}")
    print(f"Samples: {m['num_samples']}")
    print(f"Detection Rate: {m['detection_rate']:.4f}")
    print(f"Pose Success Rate: {m['pose_success_rate']:.4f}")
    print(
        "Keypoint: "
        f"mean={m['keypoint']['mean_error_px']} px, "
        f"median={m['keypoint']['median_error_px']} px, "
        f"PCK@5={m['keypoint']['pck@5px']}, "
        f"PCK@10={m['keypoint']['pck@10px']}"
    )
    print(
        "Pose: "
        f"trans_mean={m['pose']['translation_error_mean']}, "
        f"trans_median={m['pose']['translation_error_median']}, "
        f"rot_mean={m['pose']['rotation_error_deg_mean']}, "
        f"rot_median={m['pose']['rotation_error_deg_median']}"
    )
    print(
        "Speed: "
        f"det={m['speed']['det_ms_mean']} ms, "
        f"pnp={m['speed']['pnp_ms_mean']} ms, "
        f"total={m['speed']['total_ms_mean']} ms, "
        f"fps={m['speed']['fps']}"
    )
    per_range = report.get("per_range", {})
    if per_range:
        print("\n---------- Per-Altitude Metrics ----------")
        for rl in RANGE_LABELS:
            rm = per_range.get(rl)
            if rm is None:
                continue
            print(f"  [{rl}] n={rm['num_samples']}, det={rm['detection_rate']:.4f}, "
                  f"pose_ok={rm['pose_success_rate']:.4f}, "
                  f"kp_mean={rm.get('kp_mean_error_px')}, "
                  f"PCK@5={rm.get('pck_at_5')}, "
                  f"trans_err={rm.get('translation_error_mean')}, "
                  f"rot_err={rm.get('rotation_error_deg_mean')}")
