"""
综合可视化：训练曲线对比 + 测试指标对比 + 轨迹复原（论文级静态图）

用法:
  python evaluation/visualize_all.py
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]

# ── 论文级全局样式 ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

YOLO_COLOR = "#2196F3"
DETR_COLOR = "#FF5722"
GT_COLOR = "#4CAF50"
NEAR_COLOR = "#4CAF50"
MID_COLOR = "#2196F3"
FAR_COLOR = "#F44336"

OUT_DIR = ROOT / "visualization"


# ═══════════════════════════════════════════════════════════════
#  第一部分：训练曲线对比
# ═══════════════════════════════════════════════════════════════

def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_yolo_csv(csv_path: Path) -> Dict[str, List[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    out: Dict[str, List[float]] = {}
    for k in rows[0].keys():
        vals = [_to_float(r.get(k, "")) for r in rows]
        if all(v is None for v in vals):
            continue
        out[k.strip()] = [np.nan if v is None else v for v in vals]
    return out


def parse_rtdetr_log(log_path: Path) -> Dict[str, List[float]]:
    lines = [ln.strip() for ln in log_path.read_text("utf-8").splitlines() if ln.strip()]
    rows = [json.loads(ln) for ln in lines]
    out: Dict[str, List[float]] = {}
    out["epoch"] = [float(r.get("epoch", i)) for i, r in enumerate(rows)]
    for key in rows[0].keys():
        if key in ("epoch", "n_parameters"):
            continue
        vals = []
        has_num = False
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
                has_num = True
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                vals.append(float(v[0]))
                has_num = True
            else:
                vals.append(np.nan)
        if has_num:
            out[key] = vals
    return out


def plot_training_comparison(yolo_csv: Path, rtdetr_log: Path, out_dir: Path):
    """生成 YOLO vs RT-DETR 训练曲线对比（2×2 大图）"""
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo = parse_yolo_csv(yolo_csv)
    detr = parse_rtdetr_log(rtdetr_log)

    yolo_epochs = yolo.get("epoch", list(range(len(next(iter(yolo.values()))))))
    detr_epochs = detr.get("epoch", list(range(len(next(iter(detr.values()))))))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) YOLO 训练损失
    ax = axes[0, 0]
    for key, label in [
        ("train/pose_loss", "Pose Loss"),
        ("train/box_loss", "Box Loss"),
        ("train/kobj_loss", "KObj Loss"),
        ("train/cls_loss", "Cls Loss"),
    ]:
        if key in yolo:
            ax.plot(yolo_epochs, yolo[key], label=label, linewidth=1.5)
    ax.set_title("(a) YOLOv8-pose Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_yscale("log")

    # (b) RT-DETR 训练损失（主要分项）
    ax = axes[0, 1]
    for key, label in [
        ("train_loss_keypoints", "Keypoint L1"),
        ("train_loss_oks", "OKS"),
        ("train_loss_struct", "Structure"),
        ("train_loss_vfl", "VFL (cls)"),
        ("train_loss_bbox", "BBox"),
    ]:
        if key in detr:
            ax.plot(detr_epochs, detr[key], label=label, linewidth=1.5)
    ax.set_title("(b) RT-DETRv2 Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_yscale("log")

    # (c) YOLO 验证指标
    ax = axes[1, 0]
    for key, label in [
        ("val/pose_loss", "Val Pose Loss"),
        ("val/box_loss", "Val Box Loss"),
    ]:
        if key in yolo:
            ax.plot(yolo_epochs, yolo[key], label=label, linewidth=1.5)
    ax2 = ax.twinx()
    if "metrics/mAP50-95(P)" in yolo:
        ax2.plot(yolo_epochs, yolo["metrics/mAP50-95(P)"],
                 label="mAP50-95 (Pose)", color="#E91E63", linewidth=1.5, linestyle="--")
        ax2.set_ylabel("mAP", color="#E91E63")
        ax2.legend(loc="center right")
    ax.set_title("(c) YOLOv8-pose Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")

    # (d) RT-DETR 验证指标
    ax = axes[1, 1]
    coco_eval_key = "test_coco_eval_bbox"
    if coco_eval_key in detr:
        ax.plot(detr_epochs, detr[coco_eval_key], label="AP@0.5:0.95",
                color=DETR_COLOR, linewidth=2)
    # 画 OKS loss 在右轴
    if "train_loss_oks" in detr:
        ax2 = ax.twinx()
        ax2.plot(detr_epochs, detr["train_loss_oks"],
                 label="OKS Loss", color="#9C27B0", linewidth=1.5, linestyle="--")
        ax2.set_ylabel("OKS Loss", color="#9C27B0")
        ax2.legend(loc="center right")
    ax.set_title("(d) RT-DETRv2 Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("COCO AP")
    ax.legend(loc="lower right")

    fig.suptitle("Training Curves Comparison: YOLOv8-pose vs RT-DETRv2", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "training_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [训练对比] {out_path}")

    # 单独保存各自的训练曲线
    _plot_yolo_training_detail(yolo, yolo_epochs, out_dir)
    _plot_detr_training_detail(detr, detr_epochs, out_dir)


def _plot_yolo_training_detail(yolo, epochs, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    train_keys = [k for k in yolo if "train/" in k and "lr" not in k]
    val_keys = [k for k in yolo if "val/" in k]
    metric_keys = [k for k in yolo if "metrics/" in k]

    for k in train_keys[:5]:
        axes[0].plot(epochs, yolo[k], label=k.replace("train/", ""), linewidth=1.2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8); axes[0].set_yscale("log")

    for k in val_keys[:5]:
        axes[1].plot(epochs, yolo[k], label=k.replace("val/", ""), linewidth=1.2)
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8); axes[1].set_yscale("log")

    for k in metric_keys:
        axes[2].plot(epochs, yolo[k], label=k.replace("metrics/", ""), linewidth=1.2)
    axes[2].set_title("Metrics")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Value")
    axes[2].legend(fontsize=7)

    fig.suptitle("YOLOv8-pose Training Details", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "yolo_training_detail.png")
    plt.close(fig)


def _plot_detr_training_detail(detr, epochs, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    main_keys = ["train_loss_keypoints", "train_loss_oks", "train_loss_struct",
                 "train_loss_vis", "train_loss_vfl", "train_loss_bbox", "train_loss_giou"]
    for k in main_keys:
        if k in detr:
            axes[0].plot(epochs, detr[k], label=k.replace("train_loss_", ""), linewidth=1.2)
    axes[0].set_title("Main Loss Components")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7); axes[0].set_yscale("log")

    if "train_loss" in detr:
        axes[1].plot(epochs, detr["train_loss"], color=DETR_COLOR, linewidth=2)
    axes[1].set_title("Total Training Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")

    if "test_coco_eval_bbox" in detr:
        axes[2].plot(epochs, detr["test_coco_eval_bbox"], color=DETR_COLOR, linewidth=2)
    axes[2].set_title("COCO AP@0.5:0.95 (bbox)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AP")

    fig.suptitle("RT-DETRv2 Training Details", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "rtdetr_training_detail.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  第二部分：测试指标对比
# ═══════════════════════════════════════════════════════════════

def _load_metrics(json_path: Path) -> dict:
    return json.loads(json_path.read_text("utf-8"))


RANGE_LABELS = ["near", "mid", "far"]

def _compute_range_from_records(records: list, ann_path: Path) -> dict:
    """当 per_range 为空时，从 COCO annotations 中恢复 range_label 并计算"""
    ann_data = json.loads(ann_path.read_text("utf-8"))
    img_range = {}
    for img in ann_data.get("images", []):
        img_range[img["file_name"]] = img.get("range_label", "unknown")

    for r in records:
        if "range_label" not in r or r.get("range_label") == "unknown":
            r["range_label"] = img_range.get(r.get("file", ""), "unknown")

    grouped: Dict[str, list] = {}
    for r in records:
        rl = r.get("range_label", "unknown")
        grouped.setdefault(rl, []).append(r)

    per_range: Dict[str, dict] = {}
    for rl in RANGE_LABELS:
        recs = grouped.get(rl, [])
        if not recs:
            continue
        kp_err = np.array([e for r2 in recs for e in r2.get("kp_errors_px", [])], dtype=np.float64)
        pose_ok = [r2 for r2 in recs if r2.get("pose_success", False)]
        trans_err = np.array([r2["translation_error"] for r2 in pose_ok], dtype=np.float64)
        rot_err = np.array([r2["rotation_error_deg"] for r2 in pose_ok], dtype=np.float64)
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


def plot_test_comparison(yolo_json: Path, detr_json: Path, ann_path: Path, out_dir: Path):
    """生成测试指标对比图（总体 + 分段）"""
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo = _load_metrics(yolo_json)
    detr = _load_metrics(detr_json)

    ym = yolo["metrics"]
    dm = detr["metrics"]

    # (1) 总体指标对比柱状图
    _plot_overall_comparison(ym, dm, out_dir)

    # (2) 关键点误差分布对比（叠加直方图）
    _plot_kp_error_overlay(yolo["records"], detr["records"], out_dir)

    # (3) 位姿误差分布对比
    _plot_pose_error_overlay(yolo["records"], detr["records"], out_dir)

    # (4) 速度对比
    _plot_speed_comparison(ym, dm, out_dir)

    # (5) 近中远分段对比
    yolo_pr = yolo.get("per_range", {})
    detr_pr = detr.get("per_range", {})
    if not detr_pr and ann_path.exists():
        detr_pr = _compute_range_from_records(detr["records"], ann_path)
    _plot_range_comparison(yolo_pr, detr_pr, out_dir)


def _plot_overall_comparison(ym, dm, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 关键点指标
    ax = axes[0]
    labels = ["Mean Error\n(px)", "Median Error\n(px)", "PCK@5px", "PCK@10px"]
    y_vals = [ym["keypoint"]["mean_error_px"], ym["keypoint"]["median_error_px"],
              ym["keypoint"]["pck@5px"], ym["keypoint"]["pck@10px"]]
    d_vals = [dm["keypoint"]["mean_error_px"], dm["keypoint"]["median_error_px"],
              dm["keypoint"]["pck@5px"], dm["keypoint"]["pck@10px"]]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, y_vals, w, label="YOLOv8-pose", color=YOLO_COLOR, alpha=0.85)
    b2 = ax.bar(x + w/2, d_vals, w, label="RT-DETRv2", color=DETR_COLOR, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Keypoint Detection Metrics")
    ax.legend()
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            fmt = f"{h:.1f}" if h > 1 else f"{h:.3f}"
            ax.annotate(fmt, xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    # 位姿指标
    ax = axes[1]
    labels = ["Trans Mean", "Trans Median", "Rot Mean\n(deg)", "Rot Median\n(deg)"]
    y_vals = [ym["pose"]["translation_error_mean"], ym["pose"]["translation_error_median"],
              ym["pose"]["rotation_error_deg_mean"], ym["pose"]["rotation_error_deg_median"]]
    d_vals = [dm["pose"]["translation_error_mean"], dm["pose"]["translation_error_median"],
              dm["pose"]["rotation_error_deg_mean"], dm["pose"]["rotation_error_deg_median"]]
    x = np.arange(len(labels))
    b1 = ax.bar(x - w/2, y_vals, w, label="YOLOv8-pose", color=YOLO_COLOR, alpha=0.85)
    b2 = ax.bar(x + w/2, d_vals, w, label="RT-DETRv2", color=DETR_COLOR, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Pose Estimation Metrics")
    ax.legend()
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    # 速度指标
    ax = axes[2]
    labels = ["Det (ms)", "PnP (ms)", "FPS"]
    y_vals = [ym["speed"]["det_ms_mean"], ym["speed"]["pnp_ms_mean"], ym["speed"]["fps"]]
    d_vals = [dm["speed"]["det_ms_mean"], dm["speed"]["pnp_ms_mean"], dm["speed"]["fps"]]
    x = np.arange(len(labels))
    b1 = ax.bar(x - w/2, y_vals, w, label="YOLOv8-pose", color=YOLO_COLOR, alpha=0.85)
    b2 = ax.bar(x + w/2, d_vals, w, label="RT-DETRv2", color=DETR_COLOR, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Inference Speed")
    ax.legend()
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    fig.suptitle("Overall Test Metrics: YOLOv8-pose vs RT-DETRv2", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "overall_comparison.png")
    plt.close(fig)
    print(f"  [总体对比] {out_dir / 'overall_comparison.png'}")


def _plot_kp_error_overlay(yolo_records, detr_records, out_dir):
    yolo_errs = np.array([e for r in yolo_records for e in r.get("kp_errors_px", [])], dtype=np.float64)
    detr_errs = np.array([e for r in detr_records for e in r.get("kp_errors_px", [])], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 全范围
    ax = axes[0]
    bins = np.linspace(0, min(50, max(yolo_errs.max(), detr_errs.max())), 60)
    ax.hist(yolo_errs, bins=bins, alpha=0.6, label=f"YOLO (med={np.median(yolo_errs):.2f}px)",
            color=YOLO_COLOR, edgecolor="white", linewidth=0.5)
    ax.hist(detr_errs, bins=bins, alpha=0.6, label=f"RT-DETR (med={np.median(detr_errs):.2f}px)",
            color=DETR_COLOR, edgecolor="white", linewidth=0.5)
    ax.set_title("Keypoint Error Distribution (Full Range)")
    ax.set_xlabel("Pixel Error"); ax.set_ylabel("Count")
    ax.legend()

    # 放大 0-5px
    ax = axes[1]
    bins_zoom = np.linspace(0, 5, 50)
    ax.hist(yolo_errs[yolo_errs <= 5], bins=bins_zoom, alpha=0.6, label="YOLO", color=YOLO_COLOR,
            edgecolor="white", linewidth=0.5)
    ax.hist(detr_errs[detr_errs <= 5], bins=bins_zoom, alpha=0.6, label="RT-DETR", color=DETR_COLOR,
            edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, label="1px threshold")
    ax.set_title("Keypoint Error Distribution (0-5px Zoom)")
    ax.set_xlabel("Pixel Error"); ax.set_ylabel("Count")
    ax.legend()

    fig.suptitle("Keypoint Error Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "keypoint_error_comparison.png")
    plt.close(fig)
    print(f"  [关键点误差] {out_dir / 'keypoint_error_comparison.png'}")


def _plot_pose_error_overlay(yolo_records, detr_records, out_dir):
    def extract_pose(records):
        t = [r["translation_error"] for r in records if r.get("pose_success")]
        rot = [r["rotation_error_deg"] for r in records if r.get("pose_success")]
        return np.array(t), np.array(rot)

    yt, yr = extract_pose(yolo_records)
    dt, dr = extract_pose(detr_records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    clip_t = 500
    bins = np.linspace(0, clip_t, 60)
    ax.hist(np.clip(yt, 0, clip_t), bins=bins, alpha=0.6,
            label=f"YOLO (med={np.median(yt):.1f})", color=YOLO_COLOR, edgecolor="white", linewidth=0.5)
    ax.hist(np.clip(dt, 0, clip_t), bins=bins, alpha=0.6,
            label=f"RT-DETR (med={np.median(dt):.1f})", color=DETR_COLOR, edgecolor="white", linewidth=0.5)
    ax.set_title("Translation Error Distribution")
    ax.set_xlabel("Translation Error (Blender units)"); ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1]
    bins_r = np.linspace(0, min(20, max(yr.max(), dr.max())), 50)
    ax.hist(yr, bins=bins_r, alpha=0.6,
            label=f"YOLO (med={np.median(yr):.2f}°)", color=YOLO_COLOR, edgecolor="white", linewidth=0.5)
    ax.hist(dr, bins=bins_r, alpha=0.6,
            label=f"RT-DETR (med={np.median(dr):.2f}°)", color=DETR_COLOR, edgecolor="white", linewidth=0.5)
    ax.set_title("Rotation Error Distribution")
    ax.set_xlabel("Rotation Error (deg)"); ax.set_ylabel("Count")
    ax.legend()

    fig.suptitle("Pose Estimation Error Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "pose_error_comparison.png")
    plt.close(fig)
    print(f"  [位姿误差] {out_dir / 'pose_error_comparison.png'}")


def _plot_speed_comparison(ym, dm, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = ["YOLOv8-pose", "RT-DETRv2"]
    det_times = [ym["speed"]["det_ms_mean"], dm["speed"]["det_ms_mean"]]
    pnp_times = [ym["speed"]["pnp_ms_mean"], dm["speed"]["pnp_ms_mean"]]

    x = np.arange(len(methods))
    w = 0.5
    ax.bar(x, det_times, w, label="Detection", color=[YOLO_COLOR, DETR_COLOR], alpha=0.85)
    ax.bar(x, pnp_times, w, bottom=det_times, label="PnP", color=["#90CAF9", "#FFAB91"], alpha=0.85)

    for i, (d, p) in enumerate(zip(det_times, pnp_times)):
        ax.text(i, d + p + 0.3, f"Total: {d+p:.1f}ms\nFPS: {1000/(d+p):.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Inference Speed Breakdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "speed_comparison.png")
    plt.close(fig)
    print(f"  [速度对比] {out_dir / 'speed_comparison.png'}")


def _plot_range_comparison(yolo_pr, detr_pr, out_dir):
    """近中远分段对比（6 个子图）"""
    range_labels = ["near", "mid", "far"]
    range_display = {"near": "Near\n(<100m)", "mid": "Mid\n(100-300m)", "far": "Far\n(>300m)"}

    metrics_config = [
        ("kp_mean_error_px", "Keypoint Mean Error (px)", False),
        ("pck_at_5", "PCK@5px", True),
        ("pck_at_10", "PCK@10px", True),
        ("translation_error_mean", "Translation Error (mean)", False),
        ("rotation_error_deg_mean", "Rotation Error (deg)", False),
        ("detection_rate", "Detection Rate", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()

    for ax_idx, (mk, title, higher_better) in enumerate(metrics_config):
        ax = axes_flat[ax_idx]
        x_pos = np.arange(len(range_labels))
        w = 0.35
        yolo_vals, detr_vals = [], []
        for rl in range_labels:
            yv = yolo_pr.get(rl, {}).get(mk)
            dv = detr_pr.get(rl, {}).get(mk)
            yolo_vals.append(yv if yv is not None else 0)
            detr_vals.append(dv if dv is not None else 0)

        b1 = ax.bar(x_pos - w/2, yolo_vals, w, label="YOLO", color=YOLO_COLOR, alpha=0.85)
        b2 = ax.bar(x_pos + w/2, detr_vals, w, label="RT-DETR", color=DETR_COLOR, alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([range_display[rl] for rl in range_labels])
        ax.set_title(title)
        ax.legend(fontsize=8)

        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    fmt = f"{h:.3f}" if h < 10 else f"{h:.1f}"
                    ax.annotate(fmt, xy=(bar.get_x() + bar.get_width()/2, h),
                                xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7)

    fig.suptitle("Per-Altitude Comparison: YOLOv8-pose vs RT-DETRv2", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "range_comparison.png")
    plt.close(fig)
    print(f"  [近中远对比] {out_dir / 'range_comparison.png'}")


# ═══════════════════════════════════════════════════════════════
#  第三部分：轨迹复原论文级可视化（静态 matplotlib）
# ═══════════════════════════════════════════════════════════════

ROCKET_RADIUS = 1.85
ROCKET_LENGTH = 42.0
_CAM_LOCAL_T = np.array([ROCKET_RADIUS + 0.15, 0.0, ROCKET_LENGTH * 0.85], dtype=np.float64)
_CAM_LOCAL_EULER = np.array([math.radians(15.0), 0.0, math.radians(-90.0)], dtype=np.float64)


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def _euler_xyz_to_mat(x, y, z):
    return _rot_z(z) @ _rot_y(y) @ _rot_x(x)


def _load_trajectory_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(row[k]) for k in ("x","y","z","roll","pitch","yaw")})
    return rows


def _load_gt_trajectories(datasets_root: Path):
    """加载 GT 相机轨迹"""
    cam_local_rot = _euler_xyz_to_mat(*_CAM_LOCAL_EULER)
    seqs = {}

    for d in sorted(p for p in datasets_root.iterdir() if p.is_dir() and p.name.startswith("rocket_render")):
        csvs = sorted(d.glob("trajectory*.csv"))
        if not csvs:
            continue
        rows = _load_trajectory_csv(csvs[0])
        frames = []
        for i, row in enumerate(rows, 1):
            rt = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
            rr = _euler_xyz_to_mat(row["roll"], row["pitch"], row["yaw"])
            cp = rt + rr @ _CAM_LOCAL_T
            frames.append({"frame": i, "x": float(cp[0]), "y": float(cp[1]), "z": float(cp[2])})
        seqs[d.name] = frames
    return seqs


def _extract_pred_trajectory(records: list, datasets_root: Path, gt_pose_map: dict):
    """从测试记录中提取预测相机位置（利用 PnP 结果）"""
    seqs: Dict[str, list] = {}
    for r in records:
        if not r.get("pose_success"):
            continue
        fname = r["file"]
        parts = fname.split("/")
        if len(parts) >= 2:
            seq = parts[0]
        else:
            continue

        gt_pose = gt_pose_map.get(fname)
        if gt_pose is None:
            continue

        # 用 GT 来比对，但也可以重建预测轨迹
        # 此处我们同时保存 GT 和预测位置
        seqs.setdefault(seq, []).append({
            "file": fname,
            "frame": int(parts[-1].replace("rocket_render_", "").replace(".png", "")),
            "gt_pos": gt_pose["t_wc"],
            "translation_error": r.get("translation_error", 0),
            "rotation_error_deg": r.get("rotation_error_deg", 0),
        })

    for seq in seqs:
        seqs[seq].sort(key=lambda x: x["frame"])
    return seqs


def plot_trajectory_paper(datasets_root: Path, yolo_json: Path, detr_json: Path, out_dir: Path):
    """生成论文级轨迹可视化"""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_seqs = _load_gt_trajectories(datasets_root)

    yolo_data = _load_metrics(yolo_json)
    detr_data = _load_metrics(detr_json)

    # 将误差沿轨迹帧号可视化
    _plot_error_along_trajectory(yolo_data["records"], detr_data["records"], gt_seqs, out_dir)

    # 3D 轨迹静态图
    _plot_3d_trajectory_static(gt_seqs, out_dir)

    # 高度 vs 误差关系图
    _plot_altitude_vs_error(yolo_data["records"], detr_data["records"], out_dir)


def _plot_error_along_trajectory(yolo_recs, detr_recs, gt_seqs, out_dir):
    """帧号 vs 位姿误差：论文主图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for recs, method, color in [(yolo_recs, "YOLOv8-pose", YOLO_COLOR),
                                 (detr_recs, "RT-DETRv2", DETR_COLOR)]:
        by_seq: Dict[str, list] = {}
        for r in recs:
            if not r.get("pose_success"):
                continue
            parts = r["file"].split("/")
            seq = parts[0] if len(parts) >= 2 else "unknown"
            frame_num = int(parts[-1].replace("rocket_render_", "").replace(".png", ""))
            by_seq.setdefault(seq, []).append((frame_num, r))

        for seq in by_seq:
            by_seq[seq].sort(key=lambda x: x[0])

        col_idx = 0 if "YOLO" in method else 1

        # 平移误差
        ax = axes[0, col_idx]
        for seq, pairs in by_seq.items():
            frames = [p[0] for p in pairs]
            trans = [p[1]["translation_error"] for p in pairs]
            sn = seq.replace("rocket_render_", "Seq ")
            ax.plot(frames, trans, linewidth=1, alpha=0.8, label=sn)
        ax.set_title(f"{method} — Translation Error vs Frame")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Translation Error")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

        # 旋转误差
        ax = axes[1, col_idx]
        for seq, pairs in by_seq.items():
            frames = [p[0] for p in pairs]
            rot = [p[1]["rotation_error_deg"] for p in pairs]
            sn = seq.replace("rocket_render_", "Seq ")
            ax.plot(frames, rot, linewidth=1, alpha=0.8, label=sn)
        ax.set_title(f"{method} — Rotation Error vs Frame")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Rotation Error (deg)")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    fig.suptitle("Pose Error Along Trajectory", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "error_along_trajectory.png")
    plt.close(fig)
    print(f"  [轨迹误差] {out_dir / 'error_along_trajectory.png'}")


def _plot_3d_trajectory_static(gt_seqs, out_dir):
    """GT 轨迹的论文级 3D 渲染"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"rocket_render_01": "#1565C0", "rocket_render_02": "#2E7D32"}
    for seq, frames in gt_seqs.items():
        xs = [f["x"] for f in frames]
        ys = [f["y"] for f in frames]
        zs = [f["z"] for f in frames]
        sn = seq.replace("rocket_render_", "Trajectory ")
        c = colors.get(seq, "gray")
        ax.plot(xs, ys, zs, linewidth=2, color=c, label=sn)
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color="green", s=60, zorder=5, marker="^")
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="red", s=60, zorder=5, marker="v")

    ax.scatter([0], [0], [0], color="black", s=100, marker="*", label="Landing Pad", zorder=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Ground Truth Camera Trajectories", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    fig.savefig(out_dir / "gt_trajectory_3d.png")
    plt.close(fig)
    print(f"  [GT轨迹3D] {out_dir / 'gt_trajectory_3d.png'}")

    # 俯视图
    fig, ax = plt.subplots(figsize=(8, 6))
    for seq, frames in gt_seqs.items():
        xs = [f["x"] for f in frames]
        ys = [f["y"] for f in frames]
        sn = seq.replace("rocket_render_", "Traj ")
        c = colors.get(seq, "gray")
        ax.plot(xs, ys, linewidth=2, color=c, label=sn)
        ax.scatter([xs[0]], [ys[0]], color="green", s=60, zorder=5, marker="^")
        ax.scatter([xs[-1]], [ys[-1]], color="red", s=60, zorder=5, marker="v")
    ax.scatter([0], [0], color="black", s=100, marker="*", label="Landing Pad", zorder=10)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Camera Trajectory — Top View", fontsize=14, fontweight="bold")
    ax.legend(); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "gt_trajectory_topview.png")
    plt.close(fig)
    print(f"  [GT轨迹俯视] {out_dir / 'gt_trajectory_topview.png'}")


def _plot_altitude_vs_error(yolo_recs, detr_recs, out_dir):
    """高度 vs 误差散点图 — 展示远距离定位困难"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for recs, method, color, marker in [
        (yolo_recs, "YOLOv8-pose", YOLO_COLOR, "o"),
        (detr_recs, "RT-DETRv2", DETR_COLOR, "s"),
    ]:
        frames_data = []
        for r in recs:
            if not r.get("pose_success"):
                continue
            parts = r["file"].split("/")
            frame_num = int(parts[-1].replace("rocket_render_", "").replace(".png", ""))
            frames_data.append({
                "frame": frame_num,
                "trans_err": r["translation_error"],
                "rot_err": r["rotation_error_deg"],
                "kp_err": np.mean(r.get("kp_errors_px", [0])),
            })

        if not frames_data:
            continue

        # 用关键点误差作为高度代理（远处 kp 误差不同模式）
        kp_errs = np.array([d["kp_err"] for d in frames_data])
        trans_errs = np.array([d["trans_err"] for d in frames_data])
        rot_errs = np.array([d["rot_err"] for d in frames_data])
        frame_nums = np.array([d["frame"] for d in frames_data])

        # 平移误差 vs 帧号（帧号小=远，帧号大=近）
        ax = axes[0]
        ax.scatter(frame_nums, trans_errs, s=8, alpha=0.4, color=color,
                   marker=marker, label=method)

        # 旋转误差 vs 帧号
        ax = axes[1]
        ax.scatter(frame_nums, rot_errs, s=8, alpha=0.4, color=color,
                   marker=marker, label=method)

    axes[0].set_title("Translation Error vs Frame Index\n(Early frames = high altitude)")
    axes[0].set_xlabel("Frame Index"); axes[0].set_ylabel("Translation Error")
    axes[0].legend(); axes[0].set_ylim(0, 600)

    axes[1].set_title("Rotation Error vs Frame Index\n(Early frames = high altitude)")
    axes[1].set_xlabel("Frame Index"); axes[1].set_ylabel("Rotation Error (deg)")
    axes[1].legend(); axes[1].set_ylim(0, 20)

    fig.suptitle("Pose Error vs Descent Progress", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "altitude_vs_error.png")
    plt.close(fig)
    print(f"  [高度-误差] {out_dir / 'altitude_vs_error.png'}")


# ═══════════════════════════════════════════════════════════════
#  第四部分：综合摘要表格图
# ═══════════════════════════════════════════════════════════════

def plot_summary_table(yolo_json: Path, detr_json: Path, out_dir: Path):
    """生成论文级摘要对比表格图"""
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo = _load_metrics(yolo_json)
    detr = _load_metrics(detr_json)
    ym, dm = yolo["metrics"], detr["metrics"]

    rows = [
        ["Detection Rate", f"{ym['detection_rate']:.1%}", f"{dm['detection_rate']:.1%}"],
        ["Pose Success Rate", f"{ym['pose_success_rate']:.1%}", f"{dm['pose_success_rate']:.1%}"],
        ["KP Mean Error (px)", f"{ym['keypoint']['mean_error_px']:.2f}", f"{dm['keypoint']['mean_error_px']:.2f}"],
        ["KP Median Error (px)", f"{ym['keypoint']['median_error_px']:.2f}", f"{dm['keypoint']['median_error_px']:.2f}"],
        ["PCK@5px", f"{ym['keypoint']['pck@5px']:.1%}", f"{dm['keypoint']['pck@5px']:.1%}"],
        ["PCK@10px", f"{ym['keypoint']['pck@10px']:.1%}", f"{dm['keypoint']['pck@10px']:.1%}"],
        ["Trans Error Mean", f"{ym['pose']['translation_error_mean']:.1f}", f"{dm['pose']['translation_error_mean']:.1f}"],
        ["Trans Error Median", f"{ym['pose']['translation_error_median']:.1f}", f"{dm['pose']['translation_error_median']:.1f}"],
        ["Rot Error Mean (deg)", f"{ym['pose']['rotation_error_deg_mean']:.2f}", f"{dm['pose']['rotation_error_deg_mean']:.2f}"],
        ["Rot Error Median (deg)", f"{ym['pose']['rotation_error_deg_median']:.2f}", f"{dm['pose']['rotation_error_deg_median']:.2f}"],
        ["FPS", f"{ym['speed']['fps']:.1f}", f"{dm['speed']['fps']:.1f}"],
        ["Det Time (ms)", f"{ym['speed']['det_ms_mean']:.1f}", f"{dm['speed']['det_ms_mean']:.1f}"],
        ["Parameters", "11.6M", "43.5M"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    col_labels = ["Metric", "YOLOv8-pose", "RT-DETRv2"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for j in range(3):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        bg = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(3):
            table[i, j].set_facecolor(bg)

    ax.set_title("Comprehensive Evaluation Summary", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_table.png")
    plt.close(fig)
    print(f"  [摘要表格] {out_dir / 'summary_table.png'}")


# ═══════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  综合可视化生成")
    print("=" * 60)

    yolo_csv = ROOT / "yolo_landing" / "runs" / "landing_pose" / "results.csv"
    rtdetr_log = ROOT / "RT-DETR-landing" / "rtdetrv2_pytorch" / "output" / "rtdetrv2_r50vd_landing" / "log.txt"
    yolo_json = ROOT / "visualization" / "testing" / "yolo" / "metrics_summary.json"
    detr_json = ROOT / "visualization" / "testing" / "rtdetr" / "metrics_summary.json"
    ann_path = ROOT / "datasets" / "annotations_coco_test.json"
    datasets_root = ROOT / "datasets"

    training_dir = OUT_DIR / "training"
    testing_dir = OUT_DIR / "comparison"
    trajectory_dir = OUT_DIR / "trajectory"

    print("\n[1/4] 训练曲线对比...")
    if yolo_csv.exists() and rtdetr_log.exists():
        plot_training_comparison(yolo_csv, rtdetr_log, training_dir)
    else:
        print("  跳过：训练日志文件不存在")

    print("\n[2/4] 测试指标对比...")
    if yolo_json.exists() and detr_json.exists():
        plot_test_comparison(yolo_json, detr_json, ann_path, testing_dir)
    else:
        print("  跳过：测试结果文件不存在")

    print("\n[3/4] 轨迹可视化...")
    if yolo_json.exists() and detr_json.exists() and datasets_root.exists():
        plot_trajectory_paper(datasets_root, yolo_json, detr_json, trajectory_dir)
    else:
        print("  跳过：数据文件不存在")

    print("\n[4/4] 摘要表格...")
    if yolo_json.exists() and detr_json.exists():
        plot_summary_table(yolo_json, detr_json, OUT_DIR / "comparison")
    else:
        print("  跳过")

    print("\n" + "=" * 60)
    print(f"  所有可视化已保存到: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
