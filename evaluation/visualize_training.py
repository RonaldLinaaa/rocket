import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_yolo_results(csv_path: Path) -> Dict[str, List[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    out: Dict[str, List[float]] = {}
    for k in rows[0].keys():
        vals = [_to_float(r.get(k, "")) for r in rows]
        if all(v is None for v in vals):
            continue
        out[k] = [np.nan if v is None else v for v in vals]
    return out


def parse_rtdetr_log(log_path: Path) -> Dict[str, List[float]]:
    lines = [ln.strip() for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [json.loads(ln) for ln in lines]
    if not rows:
        return {}

    out: Dict[str, List[float]] = {}
    epochs = []
    for row in rows:
        epochs.append(float(row.get("epoch", len(epochs))))
    out["epoch"] = epochs

    for key in rows[0].keys():
        if key == "epoch":
            continue
        vals = []
        has_numeric = False
        for row in rows:
            val = row.get(key, None)
            if isinstance(val, (int, float)):
                vals.append(float(val))
                has_numeric = True
            elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], (int, float)):
                vals.append(float(val[0]))
                has_numeric = True
            else:
                vals.append(np.nan)
        if has_numeric:
            out[key] = vals
    return out


def _plot_yolo(data: Dict[str, List[float]], out_path: Path) -> None:
    x = data.get("epoch", list(range(len(next(iter(data.values()))))))
    train_keys = [k for k in data.keys() if "train/" in k and "lr" not in k][:4]
    val_keys = [k for k in data.keys() if "val/" in k][:4]
    metric_keys = [k for k in data.keys() if "metrics/" in k][:4]

    plt.figure(figsize=(14, 9))

    plt.subplot(2, 2, 1)
    for k in train_keys:
        plt.plot(x, data[k], label=k)
    plt.title("YOLO Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(alpha=0.25)
    if train_keys:
        plt.legend(fontsize=8)

    plt.subplot(2, 2, 2)
    for k in val_keys:
        plt.plot(x, data[k], label=k)
    plt.title("YOLO Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(alpha=0.25)
    if val_keys:
        plt.legend(fontsize=8)

    plt.subplot(2, 2, 3)
    for k in metric_keys:
        plt.plot(x, data[k], label=k)
    plt.title("YOLO Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(alpha=0.25)
    if metric_keys:
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_rtdetr(data: Dict[str, List[float]], out_path: Path) -> None:
    x = data.get("epoch", list(range(len(next(iter(data.values()))))))
    train_keys = [k for k in data.keys() if k.startswith("train_") and "n_parameters" not in k][:5]
    test_keys = [k for k in data.keys() if k.startswith("test_")][:5]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for k in train_keys:
        plt.plot(x, data[k], label=k)
    plt.title("RT-DETR Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(alpha=0.25)
    if train_keys:
        plt.legend(fontsize=8)

    plt.subplot(2, 1, 2)
    for k in test_keys:
        plt.plot(x, data[k], label=k)
    plt.title("RT-DETR Validation Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(alpha=0.25)
    if test_keys:
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo_csv = Path(args.yolo_csv).resolve()
    if yolo_csv.exists():
        yolo_data = parse_yolo_results(yolo_csv)
        if yolo_data:
            out = out_dir / "yolo_training_curves.png"
            _plot_yolo(yolo_data, out)
            print(f"YOLO 训练参数图已保存: {out}")
    else:
        print(f"未找到 YOLO 训练日志: {yolo_csv}")

    rtdetr_log = Path(args.rtdetr_log).resolve()
    if rtdetr_log.exists():
        rtdetr_data = parse_rtdetr_log(rtdetr_log)
        if rtdetr_data:
            out = out_dir / "rtdetr_training_curves.png"
            _plot_rtdetr(rtdetr_data, out)
            print(f"RT-DETR 训练参数图已保存: {out}")
    else:
        print(f"未找到 RT-DETR 训练日志: {rtdetr_log}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练过程参数可视化")
    parser.add_argument(
        "--yolo-csv",
        type=str,
        default=str(ROOT / "yolo_landing" / "runs" / "landing_pose" / "results.csv"),
        help="YOLO 训练结果 CSV",
    )
    parser.add_argument(
        "--rtdetr-log",
        type=str,
        default=str(ROOT / "RT-DETR-landing" / "rtdetrv2_pytorch" / "output" / "rtdetrv2_r50vd_landing" / "log.txt"),
        help="RT-DETR 训练日志 log.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "visualization" / "training"),
        help="训练可视化输出目录",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
