from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTDETR_ROOT = ROOT / "RT-DETR-landing" / "rtdetrv2_pytorch"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RTDETR_ROOT) not in sys.path:
    sys.path.insert(0, str(RTDETR_ROOT))


def _load_checkpoint_to_model(model, ckpt_path: Path) -> None:
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
        state = ckpt["ema"]["module"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[RT-DETR] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def _preprocess(img_bgr: np.ndarray, imgsz: int, device: torch.device) -> torch.Tensor:
    import cv2
    import numpy as np
    import torch

    img = cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)
    return tensor


def evaluate_rtdetr(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    import torch

    from evaluation.common_metrics import (
        build_gt_camera_pose_map,
        dump_report_and_figures,
        load_coco_test,
        print_metrics_to_console,
        resolve_rgb_path,
        rotation_error_deg,
    )
    from src.core import YAMLConfig
    from yolo_landing.pose_solver import LandingPoseSolver

    datasets_root = Path(args.datasets_root).resolve()
    ann_path = Path(args.ann).resolve()
    out_dir = Path(args.output).resolve()
    config_path = Path(args.config).resolve()
    weights_path = Path(args.weights).resolve()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = YAMLConfig(str(config_path))
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    _load_checkpoint_to_model(model, weights_path)

    samples = load_coco_test(ann_path)
    gt_pose_map = build_gt_camera_pose_map(datasets_root)
    solver = LandingPoseSolver(confidence_threshold=args.kpt_conf)

    records = []
    with torch.no_grad():
        for idx, item in enumerate(samples, start=1):
            img_info = item["image"]
            ann = item["ann"]
            file_name = img_info["file_name"]
            img_path = resolve_rgb_path(datasets_root, file_name)
            img = cv2.imread(str(img_path))
            range_label_early = img_info.get("range_label", "unknown")
            if img is None:
                records.append(
                    {
                        "file": file_name,
                        "range_label": range_label_early,
                        "detected": False,
                        "pose_success": False,
                        "det_time_ms": 0.0,
                        "pnp_time_ms": 0.0,
                        "kp_errors_px": [],
                        "error": "image_not_found",
                    }
                )
                continue

            inp = _preprocess(img, args.imgsz, device)
            h, w = img.shape[:2]
            orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)

            t0 = time.perf_counter()
            outputs = model(inp)
            preds = postprocessor(outputs, orig_size)[0]
            det_time_ms = (time.perf_counter() - t0) * 1000.0

            range_label = img_info.get("range_label", "unknown")

            record = {
                "file": file_name,
                "range_label": range_label,
                "detected": False,
                "pose_success": False,
                "det_time_ms": float(det_time_ms),
                "pnp_time_ms": 0.0,
                "kp_errors_px": [],
            }

            if "scores" not in preds or len(preds["scores"]) == 0:
                records.append(record)
                continue

            best_idx = int(torch.argmax(preds["scores"]).item())
            det_score = float(preds["scores"][best_idx].item())
            if det_score < args.det_conf:
                records.append(record)
                continue

            if "keypoints" not in preds or len(preds["keypoints"]) == 0:
                records.append(record)
                continue

            pred_xy = preds["keypoints"][best_idx].detach().cpu().numpy()
            if "kpt_visibility" in preds and len(preds["kpt_visibility"]) > 0:
                pred_conf = preds["kpt_visibility"][best_idx].detach().cpu().numpy()
            else:
                pred_conf = np.ones(pred_xy.shape[0], dtype=np.float32)

            record["detected"] = True
            gt_kp = np.array(ann["keypoints"], dtype=np.float64).reshape(-1, 3)
            gt_xy = gt_kp[:, :2]
            gt_vis = gt_kp[:, 2] > 0
            if np.any(gt_vis):
                kp_err = np.linalg.norm(pred_xy[gt_vis] - gt_xy[gt_vis], axis=1)
                record["kp_errors_px"] = kp_err.astype(float).tolist()

            t1 = time.perf_counter()
            pose_result = solver.solve(pred_xy, pred_conf, method=args.pnp)
            pnp_time_ms = (time.perf_counter() - t1) * 1000.0
            record["pnp_time_ms"] = float(pnp_time_ms)

            if pose_result is not None:
                gt_pose = gt_pose_map.get(file_name)
                if gt_pose is not None:
                    r_wc, t_wc = solver.get_camera_pose_in_world(pose_result)
                    trans_err = float(np.linalg.norm(t_wc - gt_pose["t_wc"]))
                    rot_err = float(rotation_error_deg(r_wc, gt_pose["R_wc"]))
                    record["pose_success"] = True
                    record["translation_error"] = trans_err
                    record["rotation_error_deg"] = rot_err
                    record["reproj_error_px"] = float(pose_result["reproj_error"])

            records.append(record)
            if idx % 200 == 0:
                print(f"[RT-DETR] processed {idx}/{len(samples)}")

    report = dump_report_and_figures("RT-DETRv2", records, out_dir)
    print_metrics_to_console(report)
    print(f"\n测试结果已保存到: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RT-DETR 方法测试模块")
    parser.add_argument(
        "--config",
        type=str,
        default=str(RTDETR_ROOT / "configs" / "landing" / "rtdetrv2_r50vd_landing.yml"),
        help="RT-DETR 配置路径",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(RTDETR_ROOT / "output" / "rtdetrv2_r50vd_landing" / "best.pth"),
        help="RT-DETR 权重路径",
    )
    parser.add_argument(
        "--datasets-root",
        type=str,
        default=str(ROOT / "datasets"),
        help="datasets 根目录",
    )
    parser.add_argument(
        "--ann",
        type=str,
        default=str(ROOT / "datasets" / "annotations_coco_test.json"),
        help="测试集 COCO 标注路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "visualization" / "testing" / "rtdetr"),
        help="测试可视化输出目录",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    parser.add_argument("--det-conf", type=float, default=0.25, help="检测阈值")
    parser.add_argument("--kpt-conf", type=float, default=0.5, help="PnP 关键点置信度阈值")
    parser.add_argument(
        "--pnp",
        type=str,
        default="epnp",
        choices=["epnp", "ippe", "iterative", "ransac", "sqpnp"],
        help="PnP 算法",
    )
    parser.add_argument("--device", type=str, default="", help="设备，例如 cuda:0 或 cpu")
    return parser


if __name__ == "__main__":
    evaluate_rtdetr(build_parser().parse_args())
