import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def evaluate_yolo(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("请先安装 ultralytics: pip install ultralytics") from exc

    from evaluation.common_metrics import (
        build_gt_camera_pose_map,
        dump_report_and_figures,
        load_coco_test,
        print_metrics_to_console,
        resolve_rgb_path,
        rotation_error_deg,
    )
    from yolo_landing.pose_solver import LandingPoseSolver

    datasets_root = Path(args.datasets_root).resolve()
    ann_path = Path(args.ann).resolve()
    out_dir = Path(args.output).resolve()

    samples = load_coco_test(ann_path)
    gt_pose_map = build_gt_camera_pose_map(datasets_root)
    model = YOLO(args.weights)
    solver = LandingPoseSolver(confidence_threshold=args.kpt_conf)

    records = []
    for idx, item in enumerate(samples, start=1):
        img_info = item["image"]
        ann = item["ann"]
        file_name = img_info["file_name"]
        img_path = resolve_rgb_path(datasets_root, file_name)
        img = cv2.imread(str(img_path))
        range_label = img_info.get("range_label", "unknown")

        if img is None:
            records.append(
                {
                    "file": file_name,
                    "range_label": range_label,
                    "detected": False,
                    "pose_success": False,
                    "det_time_ms": 0.0,
                    "pnp_time_ms": 0.0,
                    "kp_errors_px": [],
                    "error": "image_not_found",
                }
            )
            continue

        t0 = time.perf_counter()
        pred = model.predict(
            img,
            conf=args.det_conf,
            verbose=False,
            device=args.device,
        )
        det_time_ms = (time.perf_counter() - t0) * 1000.0

        record = {
            "file": file_name,
            "range_label": range_label,
            "detected": False,
            "pose_success": False,
            "det_time_ms": float(det_time_ms),
            "pnp_time_ms": 0.0,
            "kp_errors_px": [],
        }

        if len(pred) == 0 or pred[0].keypoints is None or pred[0].keypoints.data is None:
            records.append(record)
            continue

        r = pred[0]
        if r.boxes is None or len(r.boxes) == 0:
            records.append(record)
            continue

        best_idx = int(r.boxes.conf.argmax().item())
        kpts_data = r.keypoints.data[best_idx].detach().cpu().numpy()
        pred_xy = kpts_data[:, :2]
        pred_conf = kpts_data[:, 2]
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
            print(f"[YOLO] processed {idx}/{len(samples)}")

    report = dump_report_and_figures("YOLOv8-pose", records, out_dir)
    print_metrics_to_console(report)
    print(f"\n测试结果已保存到: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO 方法测试模块")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(ROOT / "yolo_landing" / "runs" / "landing_pose" / "weights" / "best.pt"),
        help="YOLO 权重路径",
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
        default=str(ROOT / "visualization" / "testing" / "yolo"),
        help="测试可视化输出目录",
    )
    parser.add_argument("--det-conf", type=float, default=0.25, help="检测阈值")
    parser.add_argument("--kpt-conf", type=float, default=0.5, help="PnP 关键点置信度阈值")
    parser.add_argument(
        "--pnp",
        type=str,
        default="epnp",
        choices=["epnp", "ippe", "iterative", "ransac", "sqpnp"],
        help="PnP 算法",
    )
    parser.add_argument("--device", type=str, default="0", help="推理设备")
    return parser


if __name__ == "__main__":
    evaluate_yolo(build_parser().parse_args())
