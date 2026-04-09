"""
对比测试: 椭圆混合方法 vs 纯 PnP 在三个高度段的精度

对测试集的每张图，同时运行两种方法，按 near/mid/far 分组对比。
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from ultralytics import YOLO
    from yolo_landing.pose_solver import LandingPoseSolver
    from yolo_landing.ellipse_pose_solver import EllipsePoseSolver
    from evaluation.common_metrics import (
        build_gt_camera_pose_map, load_coco_test,
        resolve_rgb_path, rotation_error_deg,
    )

    datasets_root = Path(ROOT / "datasets")
    ann_path = datasets_root / "annotations_coco_test.json"
    weights = ROOT / "yolo_landing" / "runs" / "landing_pose" / "weights" / "best.pt"

    model = YOLO(str(weights))
    pnp_solver = LandingPoseSolver(confidence_threshold=0.3)
    ell_solver = EllipsePoseSolver(confidence_threshold=0.3)

    samples = load_coco_test(ann_path)
    gt_map = build_gt_camera_pose_map(datasets_root)

    # 按高度段分组收集结果
    results = {rl: {"pnp": [], "ellipse": []} for rl in ["near", "mid", "far"]}

    for idx, item in enumerate(samples, start=1):
        img_info = item["image"]
        ann = item["ann"]
        file_name = img_info["file_name"]
        range_label = img_info.get("range_label", "unknown")
        if range_label not in results:
            continue

        img_path = resolve_rgb_path(datasets_root, file_name)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        pred = model.predict(img, conf=0.25, verbose=False, device="0")
        if len(pred) == 0 or pred[0].keypoints is None:
            continue
        r = pred[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        best = int(r.boxes.conf.argmax().item())
        kpts = r.keypoints.data[best].detach().cpu().numpy()
        pred_xy = kpts[:, :2]
        pred_conf = kpts[:, 2]

        # 获取 bbox
        bbox = r.boxes.xyxy[best].detach().cpu().numpy()

        gt_pose = gt_map.get(file_name)
        if gt_pose is None:
            continue

        # 方法 A: 纯 PnP
        pnp_res = pnp_solver.solve(pred_xy, pred_conf, method="epnp")
        if pnp_res is not None:
            R_wc, t_wc = pnp_solver.get_camera_pose_in_world(pnp_res)
            t_err = float(np.linalg.norm(t_wc - gt_pose["t_wc"]))
            r_err = float(rotation_error_deg(R_wc, gt_pose["R_wc"]))
            results[range_label]["pnp"].append({"t_err": t_err, "r_err": r_err})

        # 方法 B: 椭圆混合
        ell_res = ell_solver.solve(img, pred_xy, pred_conf, bbox=bbox, method="epnp")
        if ell_res is not None:
            R_wc_e, t_wc_e = ell_solver.get_camera_pose_in_world(ell_res)
            t_err_e = float(np.linalg.norm(t_wc_e - gt_pose["t_wc"]))
            r_err_e = float(rotation_error_deg(R_wc_e, gt_pose["R_wc"]))
            results[range_label]["ellipse"].append({
                "t_err": t_err_e, "r_err": r_err_e,
                "strategy": ell_res.get("strategy", "unknown"),
                "d_ellipse": ell_res.get("d_ellipse"),
                "ellipse_size": ell_res.get("ellipse_size", 0),
            })

        if idx % 200 == 0:
            print(f"  处理 {idx}/{len(samples)}")

    # 输出结果
    print("\n" + "=" * 90)
    print("对比结果: 纯 PnP vs 椭圆混合方法")
    print("=" * 90)

    for rl in ["near", "mid", "far"]:
        pnp_list = results[rl]["pnp"]
        ell_list = results[rl]["ellipse"]
        if not pnp_list and not ell_list:
            continue

        print(f"\n[{rl.upper()}]  PnP: {len(pnp_list)} samples, Ellipse: {len(ell_list)} samples")

        if pnp_list:
            t_pnp = np.array([r["t_err"] for r in pnp_list])
            r_pnp = np.array([r["r_err"] for r in pnp_list])
            print(f"  PnP     位置误差: mean={t_pnp.mean():.2f}, median={np.median(t_pnp):.2f}")
            print(f"          旋转误差: mean={r_pnp.mean():.2f}°, median={np.median(r_pnp):.2f}°")

        if ell_list:
            t_ell = np.array([r["t_err"] for r in ell_list])
            r_ell = np.array([r["r_err"] for r in ell_list])
            # 策略分布
            strategies = {}
            for r in ell_list:
                s = r.get("strategy", "unknown")
                strategies[s] = strategies.get(s, 0) + 1
            strat_str = ", ".join(f"{k}={v}" for k, v in sorted(strategies.items()))
            print(f"  Ellipse 位置误差: mean={t_ell.mean():.2f}, median={np.median(t_ell):.2f}")
            print(f"          旋转误差: mean={r_ell.mean():.2f}°, median={np.median(r_ell):.2f}°")
            print(f"          策略分布: {strat_str}")

        if pnp_list and ell_list:
            t_pnp_m = np.mean([r["t_err"] for r in pnp_list])
            t_ell_m = np.mean([r["t_err"] for r in ell_list])
            if t_ell_m < t_pnp_m:
                print(f"  → 椭圆方法位置误差降低 {(1 - t_ell_m / t_pnp_m) * 100:.1f}%")
            else:
                print(f"  → PnP 更优 ({(t_ell_m / t_pnp_m - 1) * 100:.1f}% 差距)")


if __name__ == "__main__":
    main()
