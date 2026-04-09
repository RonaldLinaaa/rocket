"""
YOLOv8-pose 着陆关键点检测训练

所有预训练权重统一存放在 datasets/pretrain/ 中, 完全离线运行。

用法:
  # 首次运行先转换数据集
  python convert_dataset.py --copy

  # 训练 (使用本地 datasets/pretrain/yolov8s-pose.pt)
  python train.py

  # 指定参数
  python train.py --model yolov8m-pose.pt --epochs 200 --batch 8 --device 0
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

DATASET_YAML = Path(__file__).resolve().parent / "yolo_dataset" / "dataset.yaml"
PROJECT_DIR = Path(__file__).resolve().parent / "runs"
ROOT_DIR = Path(__file__).resolve().parent.parent
PRETRAIN_DIR = ROOT_DIR / "datasets" / "pretrain"
DEFAULT_PRETRAIN = PRETRAIN_DIR / "yolov8s-pose.pt"


def resolve_model_path(model_arg: str) -> str:
    """
    离线优先解析权重路径:
    1) 若用户给的是存在的绝对/相对路径, 直接使用
    2) 若 datasets/pretrain 下有同名权重, 使用本地文件
    3) 否则报错, 不尝试在线下载
    """
    requested = Path(model_arg)
    if requested.exists():
        return str(requested)

    local_candidate = PRETRAIN_DIR / requested.name
    if local_candidate.exists():
        print(f"使用本地预训练权重: {local_candidate}")
        return str(local_candidate)

    raise FileNotFoundError(
        f"预训练权重未找到: {model_arg}\n"
        f"请将预训练模型放入 {PRETRAIN_DIR}/ 目录中。\n"
        f"例如: datasets/pretrain/yolov8s-pose.pt"
    )


def main():
    parser = argparse.ArgumentParser(description="Landing Keypoint Detection Training")
    parser.add_argument("--model", type=str, default=str(DEFAULT_PRETRAIN),
                        help="预训练模型路径或名称，默认使用 datasets/pretrain/yolov8s-pose.pt")
    parser.add_argument("--data", type=str, default=str(DATASET_YAML),
                        help="数据集 YAML 路径")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0",
                        help="设备: 0, 0,1, cpu")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处恢复训练")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"数据集不存在: {args.data}\n"
            "请先运行: python convert_dataset.py --copy"
        )

    model_path = resolve_model_path(args.model)
    model = YOLO(model_path)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(PROJECT_DIR),
        name="landing_pose",
        resume=args.resume,

        # 优化器
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,

        # 数据增强 (保守策略, 保护关键点语义)
        fliplr=0.0,       # 禁用水平翻转 (会破坏关键点对应关系)
        flipud=0.0,       # 禁用垂直翻转
        mosaic=0.0,        # 禁用 mosaic (单目标场景不适用)
        mixup=0.0,
        scale=0.3,         # 缩放增强
        translate=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        # 混合精度（离线环境禁用以跳过 AMP 下载检查）
        amp=False,

        # 其他
        patience=20,       # 早停耐心
        save_period=10,
        plots=True,
        verbose=True,
    )

    print(f"\n训练完成, 结果保存在: {PROJECT_DIR / 'landing_pose'}")


if __name__ == "__main__":
    main()
