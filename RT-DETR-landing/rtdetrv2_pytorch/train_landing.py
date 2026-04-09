"""
着陆关键点检测训练入口

所有预训练权重统一存放在 datasets/pretrain/ 中, 完全离线运行。

使用方法:
    # 从本地 COCO 预训练模型微调 (默认自动使用 datasets/pretrain 中的权重)
    python train_landing.py -c configs/landing/rtdetrv2_r50vd_landing.yml

    # 手动指定预训练模型
    python train_landing.py -c configs/landing/rtdetrv2_r50vd_landing.yml \
        -t /path/to/custom_pretrain.pth

    # 从上次训练恢复
    python train_landing.py -c configs/landing/rtdetrv2_r50vd_landing.yml \
        -r output/rtdetrv2_r50vd_landing/last.pth

    # 仅评估
    python train_landing.py -c configs/landing/rtdetrv2_r50vd_landing.yml \
        -r output/rtdetrv2_r50vd_landing/best.pth --test-only
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

import argparse

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS

SCRIPT_DIR = Path(__file__).resolve().parent
PRETRAIN_DIR = SCRIPT_DIR.parent.parent / "datasets" / "pretrain"
DEFAULT_PRETRAIN = PRETRAIN_DIR / "rtdetrv2_r50vd_6x_coco_ema.pth"


def main(args):
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        '不能同时使用 --tuning 和 --resume'

    # 首次训练且未指定 -t/-r 时, 自动使用本地预训练权重
    if args.tuning is None and args.resume is None and not args.test_only:
        if DEFAULT_PRETRAIN.exists():
            args.tuning = str(DEFAULT_PRETRAIN)
            print(f"自动使用本地预训练权重: {DEFAULT_PRETRAIN}")
        else:
            print(f"警告: 本地预训练权重不存在: {DEFAULT_PRETRAIN}")
            print("将从随机初始化开始训练。如需预训练微调, 请将权重文件放入 datasets/pretrain/")

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({
        k: v for k, v in args.__dict__.items()
        if k not in ['update'] and v is not None
    })

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landing Keypoint Detection Training')

    parser.add_argument('-c', '--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('-r', '--resume', type=str,
                        help='从检查点恢复训练')
    parser.add_argument('-t', '--tuning', type=str,
                        help='从预训练模型微调 (默认自动使用 datasets/pretrain 中的权重)')
    parser.add_argument('-d', '--device', type=str, help='设备')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--use-amp', action='store_true',
                        help='混合精度训练')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--summary-dir', type=str, help='TensorBoard 日志目录')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='仅评估不训练')
    parser.add_argument('-u', '--update', nargs='+',
                        help='覆盖 YAML 配置项')
    parser.add_argument('--print-method', type=str, default='builtin')
    parser.add_argument('--print-rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int)

    args = parser.parse_args()
    main(args)
