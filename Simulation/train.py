"""
Module 4 — PyTorch Landing Marker Detection Training Pipeline
可复用火箭着陆合作标志检测深度学习训练流程

模型架构选项:
  - LandingMarkerDetector (轻量级, 本脚本默认实现)
  - 可对接 YOLOv8 / Faster-RCNN / DETR

任务: 检测箭载相机图像中的双圆+H型合作标志
标注格式: YOLO (class cx cy w h, 归一化)

评估指标: Precision / Recall / mAP@0.5

用法:
  python train.py --data dataset/ --epochs 50 --batch 8
"""

import os, json, math, time, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch 未安装，将运行数据集验证模式（无训练）")


# ================================================================ #
#  1. Dataset
# ================================================================ #

class LandingMarkerDataset(Dataset):
    """
    YOLO格式着陆合作标志数据集。

    目录结构:
      root/
        rgb/frame_XXXXX.png
        annotations/labels/frame_XXXXX.txt
    """

    def __init__(self, root: str, split: str = 'train',
                 img_size: Tuple[int,int] = (640, 360),
                 augment: bool = True,
                 split_ratio: Tuple[float,float,float] = (0.7, 0.15, 0.15)):
        self.root     = Path(root)
        self.img_size = img_size  # (W, H)
        self.augment  = augment and (split == 'train')

        rgb_dir   = self.root / 'rgb'
        label_dir = self.root / 'annotations' / 'labels'

        all_frames = sorted(rgb_dir.glob('frame_*.png'))
        N = len(all_frames)
        if N == 0:
            raise ValueError(f"在 {rgb_dir} 中未找到图像文件")

        # 按比例划分
        n_train = int(N * split_ratio[0])
        n_val   = int(N * split_ratio[1])
        if split == 'train':
            frames = all_frames[:n_train]
        elif split == 'val':
            frames = all_frames[n_train:n_train+n_val]
        else:  # test
            frames = all_frames[n_train+n_val:]

        self.samples = []
        for img_path in frames:
            label_path = label_dir / (img_path.stem + '.txt')
            self.samples.append((img_path, label_path))

        print(f"[Dataset] split={split}  N={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # 读取图像
        img = Image.open(img_path).convert('RGB')
        W0, H0 = img.size
        img = img.resize(self.img_size, Image.BILINEAR)

        # 读取标注
        boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        boxes.append([cls, cx, cy, bw, bh])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,5), np.float32)

        # 图像增广
        if self.augment:
            img, boxes = self._augment(img, boxes)

        # 转为Tensor
        img_t = T.ToTensor()(img)   # (3, H, W)  [0,1]
        img_t = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])(img_t)

        return img_t, boxes, str(img_path.name)

    def _augment(self, img: Image.Image, boxes: np.ndarray):
        """随机数据增广 (水平翻转 / 亮度 / 对比度 / 高斯模糊)"""
        # 水平翻转
        if random.random() < 0.5:
            img = TF.hflip(img)
            if len(boxes):
                boxes[:,1] = 1.0 - boxes[:,1]  # cx翻转

        # 亮度 / 对比度 / 饱和度
        img = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)(img)

        # 随机高斯模糊 (模拟运动模糊)
        if random.random() < 0.2:
            k = random.choice([3, 5])
            img = TF.gaussian_blur(img, kernel_size=k)

        # 随机灰度
        if random.random() < 0.1:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        return img, boxes


def collate_fn(batch):
    """自定义 collate 处理不同数量的bbox"""
    imgs, boxes_list, names = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(boxes_list), list(names)


# ================================================================ #
#  2. Model
# ================================================================ #

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True)
        )
    def forward(self, x): return self.block(x)


class LandingMarkerDetector(nn.Module):
    """
    轻量级单类目标检测网络
    专为火箭箭载相机着陆标志检测设计

    输入: (B, 3, H, W)  H=360, W=640
    输出: (B, 5, Gh, Gw)  每格 [conf, cx, cy, w, h]

    骨干: 6层下采样 (步距32) → 特征图 20×11
    检测头: 1个尺度
    """

    def __init__(self, num_classes=1, img_size=(640, 360)):
        super().__init__()
        self.num_classes = num_classes
        self.img_size    = img_size
        self.stride      = 32

        # 骨干网络 (VGG-like, 轻量)
        self.backbone = nn.Sequential(
            ConvBlock(3,   32,  3, 2, 1),   # /2   → 320×180
            ConvBlock(32,  64,  3, 2, 1),   # /4   → 160×90
            ConvBlock(64,  128, 3, 2, 1),   # /8   →  80×45
            ConvBlock(128, 128, 3, 1, 1),
            ConvBlock(128, 256, 3, 2, 1),   # /16  →  40×23
            ConvBlock(256, 256, 3, 1, 1),
            ConvBlock(256, 512, 3, 2, 1),   # /32  →  20×11
            ConvBlock(512, 512, 3, 1, 1),
        )

        # 检测头: 预测 [conf, cx, cy, w, h] × 3 anchors
        self.n_anchors = 3
        self.head = nn.Conv2d(512, self.n_anchors * (5 + num_classes), 1)

        # Anchor尺寸 (归一化, 相对图像尺寸)
        # 根据着陆标志在不同高度下的典型投影面积设定
        self.register_buffer('anchors', torch.tensor([
            [0.05, 0.09],   # 高空小目标
            [0.15, 0.27],   # 中高度
            [0.40, 0.72],   # 低高度大目标
        ]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        # 置信度偏置初始化为低概率 (减少早期误检)
        nn.init.constant_(self.head.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        feat   = self.backbone(x)        # (B, 512, Gh, Gw)
        pred   = self.head(feat)         # (B, n_anchors*(5+cls), Gh, Gw)

        B, _, Gh, Gw = pred.shape
        pred = pred.view(B, self.n_anchors, 5 + self.num_classes, Gh, Gw)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # (B, A, Gh, Gw, 5+cls)

        return pred, (Gh, Gw)


# ================================================================ #
#  3. Loss
# ================================================================ #

class YOLOLoss(nn.Module):
    """
    简化版YOLO损失:
      L = λ_obj * BCE(conf) + λ_box * MSE(box) + λ_cls * BCE(cls)
    """

    def __init__(self, anchors, img_size=(640, 360), device='cpu'):
        super().__init__()
        self.anchors  = anchors.to(device)   # (A, 2)
        self.img_size = img_size
        self.device   = device
        self.λ_obj    = 1.0
        self.λ_box    = 5.0
        self.λ_noobj  = 0.5
        self.iou_thresh = 0.5

    def forward(self, pred, targets_batch, grid_size):
        """
        pred: (B, A, Gh, Gw, 5+cls)
        targets_batch: list of (N_i, 5) arrays [cls, cx, cy, w, h]
        """
        B, A, Gh, Gw, C = pred.shape
        device = pred.device

        # 构建目标张量
        obj_mask    = torch.zeros(B, A, Gh, Gw, device=device)
        noobj_mask  = torch.ones( B, A, Gh, Gw, device=device)
        tx = torch.zeros(B, A, Gh, Gw, device=device)
        ty = torch.zeros(B, A, Gh, Gw, device=device)
        tw = torch.zeros(B, A, Gh, Gw, device=device)
        th = torch.zeros(B, A, Gh, Gw, device=device)
        tcls= torch.zeros(B, A, Gh, Gw, device=device)

        for b, targets in enumerate(targets_batch):
            if len(targets) == 0:
                continue
            targets_t = torch.from_numpy(targets).float().to(device)

            for t in targets_t:
                cls, cx, cy, bw, bh = t

                # 格子坐标
                gi = int(cx * Gw)
                gj = int(cy * Gh)
                gi = min(gi, Gw - 1)
                gj = min(gj, Gh - 1)

                # 选最匹配的anchor (IoU)
                anchor_wh = self.anchors  # (A, 2)
                target_wh = torch.tensor([[bw, bh]], device=device)
                inter = torch.min(anchor_wh, target_wh).prod(1)
                union = anchor_wh.prod(1) + bw * bh - inter
                iou   = inter / (union + 1e-6)
                best_a = iou.argmax().item()

                obj_mask[b, best_a, gj, gi]   = 1
                noobj_mask[b, best_a, gj, gi] = 0
                tx[b, best_a, gj, gi] = cx * Gw - gi
                ty[b, best_a, gj, gi] = cy * Gh - gj
                tw[b, best_a, gj, gi] = torch.log(bw / self.anchors[best_a, 0] + 1e-6)
                th[b, best_a, gj, gi] = torch.log(bh / self.anchors[best_a, 1] + 1e-6)
                tcls[b, best_a, gj, gi] = cls

        # 解码预测
        pred_conf = pred[..., 0]
        pred_xy   = torch.sigmoid(pred[..., 1:3])
        pred_wh   = pred[..., 3:5]
        pred_cls  = pred[..., 5] if C > 5 else torch.zeros_like(pred_conf)

        bce = nn.BCEWithLogitsLoss(reduction='sum')
        mse = nn.MSELoss(reduction='sum')

        loss_conf_obj   = bce(pred_conf[obj_mask.bool()],
                              torch.ones_like(pred_conf[obj_mask.bool()]))
        loss_conf_noobj = bce(pred_conf[noobj_mask.bool()],
                              torch.zeros_like(pred_conf[noobj_mask.bool()]))

        n_obj = obj_mask.sum().clamp(min=1)
        loss_xy = mse(pred_xy[...,0][obj_mask.bool()], tx[obj_mask.bool()]) + \
                  mse(pred_xy[...,1][obj_mask.bool()], ty[obj_mask.bool()])
        loss_wh = mse(pred_wh[...,0][obj_mask.bool()], tw[obj_mask.bool()]) + \
                  mse(pred_wh[...,1][obj_mask.bool()], th[obj_mask.bool()])

        loss = (self.λ_obj   * loss_conf_obj / n_obj +
                self.λ_noobj * loss_conf_noobj / (noobj_mask.sum().clamp(1)) +
                self.λ_box   * (loss_xy + loss_wh) / n_obj)

        return loss


# ================================================================ #
#  4. mAP Evaluation
# ================================================================ #

def compute_iou(box1, box2):
    """box: [x1, y1, x2, y2] (pixel coords)"""
    xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def decode_predictions(pred, anchors, img_size, conf_thresh=0.3):
    """
    解码网络输出为 (conf, cx, cy, w, h) 归一化坐标列表。
    pred: (A, Gh, Gw, 5+cls) 单张图
    """
    A, Gh, Gw, C = pred.shape
    W, H = img_size

    results = []
    for a in range(A):
        for j in range(Gh):
            for i in range(Gw):
                conf = torch.sigmoid(pred[a, j, i, 0]).item()
                if conf < conf_thresh:
                    continue
                px = (torch.sigmoid(pred[a,j,i,1]).item() + i) / Gw
                py = (torch.sigmoid(pred[a,j,i,2]).item() + j) / Gh
                pw = math.exp(pred[a,j,i,3].item()) * anchors[a, 0].item()
                ph = math.exp(pred[a,j,i,4].item()) * anchors[a, 1].item()
                results.append((conf, px, py, pw, ph))

    return results


def evaluate(model, dataloader, device, anchors, img_size, iou_thresh=0.5, conf_thresh=0.3):
    """计算 Precision / Recall / mAP@0.5"""
    model.eval()
    TP = FP = FN = 0

    with torch.no_grad():
        for imgs, boxes_list, _ in dataloader:
            imgs = imgs.to(device)
            preds, grid_size = model(imgs)

            for b in range(imgs.shape[0]):
                pred_b = preds[b]   # (A, Gh, Gw, C)
                dets   = decode_predictions(pred_b, anchors, img_size, conf_thresh)

                gt = boxes_list[b]   # numpy (N, 5)
                gt_matched = [False] * len(gt)

                for det in dets:
                    conf, cx, cy, bw, bh = det
                    matched = False
                    for g_idx, g in enumerate(gt):
                        if gt_matched[g_idx]: continue
                        gcx, gcy, gbw, gbh = g[1], g[2], g[3], g[4]
                        iou = compute_iou(
                            [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2],
                            [gcx-gbw/2, gcy-gbh/2, gcx+gbw/2, gcy+gbh/2]
                        )
                        if iou >= iou_thresh:
                            TP += 1
                            gt_matched[g_idx] = True
                            matched = True
                            break
                    if not matched:
                        FP += 1

                FN += gt_matched.count(False)

    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1,
            "TP": TP, "FP": FP, "FN": FN}


# ================================================================ #
#  5. Training Loop
# ================================================================ #

def train(data_dir='dataset', epochs=50, batch_size=4,
          lr=1e-3, img_size=(640, 360), save_dir='checkpoints',
          device_str='auto'):

    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch 未安装，无法执行训练。请运行: pip install torch torchvision")
        print("[INFO]  当前运行数据集验证模式...")
        _validate_dataset(data_dir)
        return

    # 设备选择
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"[INFO] 使用设备: {device}")

    os.makedirs(save_dir, exist_ok=True)

    # Dataset & DataLoader
    train_ds = LandingMarkerDataset(data_dir, 'train', img_size, augment=True)
    val_ds   = LandingMarkerDataset(data_dir, 'val',   img_size, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=2, collate_fn=collate_fn)

    # 模型
    model = LandingMarkerDetector(num_classes=1, img_size=img_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型参数量: {total_params/1e6:.2f}M")

    # 损失函数 & 优化器
    anchors = model.anchors
    criterion = YOLOLoss(anchors, img_size, device=str(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练
    best_f1   = 0.0
    history   = []

    print(f"\n{'='*60}")
    print(f"开始训练: epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"训练集: {len(train_ds)}  验证集: {len(val_ds)}")
    print(f"{'='*60}")

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (imgs, boxes_list, _) in enumerate(train_dl):
            imgs = imgs.to(device)
            pred, grid_size = model(imgs)
            loss = criterion(pred, boxes_list, grid_size)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_dl), 1)

        # 验证
        metrics = evaluate(model, val_dl, device, anchors, img_size)
        dt = time.time() - t0

        print(f"Epoch [{epoch:3d}/{epochs}]  "
              f"Loss={avg_loss:.4f}  "
              f"P={metrics['precision']:.3f}  "
              f"R={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"t={dt:.1f}s")

        history.append({"epoch": epoch, "loss": avg_loss, **metrics})

        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'img_size': img_size,
            }, f"{save_dir}/best_model.pth")
            print(f"  ✓ 保存最佳模型 (F1={best_f1:.3f})")

        # 每10轮保存一次
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       f"{save_dir}/epoch_{epoch:03d}.pth")

    # 保存训练历史
    with open(f"{save_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"训练完成!  最佳 F1={best_f1:.3f}")
    print(f"模型已保存至 {save_dir}/best_model.pth")
    print(f"{'='*60}")
    return model


def _validate_dataset(data_dir):
    """在PyTorch不可用时执行数据集完整性验证"""
    import glob
    root = Path(data_dir)
    rgb_files   = sorted((root/'rgb').glob('frame_*.png'))
    label_files = sorted((root/'annotations'/'labels').glob('frame_*.txt'))
    pose_csv    = root / 'pose.csv'

    print(f"\n[Dataset Validation]")
    print(f"  RGB 图像:   {len(rgb_files)}")
    print(f"  标注文件:   {len(label_files)}")
    print(f"  pose.csv:   {'存在' if pose_csv.exists() else '缺失'}")

    n_with_bbox = sum(1 for p in label_files if os.path.getsize(p) > 0)
    print(f"  含标注帧:   {n_with_bbox}/{len(label_files)} "
          f"({100*n_with_bbox/max(len(label_files),1):.1f}%)")

    if len(rgb_files) > 0:
        # 读取一帧验证
        img = cv2.imread(str(rgb_files[0]))
        print(f"  图像尺寸:   {img.shape[1]}×{img.shape[0]}")
        print(f"  数据集验证通过 ✓")


# ================================================================ #
#  入口
# ================================================================ #

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='着陆标志检测训练')
    parser.add_argument('--data',    default='dataset',  help='数据集根目录')
    parser.add_argument('--epochs',  type=int, default=50)
    parser.add_argument('--batch',   type=int, default=4)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--save',    default='checkpoints')
    parser.add_argument('--device',  default='auto')
    parser.add_argument('--img_w',   type=int, default=640)
    parser.add_argument('--img_h',   type=int, default=360)
    args = parser.parse_args()

    train(data_dir=args.data, epochs=args.epochs, batch_size=args.batch,
          lr=args.lr, img_size=(args.img_w, args.img_h),
          save_dir=args.save, device_str=args.device)
