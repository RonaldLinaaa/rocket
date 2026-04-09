"""
着陆标志关键点数据集

基于 COCO keypoints 格式, 加载着陆图像及其关键点标注。
支持按 split 字段 (train/val/test) 筛选图像。
关键点坐标在返回时自动归一化到 [0, 1]。
"""

import os
from pathlib import Path

import torch
from PIL import Image
from .coco_dataset import CocoDetection
from .._misc import convert_to_tv_tensor
from ...core import register


__all__ = ['LandingKeypointDataset']


def _resolve_image_path(root: str, file_name: str) -> str:
    """处理 rocket_render_XX/rocket_render_YYYY.png → rocket_render_XX/rgb/YYYY.png"""
    direct = os.path.join(root, file_name)
    if os.path.exists(direct):
        return direct
    parts = file_name.split("/")
    if len(parts) == 2:
        seq, fname = parts[0], parts[1]
        frame = fname.replace("rocket_render_", "")
        for sub in ("rgb", "images"):
            alt = os.path.join(root, seq, sub, frame)
            if os.path.exists(alt):
                return alt
    return direct


@register()
class LandingKeypointDataset(CocoDetection):
    __inject__ = ['transforms']

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        num_keypoints=9,
        split=None,
        remap_mscoco_category=False,
    ):
        super().__init__(
            img_folder, ann_file, transforms,
            return_masks=False, remap_mscoco_category=remap_mscoco_category)
        self.num_keypoints = num_keypoints

        if split is not None:
            valid_ids = set(
                img['id'] for img in self.coco.dataset['images']
                if img.get('split') == split
            )
            self.ids = [id_ for id_ in self.ids if id_ in valid_ids]

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        resolved = _resolve_image_path(self.root, path)
        return Image.open(resolved).convert("RGB")

    def load_item(self, idx):
        img, target = super().load_item(idx)
        # category_id (1-based) → 0-indexed label
        if 'labels' in target:
            target['labels'] = target['labels'] - 1
        return img, target

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)

        # 关键点归一化: 像素坐标 → [0,1]
        if 'keypoints' in target and target['keypoints'].numel() > 0:
            orig_size = target['orig_size']          # [W, H]
            kpts = target['keypoints'].clone()       # [N, K, 3]
            kpts[:, :, 0] = kpts[:, :, 0] / orig_size[0].float()
            kpts[:, :, 1] = kpts[:, :, 1] / orig_size[1].float()
            kpts[:, :, :2] = kpts[:, :, :2].clamp(0, 1)
            target['keypoints'] = kpts
        else:
            # 无标注时创建空的关键点张量
            target['keypoints'] = torch.zeros(
                0, self.num_keypoints, 3, dtype=torch.float32)

        return img, target
