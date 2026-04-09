"""
着陆关键点检测后处理
将模型输出转换为像素坐标, 包含关键点位置和可见性预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register


__all__ = ['LandingPostProcessor']


def mod(a, b):
    return a - a // b * b


@register()
class LandingPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries']

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        num_keypoints=9,
    ):
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.num_keypoints = num_keypoints
        self.deploy_mode = False

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits = outputs['pred_logits']
        boxes = outputs['pred_boxes']
        kpts = outputs.get('pred_keypoints', None)
        vis = outputs.get('pred_kpt_vis', None)

        # bbox → 像素坐标
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred = bbox_pred * orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(
                scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes_out = bbox_pred.gather(
                dim=1,
                index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(
                    scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes_out = torch.gather(
                    bbox_pred, dim=1,
                    index=index.unsqueeze(-1).tile(1, 1, bbox_pred.shape[-1]))
            else:
                boxes_out = bbox_pred

        # 关键点 → 像素坐标
        kpts_out = None
        vis_out = None
        if kpts is not None:
            K = self.num_keypoints
            kpts_reshaped = kpts.reshape(kpts.shape[0], kpts.shape[1], K, 2)

            # 选择 top-k 对应的关键点
            kpts_reshaped = kpts_reshaped.gather(
                dim=1,
                index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, 2))

            # 归一化坐标 → 像素坐标
            sz = orig_target_sizes.unsqueeze(1).unsqueeze(1)  # [B,1,1,2]
            kpts_out = kpts_reshaped * sz

        if vis is not None:
            vis_out = vis.gather(
                dim=1,
                index=index.unsqueeze(-1).repeat(1, 1, self.num_keypoints))
            vis_out = torch.sigmoid(vis_out)

        if self.deploy_mode:
            return labels, boxes_out, scores, kpts_out, vis_out

        # 0-indexed label → 1-indexed COCO category_id
        labels = labels + 1

        results = []
        for i in range(labels.shape[0]):
            result = dict(
                labels=labels[i],
                boxes=boxes_out[i],
                scores=scores[i],
            )
            if kpts_out is not None:
                result['keypoints'] = kpts_out[i]
            if vis_out is not None:
                result['kpt_visibility'] = vis_out[i]
            results.append(result)

        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
