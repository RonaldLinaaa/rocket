"""
着陆关键点检测 Loss 函数

为 PnP 位姿解算优化设计的多任务损失:
  L_total = L_vfl + L_bbox + L_giou        (检测损失, 继承自 RTDETRCriterionv2)
          + L_keypoints                     (关键点 Smooth-L1 回归)
          + L_oks                           (OKS 尺度感知关键点相似度)
          + L_vis                           (关键点可见性 Focal Loss)
          + L_struct                        (结构一致性 - 保持刚体几何约束, PnP 友好)

设计思路:
  1. L_keypoints: 直接监督关键点坐标精度, 是 PnP 解算的基础
  2. L_oks: 按目标尺度归一化, 对远距离小目标的宽容度更高
  3. L_vis: 准确预测可见性, 使 PnP 只使用可靠的关键点
  4. L_struct: 惩罚不符合刚体投影约束的关键点配置,
     通过保持关键点间归一化距离矩阵的一致性, 确保几何结构正确,
     即使单个关键点有误差也能维持整体结构的可用性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rtdetrv2_criterion import RTDETRCriterionv2
from ...core import register


__all__ = ['LandingKeypointCriterion']


@register()
class LandingKeypointCriterion(RTDETRCriterionv2):
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        num_keypoints=9,
        oks_sigmas=None,
        alpha=0.75,
        gamma=2.0,
        num_classes=80,
        boxes_weight_format=None,
        share_matched_indices=False,
    ):
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
        )
        self.num_keypoints = num_keypoints

        # OKS 中每个关键点的标准差 (控制定位难度)
        # 着陆标志关键点均为明确的结构点, 使用较小的 sigma
        if oks_sigmas is None:
            oks_sigmas = [0.05] * num_keypoints
        self.register_buffer(
            'oks_sigmas',
            torch.tensor(oks_sigmas, dtype=torch.float32))

    # ------------------------------------------------------------------
    # 关键点回归损失: Smooth-L1
    # ------------------------------------------------------------------
    def loss_keypoints(self, outputs, targets, indices, num_boxes, **kw):
        if 'pred_keypoints' not in outputs:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_kpts = outputs['pred_keypoints'][idx]            # [M, K*2]
        tgt_kpts = torch.cat([
            t['keypoints'][j] for t, (_, j) in zip(targets, indices)
        ], dim=0)                                             # [M, K, 3]

        src_xy = src_kpts.reshape(-1, self.num_keypoints, 2)  # [M, K, 2]
        tgt_xy = tgt_kpts[:, :, :2]                            # [M, K, 2]
        vis = tgt_kpts[:, :, 2]                                # [M, K]

        # 只对有标注的关键点 (v > 0) 计算损失
        vis_mask = (vis > 0).float()
        loss = F.smooth_l1_loss(src_xy, tgt_xy, reduction='none')  # [M, K, 2]
        loss = (loss.sum(-1) * vis_mask).sum() / (vis_mask.sum() + 1e-6)

        return {'loss_keypoints': loss}

    # ------------------------------------------------------------------
    # OKS 损失: 尺度感知关键点相似度
    # OKS_i = mean_k[ exp(-d^2 / (2 * s^2 * sigma_k^2)) * (v>0) ]
    # 远距离小目标 s 小 → OKS 对同等像素误差更宽容
    # ------------------------------------------------------------------
    def loss_oks(self, outputs, targets, indices, num_boxes, **kw):
        if 'pred_keypoints' not in outputs:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_kpts = outputs['pred_keypoints'][idx]
        tgt_kpts = torch.cat([
            t['keypoints'][j] for t, (_, j) in zip(targets, indices)
        ], dim=0)
        tgt_boxes = torch.cat([
            t['boxes'][j] for t, (_, j) in zip(targets, indices)
        ], dim=0)                                              # [M, 4] cxcywh

        src_xy = src_kpts.reshape(-1, self.num_keypoints, 2)
        tgt_xy = tgt_kpts[:, :, :2]
        vis = tgt_kpts[:, :, 2]
        vis_mask = (vis > 0).float()

        # 目标尺度 s = sqrt(w * h)
        s = (tgt_boxes[:, 2] * tgt_boxes[:, 3]).sqrt()        # [M]
        s = s.unsqueeze(-1).clamp(min=1e-6)                    # [M, 1]
        sigmas = self.oks_sigmas.unsqueeze(0)                  # [1, K]

        d_sq = ((src_xy - tgt_xy) ** 2).sum(-1)               # [M, K]
        oks_per_kpt = torch.exp(-d_sq / (2 * s ** 2 * sigmas ** 2 + 1e-8))
        oks = (oks_per_kpt * vis_mask).sum(-1) / (vis_mask.sum(-1) + 1e-6)

        loss = (1 - oks).mean()
        return {'loss_oks': loss}

    # ------------------------------------------------------------------
    # 可见性预测损失: Binary Focal Loss
    # 目标: v == 2 → visible(1), v < 2 → not visible(0)
    # ------------------------------------------------------------------
    def loss_visibility(self, outputs, targets, indices, num_boxes, **kw):
        if 'pred_kpt_vis' not in outputs:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_vis = outputs['pred_kpt_vis'][idx]                 # [M, K]
        tgt_kpts = torch.cat([
            t['keypoints'][j] for t, (_, j) in zip(targets, indices)
        ], dim=0)
        tgt_v = tgt_kpts[:, :, 2]                              # [M, K]

        # 只在有标注的关键点上计算 (v > 0)
        labeled_mask = (tgt_v > 0).float()
        vis_target = (tgt_v == 2).float()                      # 可见为 1

        # Focal-style 权重
        p = torch.sigmoid(src_vis)
        focal_weight = (1 - p) ** 2 * vis_target + p ** 2 * (1 - vis_target)

        bce = F.binary_cross_entropy_with_logits(
            src_vis, vis_target, reduction='none')
        loss = (bce * focal_weight * labeled_mask).sum() / (labeled_mask.sum() + 1e-6)

        return {'loss_vis': loss}

    # ------------------------------------------------------------------
    # 结构一致性损失: 归一化距离矩阵
    # 关键点间归一化成对距离应保持一致, 强制刚体几何约束,
    # 即使透视畸变下也鼓励正确的空间关系
    # ------------------------------------------------------------------
    def loss_structural(self, outputs, targets, indices, num_boxes, **kw):
        if 'pred_keypoints' not in outputs:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_kpts = outputs['pred_keypoints'][idx]
        tgt_kpts = torch.cat([
            t['keypoints'][j] for t, (_, j) in zip(targets, indices)
        ], dim=0)

        K = self.num_keypoints
        src_xy = src_kpts.reshape(-1, K, 2)
        tgt_xy = tgt_kpts[:, :, :2]
        vis = tgt_kpts[:, :, 2]
        vis_mask = (vis > 0).float()                           # [M, K]

        if vis_mask.sum() < 2:
            device = src_xy.device
            return {'loss_struct': torch.tensor(0., device=device)}

        # 成对距离矩阵 [M, K, K]  (cdist 不支持 float16，强制 float32)
        src_dist = torch.cdist(src_xy.float(), src_xy.float())
        tgt_dist = torch.cdist(tgt_xy.float(), tgt_xy.float())

        # 按均值归一化 → 尺度不变
        src_mean = src_dist.mean(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        tgt_mean = tgt_dist.mean(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        src_norm = src_dist / src_mean
        tgt_norm = tgt_dist / tgt_mean

        # 只在两端都可见的关键点对上计算
        pair_mask = vis_mask.unsqueeze(-1) * vis_mask.unsqueeze(-2)  # [M, K, K]
        diag_mask = 1 - torch.eye(K, device=src_xy.device).unsqueeze(0)
        pair_mask = pair_mask * diag_mask

        loss = F.smooth_l1_loss(
            src_norm * pair_mask, tgt_norm * pair_mask, reduction='sum')
        loss = loss / (pair_mask.sum() + 1e-6)

        return {'loss_struct': loss}

    # ------------------------------------------------------------------
    # 扩展 loss 分发
    # ------------------------------------------------------------------
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'keypoints': self.loss_keypoints,
            'oks': self.loss_oks,
            'vis': self.loss_visibility,
            'struct': self.loss_structural,
        }
        assert loss in loss_map, \
            f'Loss "{loss}" not available. Choose from {list(loss_map.keys())}'

        # encoder aux 阶段没有关键点输出, 跳过
        kpt_losses = {'keypoints', 'oks', 'vis', 'struct'}
        if loss in kpt_losses and 'pred_keypoints' not in outputs:
            return {}

        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
