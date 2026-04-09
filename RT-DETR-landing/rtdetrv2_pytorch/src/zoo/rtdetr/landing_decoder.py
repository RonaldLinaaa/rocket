"""
着陆关键点检测 Decoder
在 RTDETRv2 Transformer Decoder 基础上增加关键点回归头和可见性预测头,
用于检测着陆标志中的 9 个关键点位置。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .rtdetrv2_decoder import RTDETRTransformerv2, MLP
from .utils import inverse_sigmoid
from .denoising import get_contrastive_denoising_training_group
from ...core import register


__all__ = ['RTDETRTransformerLanding']


@register()
class RTDETRTransformerLanding(RTDETRTransformerv2):
    """基于 RTDETRv2 的着陆关键点检测 Decoder

    在原有检测头 (分类 + bbox回归) 基础上增加:
    1. 关键点坐标回归头 (dec_kpt_head) - 预测 K 个关键点的归一化 2D 坐标
    2. 关键点可见性预测头 (dec_vis_head) - 预测每个关键点是否可见
    """

    def __init__(self, num_keypoints=9, **kwargs):
        super().__init__(**kwargs)
        self.num_keypoints = num_keypoints

        # 每个 decoder 层一个关键点回归头: hidden_dim -> K*2 (x,y per keypoint)
        self.dec_kpt_head = nn.ModuleList([
            MLP(self.hidden_dim, self.hidden_dim, num_keypoints * 2, num_layers=3)
            for _ in range(self.num_layers)
        ])

        # 每个 decoder 层一个可见性预测头: hidden_dim -> K
        self.dec_vis_head = nn.ModuleList([
            nn.Linear(self.hidden_dim, num_keypoints)
            for _ in range(self.num_layers)
        ])

        self._reset_kpt_parameters()

    def _reset_kpt_parameters(self):
        for kpt_head, vis_head in zip(self.dec_kpt_head, self.dec_vis_head):
            init.constant_(kpt_head.layers[-1].weight, 0)
            init.constant_(kpt_head.layers[-1].bias, 0)
            init.constant_(vis_head.bias, 0)

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        # 去噪训练
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets, self.num_classes, self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                None, None, None, None

        init_ref_contents, init_ref_points_unact, \
            enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(
                memory, spatial_shapes,
                denoising_logits, denoising_bbox_unact)

        # ========== 自定义 decoder 循环: 增加关键点预测 ==========
        dec_out_bboxes, dec_out_logits = [], []
        dec_out_kpts, dec_out_vis = [], []
        ref_points_detach = F.sigmoid(init_ref_points_unact)

        output = init_ref_contents
        for i, layer in enumerate(self.decoder.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = self.query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory,
                           spatial_shapes, attn_mask, None, query_pos_embed)

            inter_ref_bbox = F.sigmoid(
                self.dec_bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            # 关键点坐标: sigmoid 保证 [0,1] 归一化
            kpt_pred = F.sigmoid(self.dec_kpt_head[i](output))
            vis_pred = self.dec_vis_head[i](output)

            if self.training:
                dec_out_logits.append(self.dec_score_head[i](output))
                dec_out_kpts.append(kpt_pred)
                dec_out_vis.append(vis_pred)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(
                        self.dec_bbox_head[i](output)
                        + inverse_sigmoid(ref_points)))
            elif i == self.decoder.eval_idx:
                dec_out_logits.append(self.dec_score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_kpts.append(kpt_pred)
                dec_out_vis.append(vis_pred)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        out_bboxes = torch.stack(dec_out_bboxes)
        out_logits = torch.stack(dec_out_logits)
        out_kpts = torch.stack(dec_out_kpts)
        out_vis = torch.stack(dec_out_vis)

        # 去噪分支拆分
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_kpts, out_kpts = torch.split(
                out_kpts, dn_meta['dn_num_split'], dim=2)
            dn_out_vis, out_vis = torch.split(
                out_vis, dn_meta['dn_num_split'], dim=2)

        out = {
            'pred_logits': out_logits[-1],
            'pred_boxes': out_bboxes[-1],
            'pred_keypoints': out_kpts[-1],
            'pred_kpt_vis': out_vis[-1],
        }

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss_kpt(
                out_logits[:-1], out_bboxes[:-1],
                out_kpts[:-1], out_vis[:-1])

            # encoder aux 只有 box/class, 不含关键点
            out['enc_aux_outputs'] = self._set_aux_loss(
                enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {
                'class_agnostic': self.query_select_method == 'agnostic'
            }

            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss_kpt(
                    dn_out_logits, dn_out_bboxes,
                    dn_out_kpts, dn_out_vis)
                out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss_kpt(self, outputs_class, outputs_coord,
                          outputs_kpt, outputs_vis):
        return [{
            'pred_logits': a, 'pred_boxes': b,
            'pred_keypoints': c, 'pred_kpt_vis': d,
        } for a, b, c, d in zip(
            outputs_class, outputs_coord, outputs_kpt, outputs_vis)]
