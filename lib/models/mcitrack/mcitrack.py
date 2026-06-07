"""
MCITrack Model
"""
import os

import torch
import math
from torch import nn
import torch.nn.functional as F
from lib.models.mcitrack.encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from .fastitpn import fastitpnl, fastitpnb
from .neck import build_neck
from collections import OrderedDict
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
from .decoder import NonParametricMiningHead, TemplateAwareFusion

from ..language_model import build_bert


class MCITrack(nn.Module):
    """ This is the base class for MCITrack """
    def __init__(self, encoder, decoder, neck,cfg,
                 num_frames=1, num_template=1, decoder_type="CENTER",text_encoder=None ):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type
        self.neck = neck
        self.cfg = cfg

        # === 新增：初始化模块 A 和 B ===
        hidden_dim = self.cfg.MODEL.HIDDEN_DIM  # 确保这个参数在 cfg 里有，或者手动指定如 256


        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template
        self.freeze_en = cfg.TRAIN.FREEZE_ENCODER
        self.interaction_indexes = cfg.MODEL.ENCODER.INTERACTION_INDEXES
        # self.language_backbone = text_encoder
        self.text_proj = nn.Linear(768, 512)

        # === TemplateAwareFusion 模块 (从 ParaHydra+ PMIFM 迁移) ===
        # ENABLE=False 时不创建模块, 与 baseline 完全一致
        self.use_fusion = cfg.MODEL.FUSION.ENABLE
        if self.use_fusion:
            self.fusion = TemplateAwareFusion(
                d_model=cfg.MODEL.NECK.D_MODEL,
                scale_factor=cfg.MODEL.FUSION.SCALE_FACTOR,
                para_factor=cfg.MODEL.FUSION.PARA_FACTOR,
                num_heads=cfg.MODEL.FUSION.NUM_HEADS,
                dropout=cfg.MODEL.FUSION.DROPOUT,
            )


    def forward(self, template_list=None, search_list=None, template_anno_list=None,text = None,enc_opt=None,neck_h_state=None, feature=None,
                pos_box=None, neg_boxes=None, prev_pos_feat=None, prev_neg_feat=None,mode="encoder",gt_anno_list=None):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        # === 新增：序列训练模式 ===

        if mode == "encoder":
            # text_fea = self.language_backbone(text)
            return self.forward_encoder(template_list, search_list, template_anno_list)
        elif mode == "neck":
            return self.forward_neck(enc_opt,neck_h_state)
        elif mode == "decoder":
            return self.forward_decoder(feature)
        elif mode == "fusion":
            # enc_opt = Encoder 输出 (Neck 前, 模板独立)
            # feature = Neck 输出的 xs (搜索特征, 已增强)
            return self.forward_fusion(enc_opt, feature)
        else:
            raise ValueError



    def forward_encoder(self, template_list, search_list, template_anno_list):
        # Forward the encoder

        xz = self.encoder(template_list, search_list, template_anno_list)
        return xz
    def forward_neck(self,enc_out,neck_h_state):
        x = enc_out
        xs = x[:, 0:self.num_patch_x]
        x,xs,h = self.neck(x,xs,neck_h_state,self.encoder.body.blocks,self.interaction_indexes)
        x = self.encoder.body.fc_norm(x)
        xs = xs + x[:, 0:self.num_patch_x]
        return x,xs,h

    def forward_fusion(self, enc_opt, search_feat):
        """
        TemplateAwareFusion: 用循环一致性加权融合多模板信息到搜索特征。

        Args:
            enc_opt:     (B, L_x + N*L_z, D) — Encoder 输出 (Neck 之前, 模板独立)
            search_feat: (B, L_x, D) — Neck 输出的搜索特征 (xs, 已增强)

        Returns:
            enhanced_search: (B, L_x, D)

        注: enc_opt 在经过 forward_neck 后不会被修改 (已验证: neck 内无 in-place op)
        """
        if not self.use_fusion:
            return search_feat  # ENABLE=False 时直接返回, 等同 baseline
        return self.fusion(enc_opt, search_feat, self.num_patch_x, self.num_patch_z)

    def forward_decoder(self, feature, gt_score_map=None):
        # feature = feature[0]
        # feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)
        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError





    def mask_feature_map(self, feature_map, boxes_to_mask, expansion_ratio=1.0):
        """
        将 feature_map 上对应 boxes 区域的值置零 (支持扩大范围)。
        feature_map: [B, C, H, W]
        boxes_to_mask: [B, 4] (x1, y1, x2, y2) 归一化坐标 [0,1]
        expansion_ratio: 扩大比例，例如 2.0 表示宽高扩大为原来的 2 倍
        """
        B, C, H, W = feature_map.shape
        masked_feat = feature_map.clone()

        for i in range(B):
            # 1. 获取原始归一化坐标
            x1_norm, y1_norm, x2_norm, y2_norm = boxes_to_mask[i]

            # ==========================================
            # ✅ 新增逻辑：计算放大后的归一化坐标
            # ==========================================
            if expansion_ratio > 1.0:
                # 计算中心点和宽高
                w_norm = x2_norm - x1_norm
                h_norm = y2_norm - y1_norm
                cx = (x1_norm + x2_norm) / 2
                cy = (y1_norm + y2_norm) / 2

                # 放大宽高
                w_new = w_norm * expansion_ratio
                h_new = h_norm * expansion_ratio

                # 重新计算左上角和右下角
                x1_norm = cx - w_new / 2
                y1_norm = cy - h_new / 2
                x2_norm = cx + w_new / 2
                y2_norm = cy + h_new / 2

            # 2. 转换为像素坐标 (使用新的归一化坐标)
            x1_pix = int(x1_norm * W)
            y1_pix = int(y1_norm * H)
            x2_pix = int(x2_norm * W)
            y2_pix = int(y2_norm * H)

            # 3. 边界截断，防止越界 (这一步非常重要，因为放大后可能超出 [0,1] 范围)
            x1_pix, y1_pix = max(0, x1_pix), max(0, y1_pix)
            x2_pix, y2_pix = min(W, x2_pix), min(H, y2_pix)

            # 4. 如果框有效（面积>0），则置零
            if x2_pix > x1_pix and y2_pix > y1_pix:
                masked_feat[i, :, y1_pix:y2_pix, x1_pix:x2_pix] = 0.0

        return masked_feat


def build_mcitrack(cfg):


    encoder = build_encoder(cfg)
    neck = build_neck(cfg,encoder)
    decoder = build_decoder(cfg, neck)
    text_encoder = build_bert()
    model = MCITrack(
        encoder,
        decoder,
        neck,
        cfg,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        text_encoder=text_encoder
    )

    if 'MCITrack_' in cfg.MODEL.PRETRAIN_FILE:
        pretrained_path = '/home/zhanggt/MCITrack/output/checkpoints/train/mcitrack/mcitrack_b224'
        file_name = cfg.MODEL.PRETRAIN_FILE
        pth = os.path.join(pretrained_path, file_name)
        checkpoint = torch.load(pth, map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
