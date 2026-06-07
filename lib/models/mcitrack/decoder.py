import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.utils.box_ops import box_xyxy_to_cxcywh
from .consistency import cycle_consistency_diag  # 共享的循环一致性数学 (与诊断模块同源)


class NonParametricMiningHead(nn.Module):
    """
    无参数挖掘头：直接计算特征图与模版的余弦相似度
    """

    def __init__(self):
        super().__init__()

    def forward(self, search_feat, template_feat, mask_box=None):
        """
        search_feat: [B, C, H, W] 或 [B, L, C]
        template_feat: [B, C, H, W] (或者池化后的向量)
        mask_box: [B, 4] 归一化预测框
        """
        # --- [优化 1] 自动处理序列输入 ---
        if search_feat.dim() == 3:  # [B, L, C]
            B, L, C = search_feat.shape
            H = W = int(L ** 0.5)
            search_feat = search_feat.transpose(1, 2).view(B, C, H, W)
        else:
            B, C, H, W = search_feat.shape

        # 1. 模版处理
        if template_feat.shape[-1] > 1:
            t_feat = F.adaptive_avg_pool2d(template_feat, (1, 1))
        else:
            t_feat = template_feat

        # 2. 归一化 (L2 Normalize)
        s_norm = F.normalize(search_feat, p=2, dim=1)
        t_norm = F.normalize(t_feat, p=2, dim=1)

        # 3. 计算相似度图 [-1.0, 1.0]
        sim_map = (s_norm * t_norm).sum(dim=1, keepdim=True)

        # 4. Mask 掉正样本
        if mask_box is not None:
            sim_map = self.apply_mask(sim_map, mask_box)

        # 5. 找最大值
        bs, _, h, w = sim_map.shape
        sim_flat = sim_map.view(bs, -1)
        max_val, idx = torch.max(sim_flat, dim=1)

        # 索引转坐标
        idx_y = idx // w
        idx_x = idx % w

        # 转为归一化中心点
        cx = (idx_x.float() + 0.5) / w
        cy = (idx_y.float() + 0.5) / h

        # --- [优化 2] 借用正样本宽高 ---
        if mask_box is not None:
            # mask_box 是 [B, 4] (x1, y1, x2, y2)
            # 计算宽高
            current_w = mask_box[:, 2] - mask_box[:, 0]
            current_h = mask_box[:, 3] - mask_box[:, 1]
        else:
            current_w = torch.tensor([0.1], device=search_feat.device).expand(B)
            current_h = torch.tensor([0.1], device=search_feat.device).expand(B)

        # 构造 Box [cx, cy, w, h]
        pred_box = torch.stack([cx, cy, current_w, current_h], dim=-1)

        return pred_box.unsqueeze(1), max_val, sim_map

    def apply_mask(self, sim_map, boxes):
        B, _, H, W = sim_map.shape
        for i in range(B):
            x1_n, y1_n, x2_n, y2_n = boxes[i]

            w_box = x2_n - x1_n
            h_box = y2_n - y1_n

            # --- [优化 3] 扩大系数改为 0.5 (即总宽高 2.0 倍) ---
            # 这样更安全，彻底防止漂移到边缘
            pad_w = 0.001 * w_box
            pad_h = 0.001 * h_box

            x1 = int(max(0, (x1_n - pad_w) * W))
            y1 = int(max(0, (y1_n - pad_h) * H))
            x2 = int(min(W, (x2_n + pad_w) * W))
            y2 = int(min(H, (y2_n + pad_h) * H))

            # 填一个极小值
            sim_map[i, :, y1:y2, x1:x2] = -10.0

        return sim_map

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x) # x: torch.Size([b, c, h, w])
        # score_map_ctr: torch.Size([32, 1, 16, 16]) size_map: torch.Size([32, 2, 16, 16]) offset_map: torch.Size([32, 2, 16, 16])

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True) # score_map_ctr.flatten(1): torch.Size([32, 256]) idx: torch.Size([32, 1]) max_score: torch.Size([32, 1])
        idx_y = torch.div(idx, self.feat_sz, rounding_mode='floor')
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx) # size_map: torch.Size([32, 2, 16, 16])  size_map.flatten(2): torch.Size([32, 2, 256])
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset

class MLPPredictor(nn.Module):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16):
        super(MLPPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        self.num_layers = 3
        h = [channel] * (self.num_layers - 1)
        self.layers_cls = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([inplanes] + h, h + [1]))
        self.layers_reg = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([inplanes] + h, h + [4]))

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), offset_map)

        return score_map, bbox, offset_map

    def cal_bbox(self, score_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map.flatten(1), dim=1, keepdim=True)
        idx_y = torch.div(idx, self.feat_sz, rounding_mode='floor')
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 4, 1) # torch.Size([32, 4, 1])
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
        # offset: (l,t,r,b)

        # x1, y1, x2, y2
        bbox = torch.cat([idx_x.to(torch.float) / self.feat_sz - offset[:, :1], # the offset should not divide the self.feat_sz, since I use the sigmoid to limit it in (0,1)
                          idx_y.to(torch.float) / self.feat_sz - offset[:, 1:2],
                          idx_x.to(torch.float) / self.feat_sz + offset[:, 2:3],
                          idx_y.to(torch.float) / self.feat_sz + offset[:, 3:4],
                          ], dim=1)
        bbox = box_xyxy_to_cxcywh(bbox)
        if return_score:
            return bbox, max_score
        return bbox

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        x_cls = x
        for i, layer in enumerate(self.layers_cls):
            x_cls = F.relu(layer(x_cls)) if i < self.num_layers - 1 else layer(x_cls)
        x_cls = x_cls.permute(0,2,1).reshape(-1,1,self.feat_sz,self.feat_sz)

        x_reg = x
        for i, layer in enumerate(self.layers_reg):
            x_reg = F.relu(layer(x_reg)) if i < self.num_layers - 1 else layer(x_reg)
        x_reg = x_reg.permute(0, 2, 1).reshape(-1, 4, self.feat_sz, self.feat_sz)

        return _sigmoid(x_cls), _sigmoid(x_reg)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ============================================================
# TemplateAwareFusion — 从 ParaHydra+ PMIFM 迁移
# ============================================================

class TokenResidualBlock(nn.Module):
    """
    对应 PMIFM 的 ResidualBlock (res_blk.py L124-154)
    原始: Conv3x3(in, out) → GELU → Conv3x3(out, out) → GELU + skip(in→out)
    token: Linear(in, out) → GELU → Linear(out, out) → GELU + skip(in→out)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.act = nn.GELU()
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x):
        identity = x
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        if self.skip is not None:
            identity = self.skip(identity)
        return out + identity


class TemplateAwareFusion(nn.Module):
    """
    从 ParaHydra+ PMIFM 迁移的多模板融合模块。
    原始 PMIFM: 多视角 4D 特征 → 水平+垂直视差 → V 加权融合 → 防重影
    迁移版本:   多模板 3D token → 全局 token 注意力 → V 加权融合 → 防漂移

    关键对应关系:
      - PMIFM.deep_feature (2×ResBlock)  → self.deep_feature (2×TokenResidualBlock)
      - MultiParallax (BN+Conv1x1 Q/K)   → self.q_norm+q_proj / k_norm+k_proj (LN+Linear)
      - PMIFM.attn (EfficientAttention)   → self.cross_attn (nn.MultiheadAttention)
      - PMIFM.fusion (ResBlock(C*2,C)+ResBlock(C,C)) → self.fusion (同结构 TokenResidualBlock)
    """
    def __init__(self, d_model=512, scale_factor=24.0, para_factor=10.0,
                 num_heads=8, dropout=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.para_factor = para_factor
        self.d_model = d_model

        # 对应 PMIFM 的 self.deep_feature = 2 × ResidualBlock(C, C)
        # [修复 ④] 搜索特征来自 Neck 之后 (已被目标上下文增强)，模板特征来自 Neck 之前 (独立)，
        # 二者处于不同表征分布。PMIFM 中所有视角同阶段、共享 deep_feature；这里两侧异源，
        # 因此给搜索/模板各自独立的 deep_feature，让各自分布单独适配 (不再强行共享一套权重)。
        self.deep_feature_search = nn.Sequential(
            TokenResidualBlock(d_model, d_model),
            TokenResidualBlock(d_model, d_model),
        )
        self.deep_feature_template = nn.Sequential(
            TokenResidualBlock(d_model, d_model),
            TokenResidualBlock(d_model, d_model),
        )

        # 对应 MultiParallax 的 Q/K 投影
        # 原始: BN → Conv1x1 (因为 SKFF=identity)
        # token 空间: LayerNorm → Linear
        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.scale = d_model ** -0.5

        # 对应 PMIFM 的 self.attn (EfficientAttention)
        # 原始: EfficientAttention(key_channels=C//8, head_count=2, value_channels=C//4)
        # 迁移: nn.MultiheadAttention — 标准 scaled dot-product attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # 对应 PMIFM 的 self.fusion = ResBlock(C*2, C) + ResBlock(C, C)
        self.fusion = nn.Sequential(
            TokenResidualBlock(d_model * 2, d_model),
            TokenResidualBlock(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

        # [修复 ③] ReZero / LayerScale 式残差门控，初值为 0。
        # 输出 fused = identity + gamma * norm(fusion(...))，gamma=0 时整个模块是恒等映射，
        # 即微调起点与 baseline 完全一致；训练只在 baseline 上学残差，避免随机初始化破坏预训练特征。
        # (与本仓库 neck.Injector.gamma 的 init=0 做法一致；参考 ReZero / CaiT-LayerScale)
        self.gamma = nn.Parameter(torch.zeros(d_model))

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform 初始化所有新增参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def compute_parallax(self, search_feat, template_feat):
        """
        对应 MultiParallax.forward (但在 token 空间而非图像空间)

        Args:
            search_feat:   (B, L_x, D)  — deep_feature 后的搜索特征
            template_feat: (B, L_z, D)  — deep_feature 后的模板特征
        Returns:
            V:       (B, 1, L_x) — token 级一致性 (对应原始的 (B,1,H,W))
            aligned: (B, L_x, D) — 对齐特征

        与原始 HorizontalParallax 的差异:
          1. 原始沿 W 维去均值 (Q = Q - mean(Q, dim=W))，此处用 LayerNorm 代替
          2. 原始无 score 缩放因子，此处加了 d^{-0.5} 防止高维度时 softmax 过尖
        """
        Q = self.q_proj(self.q_norm(search_feat))    # (B, L_x, D)
        K = self.k_proj(self.k_norm(template_feat))  # (B, L_z, D)

        score = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, L_x, L_z)

        # [修复 ⑤] 复用与诊断模块同源的循环一致性数学 (cycle_consistency_diag)，
        # 保证 "融合内部用的 V" 与 "诊断输出的 V" 算法完全一致，只在前端特征 (raw vs 投影) 上有意区分。
        diag, M_s2t = cycle_consistency_diag(score)           # diag: (B, L_x)
        # 融合按 token 加权多模板, 故保留逐 token 的 V (不做 Top-K 聚合)
        V = torch.tanh(self.para_factor * diag).unsqueeze(1)  # (B, 1, L_x)

        # 对齐特征 — 对应 x_leftT = M_r2l @ x_right
        aligned = torch.bmm(M_s2t, template_feat)             # (B, L_x, D)

        return V, aligned

    def forward(self, enc_opt, search_feat, num_patch_x, num_patch_z):
        """
        对应 PMIFM.forward 的完整流程

        Args:
            enc_opt:     (B, L_x + N*L_z, D) — Encoder 输出 (Neck 之前! 模板独立)
            search_feat: (B, L_x, D) — Neck 输出的搜索特征 (xs, 已增强)
            num_patch_x: int — 搜索 token 数 (256)
            num_patch_z: int — 每个模板 token 数 (64)
        Returns:
            enhanced_search: (B, L_x, D)
        """
        identity = search_feat  # 对应 PMIFM 的 identity_views[i]

        # === deep_feature ===
        # 注: PMIFM 原始代码 L207: rb_x = self.deep_feature(x)  ← 无额外残差!
        # ResidualBlock 内部已有 skip connection, 不需要再 + search_feat
        # [修复 ④] 搜索/模板使用各自独立的 deep_feature
        rb_search = self.deep_feature_search(search_feat)

        # 从 enc_opt (Neck 之前!) 切出模板特征 — 模板之间尚未混合
        template_tokens = enc_opt[:, num_patch_x:]  # (B, N*L_z, D)
        N = template_tokens.shape[1] // num_patch_z
        template_list = [template_tokens[:, t*num_patch_z:(t+1)*num_patch_z]
                         for t in range(N)]

        # 对模板过模板专用 deep_feature (同样无额外残差)
        rb_templates = [self.deep_feature_template(t) for t in template_list]

        # === 对应 PMIFM 的 for i, rb_cur / for j, rb_oth 循环 ===
        V_list = []
        aligned_list = []
        for rb_t in rb_templates:
            V_t, aligned_t = self.compute_parallax(rb_search, rb_t)
            V_list.append(V_t)         # (B, 1, L_x)
            aligned_list.append(aligned_t)  # (B, L_x, D)

        if len(aligned_list) == 0:
            return identity

        # === 对应 PMIFM 的 softmax 加权 ===
        V = torch.cat(V_list, dim=1) * self.scale_factor  # (B, N, L_x)
        weights = F.softmax(V, dim=1)                      # (B, N, L_x)

        agg = torch.zeros_like(aligned_list[0])
        for k, aligned in enumerate(aligned_list):
            w = weights[:, k:k+1, :].transpose(1, 2)       # (B, L_x, 1)
            agg = agg + aligned * w                         # (B, L_x, D)

        # === 对应 PMIFM 的 self.attn + self.fusion ===
        agg, _ = self.cross_attn(rb_search, agg, agg)       # Q=search, K/V=agg
        # [修复 ③] gamma (初值 0) 门控残差: 起点 = identity (baseline)，训练只学增量
        fused = identity + self.gamma * self.norm(self.fusion(
            torch.cat([rb_search, agg], dim=-1)
        ))

        return fused


def build_decoder(cfg, encoder):

    num_channels_enc = encoder.num_channels
    stride = cfg.MODEL.ENCODER.STRIDE
    if cfg.MODEL.DECODER.TYPE == "MLP":
        in_channel = num_channels_enc
        hidden_dim = cfg.MODEL.DECODER.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        mlp_head = MLPPredictor(inplanes=in_channel, channel=hidden_dim,
                                feat_sz=feat_sz, stride=stride)
        return mlp_head
    elif "CORNER" in cfg.MODEL.DECODER.TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD.TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    elif cfg.MODEL.DECODER.TYPE == "CENTER":
        in_channel = num_channels_enc
        out_channel = cfg.MODEL.DECODER.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                      feat_sz=feat_sz, stride=stride)
        return center_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)
