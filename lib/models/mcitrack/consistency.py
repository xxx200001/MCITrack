"""
Template-Search Consistency Module
===================================
借鉴 ParaHydra+ 的视差循环一致性 (Parallax Cycle Consistency)，
用于衡量每个模板与搜索区域之间的匹配可靠度。

核心思想 (循环一致性):
  搜索 token i 通过 M_s2t 找到最匹配的模板 token，再通过 M_t2s 映射回来；
  若能回到 i 自身，则该匹配可靠 (diag[i] -> 1)，否则不可靠 (diag[i] -> 0)。
  形式上 diag = (M_s2t @ M_t2s) 的对角线。

针对 "跟踪 vs 立体压缩" 差异所做的适配:
  1. [去均值] 复刻 ParaHydra parallax.py 中 `Q = Q - mean(Q, 匹配维)` 的做法，
     沿 token 维做零均值，消除公共 DC 分量、提升匹配判别力。
     (原迁移版漏掉了这一步，相似度容易被公共分量主导。)
  2. [Top-K 聚合] 立体压缩里整张图每个像素都有真实对应，可以全图平均；
     但跟踪里目标只占搜索区一小块、其余是背景、不可能循环一致。
     若对全部 token 取均值，可靠度会被大量背景 token 稀释、几乎失去区分度。
     因此只聚合一致性最高的 Top-K 个 token (目标所在区域) 求均值，
     得到聚焦目标的可靠度分数 (top-k pooling，弱监督定位中的常用稳健池化)。
  3. [无极线约束] 立体匹配沿极线 (行) 进行，跟踪目标可在任意位置，故用全局 token 注意力。

设计决策:
  - 不使用可学习 Q/K 投影: Neck 输出的特征已有充分语义结构，
    方向 C 的目标是不重新训练即可验证效果。
  - 核心循环一致性数学抽成模块级函数 cycle_consistency_diag / topk_reliability，
    供 TemplateAwareFusion 复用，保证两处的 "循环一致性" 算法完全一致 (避免悄悄分叉)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cycle_consistency_diag(score):
    """
    给定 search<->template 的相似度矩阵 score (B, L_x, L_z)，
    返回每个搜索 token 的循环一致性对角线 diag (B, L_x) 以及 M_s2t。

    diag[i] = Σ_k M_s2t[i,k] * M_t2s[k,i]   (即 (M_s2t @ M_t2s) 的对角线)
    """
    M_s2t = F.softmax(score, dim=-1)                    # (B, L_x, L_z)
    M_t2s = F.softmax(score.transpose(1, 2), dim=-1)    # (B, L_z, L_x)
    # diag[i] = Σ_k M_s2t[i,k] * M_t2s[k,i]
    diag = (M_s2t * M_t2s.transpose(1, 2)).sum(dim=-1)  # (B, L_x)
    return diag, M_s2t


def topk_reliability(diag, topk_ratio, para_factor):
    """
    将逐 token 的循环一致性 diag (B, L) 聚合成单标量可靠度 (B,)。

    只取一致性最高的 Top-K 个 token 求均值 (聚焦目标区, 排除背景稀释)，
    再用 tanh 放大。topk_ratio 不在 (0,1) 时退化为全体均值 (兼容旧行为)。
    """
    L = diag.shape[1]
    if topk_ratio is not None and 0.0 < topk_ratio < 1.0:
        k = max(1, int(round(topk_ratio * L)))
        k = min(k, L)
        topk_vals = torch.topk(diag, k, dim=1).values   # (B, k)
        V_raw = topk_vals.mean(dim=1)                    # (B,)
    else:
        V_raw = diag.mean(dim=1)                         # (B,)
    return torch.tanh(para_factor * V_raw)               # (B,)


class TemplateSearchConsistency(nn.Module):
    """
    计算每个模板与搜索区域之间的循环一致性分数 (无可学习参数)。

    输入:
        search_feat:   (B, L_x, D)  搜索区域特征 (Neck 之前的搜索 tokens)
        template_feat: (B, L_z, D)  单个模板的特征

    输出:
        V:       (B,)        一致性分数 ∈ [0, 1]，越高越可靠 (已做 Top-K 聚焦)
        aligned: (B, L_x, D) 模板->搜索的对齐特征 (用原始模板特征 warp)
    """

    def __init__(self, d_model, para_factor=10.0, topk_ratio=0.25):
        """
        Args:
            d_model:     输入特征维度 (与 Neck 输出一致，如 512)
            para_factor: tanh 放大因子 (默认 10.0，与 ParaHydra 一致)
            topk_ratio:  只聚合一致性最高的 Top-K 比例 token，默认 0.25；
                         <=0 或 >=1 时退化为全体均值。
        """
        super().__init__()
        self.d_model = d_model
        self.para_factor = para_factor
        self.topk_ratio = topk_ratio
        self.scale = d_model ** -0.5

    @staticmethod
    def _center_tokens(feat):
        # 复刻 ParaHydra parallax.py 的去均值: 沿 token 维 (匹配维) 做零均值
        return feat - feat.mean(dim=1, keepdim=True)

    def forward(self, search_feat, template_feat):
        """
        Args:
            search_feat:   (B, L_x, D)
            template_feat: (B, L_z, D)
        Returns:
            V:       (B,)        Top-K 聚焦后的一致性分数 ∈ [0, 1]
            aligned: (B, L_x, D) 模板->搜索的对齐特征
        """
        # 去均值仅用于计算相似度；warp 时仍用原始特征 (与 ParaHydra 一致)
        s_c = self._center_tokens(search_feat)
        t_c = self._center_tokens(template_feat)

        score = torch.bmm(s_c, t_c.transpose(1, 2)) * self.scale  # (B, L_x, L_z)
        diag, M_s2t = cycle_consistency_diag(score)
        V = topk_reliability(diag, self.topk_ratio, self.para_factor)  # (B,)

        aligned = torch.bmm(M_s2t, template_feat)  # (B, L_x, D) 用原始模板特征
        return V, aligned

    def compute_multi_template(self, search_feat, template_feats_list):
        """
        对多个模板逐一计算一致性分数。

        Args:
            search_feat:         (B, L_x, D)
            template_feats_list: list of (B, L_z, D)

        Returns:
            V_scores: (B, N)  每个模板的一致性分数
        """
        V_list = []
        for t_feat in template_feats_list:
            V, _ = self(search_feat, t_feat)  # 用 self() 保留 hooks 兼容性
            V_list.append(V)
        return torch.stack(V_list, dim=1)  # (B, N)
