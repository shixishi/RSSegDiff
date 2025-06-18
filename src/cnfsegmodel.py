import torch as th
import torch.nn as nn


class FeatureFusionModule(nn.Module):
    """
    Implements Eq.(9):
        W = softmax(Q K^T / sqrt(C))
        O = MLP(W V) + F_cn
        F = FFN(O) + O
    where Q comes from CN bottleneck, K=V from NN bottleneck.
    All ops are token‑wise; spatial dims preserved.
    """

    def __init__(self, in_c_cn, in_c_nn, proj_c, num_heads, ffn_ratio):
        super().__init__()
        self.proj_c = proj_c
        # 1×1 projections → shared dim C
        self.q_proj = nn.Conv2d(in_c_cn, proj_c, 1)
        self.k_proj = nn.Conv2d(in_c_nn, proj_c, 1)
        self.v_proj = nn.Conv2d(in_c_nn, proj_c, 1)

        self.attn = nn.MultiheadAttention(embed_dim=proj_c,
                                          num_heads=num_heads,
                                          batch_first=True)
        # token MLP, 将 channel 从 proj_c 变为 in_c_cn
        # 用于残差 
        self.mlp = nn.Linear(proj_c, in_c_cn)
        # channel/spatial FFN (conv‑ffn)
        self.ffn = nn.Sequential(
            nn.Linear(in_c_cn, in_c_cn * ffn_ratio),
            nn.GELU(),
            nn.Linear(in_c_cn * ffn_ratio, in_c_cn),
        )
        self.norm_qkv = nn.LayerNorm(proj_c)
        self.norm_o = nn.LayerNorm(in_c_cn)

    def forward(self, cn_feat, nn_feat):
        """Args:
            cn_feat, nn_feat: [B, C, H, W] (may have different C)
           Returns:
            fused feature  [B, C, H, W]
        """
        B, C, H, W = cn_feat.shape
        # project & flatten
        q = self.q_proj(cn_feat).flatten(2).transpose(1, 2)  # [B, HW_cn, C]
        k = self.k_proj(nn_feat).flatten(2).transpose(1, 2)  # [B, HW_nn, C]
        v = self.v_proj(nn_feat).flatten(2).transpose(1, 2)  # [B, HW_nn, C]

        # 2. cross‑attention  (W V)
        attn_out, _ = self.attn(q, k, v)                     # [B, HW_cn, C]

        # 3. O = MLP(WV) + F_cn
        O = self.mlp(self.norm_qkv(attn_out)) + cn_feat.flatten(2).transpose(1, 2)            # residual
        # 4. F = ffn(O) + O   (conv‑ffn)
        F = self.norm_o(self.ffn(O)) + O  # residual
        F = F.transpose(1, 2).view(B, C, H, W)

        return F 


class CnfSegModel(nn.Module):
    """Wraps cond_net + noise_net + fusion."""

    def __init__(self, cond_net, noise_net, ffm, train_flag):
        super().__init__()
        self.cond_net = cond_net
        self.noise_net = noise_net
        self.ffm = ffm
        # 训练模式下，需要返回eps_pred，用于计算loss
        # 推理模式下，只返回logits
        self.train_flag = train_flag  

    def forward(self, img, noise_img, t):
        """
        因为是两个模型，一个是CN，一个是NN，所以需要两个输入，一个是原图，一个是噪声图
        """
        # Noise branch 
        eps_pred, n_mid = self.noise_net(noise_img, t)

        # Condition branch
        hs = []
        h = img
        for block in self.cond_net.input_blocks:
            h = block(h)
            # print(h.shape)
            hs.append(h)
        h = self.cond_net.middle_block(h)

        # 从这里开始进行特征融合模块
        # 更新c_mid
        # print(f"beford ffm c_mid shape: {h.shape}")
        h = self.ffm(h, n_mid)
        # print(f"after ffm c_mid shape: {h.shape}")
        # 然后再跑c_net的decoder
        for block in self.cond_net.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = block(h)
        logits = self.cond_net.out(h)
        if self.train_flag:  # 训练模式下，需要返回eps_pred
            return logits, eps_pred
        else:   # 推理模式下，只返回logits
            return logits