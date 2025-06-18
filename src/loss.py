import torch
import torch.nn.functional as F
# from .lovasz_losses import lovasz_hinge

def compute_loss(logits: torch.Tensor,
                     mask: torch.Tensor,
                     eps_pred: torch.Tensor,
                     eps_target: torch.Tensor
                    ) -> torch.Tensor:
    """
    logits     : [N, C, H, W]  —– CN 输出分割 logits
    mask       : [N, H, W]     —– ground-truth 分割标签
    eps_pred   : [N, 3, H, W]  —– NN 对噪声 ε 的预测
    eps_target : [N, 3, H, W]  —– 真实噪声 ε（标准正态分布）
    返回：
    total_loss, loss_ce, loss_mse
    """
    
    # 目前先不使用 lovasz 损失
    # 1) 语义分割损失 L(ψ)：CrossEntropy
    loss_ce = F.cross_entropy(logits, mask)

    # 2) 扩散噪声预测损失 L(θ)：MSE
    loss_mse = F.mse_loss(eps_pred, eps_target)

    # 3) 几何均衡 (Geometric Loss Strategy, Eq.11)
    #    L_total = sqrt(L_theta) * sqrt(L_psi)
    total_loss = torch.sqrt(loss_ce) * torch.sqrt(loss_mse)

    return total_loss, loss_ce, loss_mse
