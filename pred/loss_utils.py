import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi

class FocalTverskyLoss(nn.Module):
    """Abraham & Khan (Focal Tversky) - good for small / slender structures."""
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # flatten
        probs_f = probs.view(-1)
        targets_f = targets.view(-1)
        TP = (probs_f * targets_f).sum()
        FP = (probs_f * (1 - targets_f)).sum()
        FN = ((1 - probs_f) * targets_f).sum()

        tversky = TP / (TP + self.alpha * FN + self.beta * FP + 1e-6)
        return (1 - tversky) ** self.gamma


class BoundaryLoss(nn.Module):
    """
    3D boundary/surface loss based on distance transform (Kervadec et al.).
    This implements a differentiable weighting by the ground-truth signed distance map:
      L = mean( p * d_gt ) where p is predicted probability and d_gt is distance transform (inside/outside).
    NOTE: distance transform computed on CPU (numpy) per-sample in the batch.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        logits: (B,1,D,H,W)
        targets: (B,1,D,H,W) binary in {0,1}
        """
        probs = torch.sigmoid(logits)
        device = logits.device
        b = targets.shape[0]
        total = 0.0
        for i in range(b):
            tgt = targets[i,0].detach().cpu().numpy().astype(np.uint8)
            if tgt.sum() == 0:
                # no foreground - boundary loss zero
                total += 0.0
                continue
            # distance inside and outside
            posdt = ndi.distance_transform_edt(tgt)
            negdt = ndi.distance_transform_edt(1 - tgt)
            # signed distance: negative inside
            sdt = negdt - posdt
            sdt = torch.from_numpy(sdt).to(device).float()
            p = probs[i,0]
            total += torch.mean(p * sdt)
        return total / max(1, b)


class CombinedLoss(nn.Module):
    """Focal-Tversky + BCE + Boundary (weights chosen empirical)."""
    def __init__(self, w_ft=1.0, w_bce=0.7, w_bl=0.1):
        super().__init__()
        self.ft = FocalTverskyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.bl = BoundaryLoss()
        self.w_ft = w_ft
        self.w_bce = w_bce
        self.w_bl = w_bl

    def forward(self, logits, target):
        return self.w_ft * self.ft(logits, target) + self.w_bce * self.bce(logits, target) + self.w_bl * self.bl(logits, target)


def dice_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    union = prob.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)


if __name__ == "__main__":
    # ---------------------------------------------------------
    #  快速跑通测试：验证 CombinedLoss 混合损失函数的计算逻辑
    # ---------------------------------------------------------
    print("测试损失函数 CombinedLoss...")

    # 1. 实例化混合损失函数
    # 内部集成了 Focal-Tversky (擅长细长结构), BCE (基础像素分类), Boundary (边界约束)
    loss_fn = CombinedLoss(w_ft=1.0, w_bce=0.7, w_bl=0.1)

    # 2. 模拟模型输出 (未经过 Sigmoid 的 logits) 和真实标签 (Ground Truth)
    # 形状为 (B, C, D, H, W)
    dummy_logits = torch.randn(2, 1, 32, 32, 32)
    # 真实标签通常是 0 或 1 的二进制张量
    dummy_targets = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()

    # 3. 计算损失值
    loss = loss_fn(dummy_logits, dummy_targets)
    print(f" 计算得到的总损失 (Loss): {loss.item():.4f}")

    # 4. 测试辅助指标：Dice 系数
    dice = dice_from_logits(dummy_logits, dummy_targets)
    print(f" 计算得到的 Dice 系数: {dice.item():.4f}")

    print(" 损失函数测试通过！(注意: BoundaryLoss 在 CPU 上计算距离变换，可能会稍慢)")