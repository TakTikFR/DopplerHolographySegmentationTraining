import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utils
# -------------------------

def _ensure_shape(pred, targ):
    if targ.ndim == 3:
        targ = targ.unsqueeze(1)
    return pred.float(), targ.float()


def soft_skeletonize(img, thresh_width=10):
    for _ in range(thresh_width):
        min_pool = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
        img = torch.relu(img - torch.relu(img - min_pool))
    return img


# -------------------------
# Base losses
# -------------------------

class BCELossFlat:
    """Multi-label BCE loss (replacement for CE)"""
    def __init__(self, pos_weight=None):
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def __call__(self, pred, targ):
        pred, targ = _ensure_shape(pred, targ)
        return self.loss_func(pred, targ)

    def activation(self, x): return torch.sigmoid(x)
    def decodes(self, x): return (torch.sigmoid(x) > 0.5).float()


class DiceLoss:
    """Multi-label Dice loss"""
    def __init__(self, smooth=1e-6):
        self.smooth = smooth

    def __call__(self, pred, targ):
        pred, targ = _ensure_shape(pred, targ)

        probs = torch.sigmoid(pred)

        intersection = (probs * targ).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targ.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class FocalLoss:
    """Multi-label focal loss"""
    def __init__(self, gamma=2.0):
        self.gamma = gamma

    def __call__(self, pred, targ):
        pred, targ = _ensure_shape(pred, targ)

        bce = F.binary_cross_entropy_with_logits(pred, targ, reduction='none')
        pt = torch.exp(-bce)

        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()
    
# -------------------------
# Combined losses
# -------------------------

class CEDiceLoss:
    def __init__(self, smooth=1., alpha=1., pos_weight=None):
        self.bce = BCELossFlat(pos_weight)
        self.dice = DiceLoss(smooth)
        self.alpha = alpha

    def __call__(self, pred, targ):
        return self.bce(pred, targ) + self.alpha * self.dice(pred, targ)

    def activation(self, x): return torch.sigmoid(x)
    def decodes(self, x): return (torch.sigmoid(x) > 0.5).float()

class FocalDiceLoss:
    def __init__(self, smooth=1., alpha=1., gamma=2.0):
        self.focal = FocalLoss(gamma)
        self.dice = DiceLoss(smooth)
        self.alpha = alpha

    def __call__(self, pred, targ):
        return self.focal(pred, targ) + self.alpha * self.dice(pred, targ)

    def activation(self, x): return torch.sigmoid(x)
    def decodes(self, x): return (torch.sigmoid(x) > 0.5).float()

class CLDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, cl_weight=0.5, dice_weight=0.3, bce_weight=0.2, pos_weight=None):
        super().__init__()
        self.smooth = smooth
        self.cl_weight = cl_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        logits, targets = _ensure_shape(logits, targets)

        probs = torch.sigmoid(logits)

        # BCE
        bce = self.bce(logits, targets)

        # Dice
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1 - ((2 * intersection + self.smooth) / (union + self.smooth)).mean()

        # clDice
        pred_skel = soft_skeletonize(probs)
        gt_skel = soft_skeletonize(targets)

        tprec = (pred_skel * targets).sum(dim=(2, 3)) / (pred_skel.sum(dim=(2, 3)) + self.smooth)
        tsens = (gt_skel * probs).sum(dim=(2, 3)) / (gt_skel.sum(dim=(2, 3)) + self.smooth)

        cl_dice = 1 - (2 * tprec * tsens / (tprec + tsens + self.smooth)).mean()

        return (
            self.cl_weight * cl_dice +
            self.dice_weight * dice_loss +
            self.bce_weight * bce
        )