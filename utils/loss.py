import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_function(preds, targets):
    smooth = 1e-4
    n = preds.size(0)
    iflat = preds.view(n, -1)
    tflat = targets.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


