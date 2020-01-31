import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class GaussianNLLloss(nn.Module):
    def __init__(self):
        super(GaussianNLLloss, self).__init__()

    def forward(self, pred_mean, pred_logvar, target, epsilon=1e-7, mask_zero=True):
        valid_mask = (target>0).detach()
        #assume pred is a tensor with two channels: mean and logvar
        #need to know whether to apply mask
        # pred_mean = pred_mean[valid_mask]
        # pred_logvar = pred_logvar[valid_mask]

        pred_var = torch.exp(pred_logvar) + epsilon
        log_std = 0.5*pred_logvar
        nll = ((target - pred_mean) ** 2) / (2 * pred_var) + log_std + math.log(math.sqrt(2 * math.pi))
        # print(pred_var.min())
        # print(nll.mean())
        if mask_zero:
            nll = nll[valid_mask]
        self.loss = nll.mean()
        return self.loss

class LaplaceNLLloss(nn.Module):
    def __init__(self):
        super(LaplaceNLLloss, self).__init__()

    def forward(self, pred_mean, pred_logvar, target, epsilon=1e-9, mask_zero=True):
        valid_mask = (target>0).detach()
        #assume pred is a tensor with two channels: mean and logvar
        #need to know whether to apply mask
        # pred_mean = pred_mean[valid_mask]
        # pred_logvar = pred_logvar[valid_mask]
        nll = 0.5*pred_logvar + 0.5* math.log(2) + torch.abs(target - pred_mean) /torch.sqrt((0.5 * torch.exp(pred_logvar)))
        #nll = ((target - pred_mean) ** 2) / (2 * pred_var) + log_std + math.log(math.sqrt(2 * math.pi))
        # print(pred_var.min())
        # print(nll.mean())
        if mask_zero:
            nll = nll[valid_mask]
        self.loss = nll.mean()
        return self.loss