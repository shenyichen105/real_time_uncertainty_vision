import torch
import torch.nn.functional as F
import math


def cross_entropy2d(input, target, weight=None, size_average=True, ignore_index=250):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=ignore_index
    )
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None, ignore_index=250):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average, ignore_index=ignore_index)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            target.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average, ignore_index=ignore_index
        )

    return loss

def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)

def nll_gaussian_2d(pred_mean, pred_logvar, soft_target, gt_target, ignore_index, weight=None, size_average=True):
    # mean, var are (b, c, h, w ) tensor
    # mask is (b, h, w) tensor
    # assume diagonal convariance matrix
    
    pred_var = torch.exp(pred_logvar) + 1e-15
    log_std = 0.5* pred_logvar
    nll = ((soft_target - pred_mean) ** 2) / (2 * pred_var) + log_std + math.log(math.sqrt(2 * math.pi))
    if weight is not None:
        weight_tensor = torch.tensor(weight, dtype = torch.float32).to(soft_target.device).view(-1,1,1)
        nll = nll * weight_tensor

    mask = (gt_target != ignore_index)
    mask = torch.repeat_interleave(mask.unsqueeze(1), pred_mean.size()[1], dim=1)
    nll = nll.flatten()[mask.flatten()]
    if size_average:
        loss = nll.mean()
    else:
        loss = nll.sum()
    return loss

def nll_laplace_2d(pred_mean, pred_logvar, soft_target, gt_target, ignore_index, weight=None, size_average=True):
    # mean, var are (b, c, h, w ) tensor
    # mask is (b, h, w) tensor
    # assume diagonal convariance matrix
    # print(pred_mean[0,:,0,0])
    # print(pred_logvar[0,:,0,0])
    
    pred_var = torch.exp(pred_logvar) + 1e-10
    #nll = ((soft_target - pred_mean) ** 2) / (2 * pred_var) + log_std + math.log(math.sqrt(2 * math.pi))
    nll = 0.5*pred_logvar + 0.5* math.log(2) + torch.abs(soft_target - pred_mean) /torch.sqrt((0.5 * torch.exp(pred_logvar)))
    # print(nll[0,:,0,0])
    # print(pred_var[0,:,0,0])
    if weight is not None:
        weight_tensor = torch.tensor(weight, dtype = torch.float32).to(soft_target.device).view(-1,1,1)
        nll = nll * weight_tensor
    
    mask = (gt_target != ignore_index)
    mask = torch.repeat_interleave(mask.unsqueeze(1), pred_mean.size()[1], dim=1)
    nll = nll.flatten()[mask.flatten()]
    if size_average:
        loss = nll.mean()
    else:
        loss = nll.sum()
    return loss
