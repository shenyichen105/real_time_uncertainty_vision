import torch
import math
import numpy as np
from torch.distributions.normal import Normal

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

def calculate_nll(pred_mu, pred_logvar, target, dist="gaussian", valid_mask=None):
    if dist == "gaussian":
        #pred_var = torch.exp(pred_logvar)
        log_std = 0.5*pred_logvar
        nll = ((target - pred_mu) ** 2) / (2 * pred_var) + log_std + math.log(math.sqrt(2 * math.pi))
    elif dist == "laplace":
        nll = 0.5*pred_logvar + 0.5* math.log(2) + torch.abs(target - pred_mu) /torch.sqrt((0.5 * torch.exp(pred_logvar)))
    if valid_mask is not None:
        nll = nll * valid_mask.float()
    return nll.mean()

def calculate_kl(pred_mu, pred_logvar, valid_mask, teacher_dropout_outputs):
    batch_size = pred_mu.size()[0]
    w = pred_mu.size()[2]
    h = pred_mu.size()[3]
    #need to check the shape of tensors
    teacher_dropout_outputs_mean = teacher_dropout_outputs.view(-1, batch_size, 1, w, h).mean(axis=0)
    teacher_dropout_outputs_std = teacher_dropout_outputs.view(-1, batch_size, 1, w, h).std(axis=0)

    teacher_dropout_outputs_mean = teacher_dropout_outputs_mean[valid_mask]
    teacher_dropout_outputs_std = teacher_dropout_outputs_std[valid_mask]
    pred_mu = pred_mu[valid_mask]
    pred_logvar = pred_logvar[valid_mask]

    var_ratio = (torch.exp(0.5*pred_logvar)/teacher_dropout_outputs_std).pow(2)
    t1 = ((pred_mu - teacher_dropout_outputs_mean)/teacher_dropout_outputs_std).pow(2)
    result = 0.5 * (var_ratio + t1 -1 -var_ratio.log())
    return result.mean()

def calculate_ause(pred_mu, pred_std, target):
    abs_error = (pred_mu - target).abs()
    abs_error  /= abs_error.max()
    ind_oracle = torch.argsort(abs_error)
    ind_uncertainty = torch.argsort(pred_std.flatten())
    
    nth_pixel = torch.arange(abs_error.size()[0], device=abs_error.device).float() + 1
    avg_brier_oracle = torch.cumsum(abs_error[ind_oracle], dim=0)/nth_pixel
    avg_brier_w_uncertainty = torch.cumsum(abs_error[ind_uncertainty], dim=0)/nth_pixel
    x = torch.linspace(0, 1, steps=abs_error.size()[0], device=abs_error.device).float()
    ause = torch.trapz((avg_brier_w_uncertainty - avg_brier_oracle), x=x)
    return ause

def calculate_normal_quantile(target, pred_mu, pred_std):
    dist = Normal(pred_mu, pred_std)
    quantile = dist.cdf(target)
    return quantile

def calculate_ece(pred_mu, pred_std, target, n_bins=30):
    """
    metrics from:
    https://arxiv.org/pdf/1807.00263.pdf
    """
    pred_mu = pred_mu.flatten()
    # pred_logvar = pred_logvar.flatten()
    target = target.flatten()
    # pred_std = torch.exp(pred_logvar*0.5)
    pred_std = pred_std.flatten()
    dist = Normal(pred_mu, pred_std)
    p_target = dist.cdf(target)
    #bining the pixels based on the probability
    thesholds = np.linspace(0,1,n_bins)
    frac = []
    for t in thesholds:
        frac.append(pred_mu[p_target < t].size()[0]/ pred_mu.size()[0])
    ece = np.mean((np.array(frac) - thesholds)**2)
    return ece

class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        valid_mask = target>0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class ResultTeacher(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        #uncertainty
        self.ause, self.ece = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        #uncertainty
        self.ause, self.ece = np.inf, np.inf

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time, ause, ece):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time
        self.ause, self.ece = ause, ece

    def evaluate(self, output, target):
        valid_mask = target>0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

    def mc_evaluate(self, output_mean, output_std, target):
        output_mean = output_mean.flatten()
        output_std = output_std.flatten()
        target = target.flatten()

        valid_mask = target>0
        target = target[valid_mask]
        output_mean= output_mean[valid_mask]
        output_std =output_std[valid_mask]


        abs_diff = (output_mean - target).abs()
        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())

        self.lg10 = float((log10(output_mean) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output_mean / target, target /output_mean)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output_mean
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

        #uncertainty     
        self.ause = calculate_ause(output_mean, output_std, target)
        self.ece = calculate_ece(output_mean, output_std, target)

 
class ResultStudent(object):
    def __init__(self):
        #uncertainty
        self.gt_loss= 0
        self.teacher_loss= 0
        self.kl = 0
        #performance
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        #caliberation and uncertainty 
        self.ause, self.ece = 0,0
        #save the 
        self.output_n

    def set_to_worst(self):
        self.gt_loss, self.teacher_loss, self.kl =  np.inf, np.inf, np.inf
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        self.ause, self.ece = np.inf,np.inf

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time, gt_loss, teacher_loss, kl, ause, ece):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time
        self.gt_loss = gt_loss
        self.kl = kl
        self.teacher_loss = teacher_loss
        self.ause = ause
        self.ece = ece

    def evaluate(self, output_mu, output_logvar, target, teacher_sampled_outputs, teacher_sampled_data_logvar=None, dist="laplace", teacher_loss=None, gt_loss=None, calc_kl=True):
        valid_mask = target>0
        batch_size = output_mu.size()[0]
        
        output_mu_origin = output_mu
        output_logvar_origin = output_logvar

        output_mu = output_mu[valid_mask]
        output_logvar = output_logvar[valid_mask]
        target = target[valid_mask]
        
        #performance 
        output = output_mu
        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

        #uncertainty
        if teacher_loss is not None:
            self.teacher_loss = teacher_loss
        else:
            self.teacher_loss = calculate_nll(output_mu_origin, output_logvar_origin, teacher_sampled_outputs, dist=dist, valid_mask=valid_mask)
        if gt_loss is not None:
            self.gt_loss = gt_loss
        else:
            self.gt_loss = calculate_nll(output_mu, output_logvar, target, dist=dist)
        if calc_kl:
            self.kl = calculate_kl(output_mu_origin, output_logvar_origin, valid_mask, teacher_sampled_outputs)
        
        output_std = torch.exp(0.5 * output_logvar)
        self.ause = calculate_ause(output_mu,  output_std , target)
        self.ece = calculate_ece(output_mu,  output_std , target)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, 
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        return avg

class AverageMeterTeacher(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0
        self.sum_ause = 0
        self.sum_ece = 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time
        self.sum_ause += n*result.ause.item()
        self.sum_ece += n*result.ece

    def average(self):
        avg = ResultTeacher()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, 
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count, self.sum_ause/self.count, self.sum_ece/self.count)
        return avg

class AverageMeterStudent(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0
        self.sum_teacher_loss, self.sum_gt_loss, self.sum_kl = 0,0,0
        self.sum_ause = 0
        self.sum_ece = 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time
        self.sum_teacher_loss += n*result.teacher_loss.item()
        self.sum_gt_loss += n*result.gt_loss.item()
        self.sum_kl += n*result.kl.item()
        self.sum_ause += n*result.ause.item()
        self.sum_ece += n*result.ece

    def average(self):
        avg = ResultStudent()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, 
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count, 
            self.sum_gt_loss/self.count, self.sum_teacher_loss/self.count, self.sum_kl/self.count,
            self.sum_ause/self.count, self.sum_ece/self.count)
        return avg