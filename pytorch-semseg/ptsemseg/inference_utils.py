"""
Utility functions for uncertainty inference
"""
import torch
from torch import nn
import numpy as np

def enable_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

def disable_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.eval()

# def sample_from_predicted_gaussian(mean, logvar, n_samples):
#     sd = torch.exp(0.5*logvar)
#     randn_samples = mean.new(mean.size()).normal_()
#     pred = mean + sd*randn_samples
#     return pred

def sample_gaussian_logits(logits_mean, logits_logvar, n_sample):
    """
    sample logits from a gaussain prior 
    rand tensor shape (n,m, c, h, w) m is the # of samples per data point
    """
    rand_tensor =  torch.randn(n_sample, logits_mean.size(0), logits_mean.size(1), 
                logits_mean.size(2), logits_mean.size(3), device=logits_mean.device)
    sampled_logits = rand_tensor * torch.exp(0.5*logits_logvar.unsqueeze(0)) \
                    + logits_mean.unsqueeze(0)  
    return sampled_logits

def sample_gaussian_logits_from_var(logits_mean, logits_var, n_sample):
    """
    sample logits from a gaussain prior using var instead of logvar
    rand tensor shape (n,m, c, h, w) m is the # of samples per data point
    """
    rand_tensor =  torch.randn(n_sample, logits_mean.size(0), logits_mean.size(1), 
                logits_mean.size(2), logits_mean.size(3), device=logits_mean.device)
    sampled_logits = rand_tensor * (torch.sqrt(logits_var.unsqueeze(0))) \
                    + logits_mean.unsqueeze(0)  
    return sampled_logits

def sample_laplace_logits(logits_mean, logits_logvar, n_sample):
    """
    ------------this is outdated and need work. not up to date---------------

    sample logits from a laplace prior 
    rand tensor shape (n,m, c, h, w) m is the # of samples per data point
    
    """
    rand_tensor = torch.empty(n_sample, logits_mean.size(0),logits_mean.size(1), 
                logits_mean.size(2), logits_mean.size(3), dtype=logits_mean.dtype,
                 device=logits_mean.device).uniform_(-1+1e-10, 1)
    
    scale = torch.sqrt(torch.exp(logits_logvar.unsqueeze(0)) *0.5)
    sampled_logits = logits_mean.unsqueeze(0)  - scale * rand_tensor.sign() * torch.log1p(-rand_tensor.abs())
    return sampled_logits

def sample_from_teacher(teacher_model, input, n_sample=5, data_uncertainty=False, n_logits_sample=5):
    assert n_sample > 0
    #monte carlo sampling teacher's preedictions
    #return an output of [n_sample*batch_size, w, h] and expanded input
    with torch.no_grad():
        teacher_model.apply(enable_dropout)
        all_samples = []
        all_var = []
        down3, indices, unpool_shapes = teacher_model.forward_to_dropout(input)
        for i in range(n_sample):
            if data_uncertainty:
                teacher_mean, teacher_logvar = teacher_model.forward_from_dropout(down3, indices, unpool_shapes)
            #     n, c, h, w = teacher_mean.size()
                #sample data uncertainty from (0, avg_var)
            #     all_samples.append(sampled_logits.view(-1, c, h, w))
                teacher_logvar = torch.clamp(teacher_logvar, max=5)
                all_var.append(torch.exp(teacher_logvar))
                all_samples.append(teacher_mean)
            else:
                all_samples.append(teacher_model.forward_from_dropout(down3, indices, unpool_shapes))
        if data_uncertainty:
            #adding samples sampled from the data uncertainty distribution
            avg_var= torch.stack(all_var).mean(0)
            n, c, h, w = avg_var.size()
            sampled_data_uncertainty = sample_gaussian_logits_from_var(avg_var.new_zeros(avg_var.size()), avg_var, n_sample=n_logits_sample * n_sample).view(n_logits_sample, -1, c, h, w)
            all_samples = torch.cat(all_samples, 0).unsqueeze(0)+ sampled_data_uncertainty
            all_samples = all_samples.view(-1, c, h, w)
        else:  
            all_samples = torch.cat(all_samples, 0)
        teacher_model.apply(disable_dropout)
        return all_samples
    

def sample_from_teacher_ensemble(teacher_ensemble, input, data_uncertainty=False, n_logits_sample=5):
    #ensemble teacher's predictions
    #return an output of [n_sample*batch_size, w, h] and expanded input
    assert isinstance(teacher_ensemble, list) == True, \
    "input 'model' needs to be a list for ensemble mode"
    with torch.no_grad():
        n_models = len(teacher_ensemble)
        all_samples = []
        all_var = []
        for i in range(n_models):
            teacher_model = teacher_ensemble[i]
            if data_uncertainty:
                teacher_mean, teacher_logvar = teacher_model(input)
                # n, c, h, w = teacher_mean.size()
                # sampled_logits = sample_gaussian_logits(teacher_mean, teacher_logvar, n_sample=n_logits_sample)
                # all_samples.append(sampled_logits.view(-1, c, h, w))
                teacher_logvar = torch.clamp(teacher_logvar, max=5)
                all_var.append(torch.exp(teacher_logvar))
                all_samples.append(teacher_mean)
            else:
                all_samples.append(teacher_model(input))
        if data_uncertainty:
            #adding samples sampled from the data uncertainty distribution
            avg_var = torch.stack(all_var).mean(0)
            n, c, h, w = avg_var.size()
            sampled_data_uncertainty = sample_gaussian_logits_from_var(avg_var.new_zeros(avg_var.size()), avg_var, n_sample=n_logits_sample * n_models).view(n_logits_sample, -1, c, h, w)
            all_samples = torch.cat(all_samples, 0).unsqueeze(0)+ sampled_data_uncertainty
            all_samples = all_samples.view(-1, c, h, w)
        else:
            all_samples = torch.cat(all_samples, 0)
        return all_samples

def mc_inference(model, input, n_samples=100, data_uncertainty=False):
    #model: a teacher model with dropout layers
    output = sample_from_teacher(model, input, n_samples, data_uncertainty)
    softmax_func = nn.Softmax(dim=1)
    sm_output = softmax_func(output)
    pred_mean = torch.mean(sm_output,dim=0)
    if sm_output.size()[0] == 1:
        pred_var_sm = pred_mean * 0
    else:
        pred_var_sm = torch.var(sm_output,dim=0)
    return pred_mean, pred_var_sm, sm_output

def ensemble_inference(models, input, data_uncertainty=False, n_logits_sample=5):
    # models: the ensemble of teacher models with differnent random initialization
    output = sample_from_teacher_ensemble(models, input, data_uncertainty, n_logits_sample=n_logits_sample)
    softmax_func = nn.Softmax(dim=1)
    sm_output = softmax_func(output)
    pred_mean = torch.mean(sm_output,dim=0)
    if sm_output.size()[0] == 1:
        pred_var_sm = pred_mean * 0
    else:
        pred_var_sm = torch.var(sm_output,dim=0)
    return pred_mean, pred_var_sm, sm_output

def calculate_softmax_jacobian(softmax_output):
    """
    cpu (numpy) version
    need to implement a pytorch version for faster inference
    """
    outer = np.einsum("ijk,ijl->ijkl", softmax_output, softmax_output)
    #create diagnoilize matrix for softmax arry in each pixel
    diag = np.zeros(outer.shape)
    diag[:,:, np.arange(outer.shape[2]), np.arange(outer.shape[2])] = softmax_output
    #jacobian shape: HWCC
    jacobian = diag - outer
    return jacobian
    
def propagate_logit_uncertainty(softmax_output, logits_var):
    """
    cpu (numpy) version 
    """
    jacobian = calculate_softmax_jacobian(softmax_output)
    jacobian_t = np.transpose(jacobian, (0,1,3,2))
    #diag covariance matrix shape: HWCC
    diag_var = np.zeros(jacobian.shape)
    diag_var[:,:, np.arange(jacobian.shape[2]), np.arange(jacobian.shape[2])] = logits_var
    #propagated convariance matrix shape: HWCC
    var_propagated = np.matmul(jacobian_t, np.matmul(diag_var, jacobian))
    return var_propagated 

def propagate_logit_uncertainty_gpu(logits_mean, logits_logvar):
    """
    gpu (pytorch) version to propagate uncertainty
    """
    with torch.no_grad():
        softmax_func = nn.Softmax(dim=1)
        #NCHW -> NHWC
        softmax_output = softmax_func(logits_mean).permute(0,2,3,1)
        logits_logvar = logits_logvar.permute(0,2,3,1)
        #calulating jacobian matrix
        outer = torch.einsum("nijk,nijl->nijkl",softmax_output, softmax_output)
        diag = torch.zeros(outer.size(), dtype=outer.dtype, device=outer.device)
        diag[:,:,:, torch.arange(0, outer.size()[3], dtype=torch.long),torch.arange(0, outer.size()[3], dtype=torch.long)] = softmax_output
        jacobian = diag - outer
       
        #calculating variance
        jacobian_t = torch.transpose(jacobian, 3,4)
        diag_var = torch.zeros(outer.size(), dtype=outer.dtype, device=outer.device)
        
        diag_var[:,:,:, torch.arange(0, outer.size()[3], dtype=torch.long),torch.arange(0, outer.size()[3], dtype=torch.long)] = torch.exp(logits_logvar)
        var_propagated = torch.matmul(jacobian_t, torch.matmul(diag_var, jacobian))
        return var_propagated, softmax_output
