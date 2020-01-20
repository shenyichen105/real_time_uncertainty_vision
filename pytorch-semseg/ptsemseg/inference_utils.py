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

def sample_from_teacher(teacher_model, input, n_sample=5, data_uncertainty=False):
    assert n_sample > 0
    #monte carlo sampling teacher's preedictions
    #return an output of [n_sample*batch_size, w, h] and expanded input
    teacher_model.apply(enable_dropout)
    all_samples = []
    for i in range(n_sample):
        if data_uncertainty:
            teacher_mean, teacher_logvar = teacher_model(input)
            print('mean pred:')
            print(teacher_mean[0,:,0,0])
            print('variance pred')
            print(teacher_logvar[0,:,0,0])
            teacher_mean, teacher_logvar = teacher_mean.detach(), teacher_logvar.detach()
            teacher_sd = (torch.exp(teacher_logvar) + 1e-7)**0.5
            m = torch.distributions.normal.Normal(torch.zeros(teacher_mean.size()), torch.ones(teacher_mean.size()))
            gaussian_samples = m.sample().to(teacher_mean.device)
            teacher_pred = teacher_mean + teacher_sd*gaussian_samples
        else:
            teacher_pred = teacher_model(input).detach()
        all_samples.append(teacher_pred)    
    all_samples = torch.cat(all_samples, 0).to(input.device)
    teacher_model.apply(disable_dropout)
    return all_samples

def mc_inference(model, input, data_uncertainty=False, n_samples=100):
    output = sample_from_teacher(model, input, n_samples, data_uncertainty)
    softmax_func = nn.Softmax(dim=1)
    sm_output = softmax_func(output)
    pred_mean = torch.mean(sm_output,dim=0)
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
    
def propogate_logit_uncertainty(softmax_output, logits_var):
    """
    cpu (numpy) version 
    """
    jacobian = calculate_softmax_jacobian(softmax_output)
    jacobian_t = np.transpose(jacobian, (0,1,3,2))
    #diag covariance matrix shape: HWCC
    diag_var = np.zeros(jacobian.shape)
    diag_var[:,:, np.arange(jacobian.shape[2]), np.arange(jacobian.shape[2])] = logits_var
    #propogated convariance matrix shape: HWCC
    var_propagated = np.matmul(jacobian_t, np.matmul(diag_var, jacobian))
    return var_propagated 

