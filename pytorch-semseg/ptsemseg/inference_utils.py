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

def sample_from_teacher(teacher_model, input, n_sample=5):
    assert n_sample > 0
    #monte carlo sampling teacher's preedictions
    #return an output of [n_sample*batch_size, w, h] and expanded input
    teacher_model.apply(enable_dropout)
    all_samples = []
    for i in range(n_sample):
        all_samples.append(teacher_model(input).detach())
    all_samples = torch.cat(all_samples, 0).to(input.device)
    teacher_model.apply(disable_dropout)
    return all_samples

def sample_from_teacher_ensemble(teacher_ensemble, input):
    #ensemble teacher's predictions
    #return an output of [n_sample*batch_size, w, h] and expanded input
    assert isinstance(teacher_ensemble, list) == True, "input 'model' needs to be a list for ensemble mode"
    n_models = len(teacher_ensemble)
    all_samples = []
    for i in range(n_models):
        teacher_model = teacher_ensemble[i]
        all_samples.append(teacher_model(input).detach())
    all_samples = torch.cat(all_samples, 0).to(input.device)
    return all_samples

def mc_inference(model, input, n_samples=100):
    #model: a teacher model with dropout layers
    output = sample_from_teacher(model, input, n_samples)
    softmax_func = nn.Softmax(dim=1)
    sm_output = softmax_func(output)
    pred_mean = torch.mean(sm_output,dim=0)
    pred_var_sm = torch.var(sm_output,dim=0)
    return pred_mean, pred_var_sm, sm_output

def ensemble_inference(models, input):
    # models: the ensemble of teacher models with differnent random initialization
    output = sample_from_teacher_ensemble(models, input)
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

