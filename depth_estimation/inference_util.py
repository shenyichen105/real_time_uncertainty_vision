
import torch

def enable_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

def disable_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.eval()

def generate_mcdropout_predictions(model, input, n_samples):
    #naive implementation, can be slow
    model.apply(enable_dropout) #enable dropout in the inference
    all_pred = []
    middle_rep = model.forward_to_dropout(input)
    for i in range(n_samples):
        pred = model.forward_from_dropout(middle_rep)
        all_pred.append(pred)
    all_pred = torch.stack(all_pred)
    model.apply(disable_dropout)
    return all_pred

def generate_ensemble_predictions(models, input):
    #naive implementation, can be slow
    all_pred = []
    for model in models:
        pred = model(input)
        all_pred.append(pred)
    all_pred = torch.stack(all_pred)
    return all_pred

def generate_mcdropout_predictions_w_var(model, input, n_samples):
    """
    To evaluate teacher with data uncertainty
    """
    model.apply(enable_dropout) #enable dropout in the inerence
    all_pred_mean = []
    all_pred_logvar = []
    middle_rep = model.forward_to_dropout(input)
    for i in range(n_samples):
        pred_mean, pred_logvar = model.forward_from_dropout(middle_rep)
        #pred_mean, pred_logvar = model(input)
        all_pred_mean.append(pred_mean)
        all_pred_logvar.append(pred_logvar)
    all_pred_mean = torch.stack(all_pred_mean)
    all_pred_logvar = torch.stack(all_pred_logvar)
    model.apply(disable_dropout)
    return all_pred_mean, all_pred_logvar

def generate_ensemble_predictions_w_var(models, input):
    """
    To evaluate teacher with data uncertainty
    """
    all_pred_mean = []
    all_pred_logvar = []
    for model in models:
        pred_mean, pred_logvar = model(input)
        all_pred_mean.append(pred_mean)
        all_pred_logvar.append(pred_logvar)
    all_pred_mean = torch.stack(all_pred_mean)
    all_pred_logvar = torch.stack(all_pred_logvar)
    return all_pred_mean, all_pred_logvar

def sample_laplace(pred_mean, pred_logvar, n_samples=5):
    """
    sample logits from a laplace prior 
    rand tensor shape (n,m, c, h, w) n is the # of samples per data point
    """
    rand_tensor = torch.empty(n_samples, pred_mean.size(0),pred_mean.size(1), 
                pred_mean.size(2), pred_mean.size(3), dtype=pred_mean.dtype,
                 device=pred_mean.device).uniform_(-1+1e-10, 1)
    
    scale = torch.exp(0.5*pred_logvar.unsqueeze(0)) * (0.5**0.5)
    samples = pred_mean.unsqueeze(0)  - scale * rand_tensor.sign() * torch.log1p(-rand_tensor.abs())
    return samples

def sample_gaussian(pred_mean, pred_logvar, n_samples=5):
    """
    sample logits from a gaussain prior 
    rand tensor shape (n,m, c, h, w) n is the # of samples per data point
    """
    rand_tensor =  torch.randn(n_samples, pred_mean.size(0), pred_mean.size(1), 
                pred_mean.size(2), pred_mean.size(3), device=pred_mean.device)
    samples = rand_tensor * torch.exp(0.5*pred_logvar.unsqueeze(0)) \
                    + pred_mean.unsqueeze(0)  
    return samples

def sample_laplace_from_var(pred_mean, pred_var, n_samples=5):
    """
    sample logits from a laplace prior 
    rand tensor shape (n,m, c, h, w) n is the # of samples per data point
    """
    finfo = torch.finfo(pred_mean.dtype)
    rand_tensor = torch.empty(n_samples, pred_mean.size(0),pred_mean.size(1), 
                pred_mean.size(2), pred_mean.size(3), dtype=pred_mean.dtype,
                 device=pred_mean.device).uniform_(finfo.eps-1, 1)
    
    scale = torch.sqrt(pred_var.unsqueeze(0) * 0.5)
    samples = pred_mean.unsqueeze(0)  - scale * rand_tensor.sign() * torch.log1p(-rand_tensor.abs())
    return samples

def sample_gaussian_from_var(pred_mean, pred_var, n_samples=5):
    """
    sample logits from a gaussain prior 
    rand tensor shape (n,m, c, h, w) n is the # of samples per data point
    """
    rand_tensor =  torch.randn(n_samples, pred_mean.size(0), pred_mean.size(1), 
                pred_mean.size(2), pred_mean.size(3), device=pred_mean.device)
    samples = rand_tensor * torch.sqrt(pred_var.unsqueeze(0)) + pred_mean.unsqueeze(0)  
    return samples

def sample_from_mcdropout_predictions_w_var(model, input, n_mc_samples, n_data_samples=5, criterion="l1"):
    """
    generating samples from teacher w data uncertainty to train students
    """
    #naive implementation, can be slow
    model.apply(enable_dropout) #enable dropout in the inference
    if criterion == "l1":
        sampling_function = sample_laplace_from_var
    else:
        sampling_function = sample_gaussian_from_var
    all_samples = []
    all_var = []
    middle_rep = model.forward_to_dropout(input)
    for i in range(n_mc_samples):
        #pred_mean, pred_logvar = model(input)
        pred_mean, pred_logvar = model.forward_from_dropout(middle_rep)
        all_var.append(torch.exp(pred_logvar))
        #samples = sampling_function(pred_mean, pred_logvar, n_samples=n_data_samples)
        all_samples.append(pred_mean)
    
    avg_var = torch.stack(all_var).mean(0)
    n, c, h, w = avg_var.size()
    zero_mean = avg_var.new_zeros(avg_var.size())
    sampled_data_uncertainty = sampling_function(zero_mean, avg_var, n_samples=n_data_samples * n_mc_samples)\
                                .view(n_data_samples, -1, c, h, w)
    all_samples = torch.cat(all_samples, 0).unsqueeze(0) + sampled_data_uncertainty
    all_samples = all_samples.view(-1, n, c, h, w)
    #all_samples = torch.cat(all_samples, dim=0)
    model.apply(disable_dropout)
    return all_samples

def sample_from_ensemble_predictions_w_var(models, input, n_data_samples=10, criterion="l1"):
    """
    generating samples from teacher w data uncertainty to train students
    """
    #naive implementation, can be slow
    if criterion == "l1":
        sampling_function = sample_laplace_from_var
    else:
        sampling_function = sample_gaussian_from_var
    all_samples = []
    all_var = []
    for model in models:
        pred_mean, pred_logvar = model(input)
        #samples = sampling_function(pred_mean, pred_logvar, n_samples=n_data_samples)
        all_var.append(torch.exp(pred_logvar)+1e-9)
        all_samples.append(pred_mean)
    
    avg_var = torch.stack(all_var).mean(0)
    n, c, h, w = avg_var.size()
    zero_mean = avg_var.new_zeros(avg_var.size())
    sampled_data_uncertainty = sampling_function(zero_mean, avg_var, n_samples=n_data_samples * len(models))\
                                .view(n_data_samples, -1, c, h, w)
    all_samples = torch.cat(all_samples, 0).unsqueeze(0) + sampled_data_uncertainty
    all_samples = all_samples.view(-1,n, c, h, w)
    return all_samples