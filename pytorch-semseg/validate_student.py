import yaml
import torch
import argparse
import timeit
import numpy as np

from torch.utils import data
from torch import nn

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore, runningUncertaintyScore
from ptsemseg.utils import convert_state_dict
from ptsemseg.inference_utils import (propagate_logit_uncertainty, 
                                    propagate_logit_uncertainty_gpu, 
                                    sample_gaussian_logits,
                                    sample_laplace_logits)

import pickle
import os
from functools import partial


pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#orch.backends.cudnn.benchmark = True
N_SAMPLE = 100
UNCERTAINTY = "mutual_information"

def inference_student_model(model, images, propagation_mode="gpu", sampling_dist="gaussian"):
    """
    propagation_mode:
                    gpu -> using pytorch to compute jacobian and var propagation
                    sample -> mc estimate of variance
    """
    
    pred_mean, pred_logvar = model(images)
    pred = pred_mean.data.max(1)[1]
    
    # if propagation_mode == "cpu":
    #     logits_var = np.exp(np.squeeze(pred_logvar.data.cpu().numpy(), axis=0).transpose(1,2,0))
    #     #uncertainty propagation (#need to implement an pytorch version)
    #     softmax_var_propagated = propagate_logit_uncertainty(softmax_output, logits_var)
    if propagation_mode == "gpu":
        softmax_var_propagated, softmax_mean_propagated = propagate_logit_uncertainty_gpu(pred_mean,  pred_logvar)
        #can't estimate average entropy using error propagation
        avg_entropy = None
        #HWC
    elif propagation_mode == "sample":
        n_sample = N_SAMPLE
        with torch.no_grad():
            #softmax_output_mean = nn.Softmax(dim=1)(pred_mean).permute(0,2,3,1)
            #rand tensor shape (n,m, c, h, w) m is the # of samples per data point
            if sampling_dist =="gaussian":
                rand_logistic = sample_gaussian_logits(pred_mean, pred_logvar, n_sample)
            elif sampling_dist == "laplace":
                rand_logistic = sample_laplace_logits(pred_mean, pred_logvar, n_sample)
            else:
                raise NotImplementedError("not implemented")
            softmax_output_sampled = nn.Softmax(dim=2)(rand_logistic).permute(0,1,3,4,2)
            entropy = -torch.sum(softmax_output_sampled * torch.log(softmax_output_sampled + 1e-9), dim=-1)/softmax_output_sampled.size()[-1]
            softmax_var_propagated = torch.var(softmax_output_sampled, dim=0)
            softmax_mean_propagated = torch.mean(softmax_output_sampled, dim=0)
            avg_entropy = torch.mean(entropy, dim=0)

    return pred, softmax_mean_propagated, softmax_var_propagated, avg_entropy

def calculate_student_agg_var(softmax_var, method="l2"):
    """
    given a covariance matrix for softmax output calculate the aggregate variance per image
    """
    if method == "l2":
        softmax_var[softmax_var < 1e-50] =0
        agg_var = np.sqrt(np.einsum('nijkl, nijkl -> nij', softmax_var, softmax_var))
    else:
        raise NotImplementedError("haven't implemented this aggregation method")
    return agg_var

def calculate_student_agg_var1d(softmax_var, method="l2"):
    """
    given a diagonal of covariance matrix for softmax output calculate the aggregate variance per image
    """
    if method == "l2":
        softmax_var[softmax_var < 1e-50] =0
        agg_var = np.mean(np.sqrt(softmax_var), axis= -1)
        #agg_var = np.sqrt(np.sum(softmax_var * softmax_var, axis=-1))
    else:
        raise NotImplementedError("haven't implemented this aggregation method")
    return agg_var

def calculate_student_mutual_information1d(softmax_output, avg_entropy):
    """
    given a avg entropy and softmax calculate mutual information 
    """
    mean_entropy = -np.sum(softmax_output * np.log(softmax_output+1e-9), axis=-1)/softmax_output.shape[-1]
    return mean_entropy-avg_entropy


def validate(cfg, args):
    global N_SAMPLE
    global UNCERTAINTY

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
        is_transform=True,
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    ignore_index = []
    if "ignore_index" in cfg["data"]:
        ignore_index = cfg["data"]["ignore_index"]

    if 'output_ignored_cls' in cfg['training'] and (cfg["training"]['output_ignored_cls']==True):
        #some model will still output the probability of ignored class
        #tailor for segnet -> sunrgbd with 38 classes (class 0 ignored)
        n_classes = loader.n_classes
    else:
        n_classes = loader.n_classes - len(ignore_index)

    valloader = data.DataLoader(loader, batch_size=1, num_workers=8)
    running_metrics = runningScore(n_classes, ignore_index=ignore_index[0])
    running_uncertainty_metrics = runningUncertaintyScore(n_classes, ignore_index=ignore_index[0])

    # Setup Model

    model = get_model(cfg["model"], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    sampling_dist = None
    if cfg["training"]["soft_loss"]["name"] == "nll_guassian_loss":
        sampling_dist = "gaussian"
    elif cfg["training"]["soft_loss"]["name"] ==  "nll_laplace_loss":
        sampling_dist = "laplace"

    for i, (images, labels) in enumerate(valloader):
        start_time = timeit.default_timer()

        images = images.to(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:

        start.record()
        
        pred, \
        softmax_output, \
        softmax_var_propagated, \
        avg_entropy = inference_student_model(model, images, propagation_mode=args.propagate_mode, sampling_dist=sampling_dist)

        end.record()
        torch.cuda.synchronize()
        # cuda_time = sum([evt.cuda_time_total for evt in prof.key_averages()]) / 1000
        # cpu_time = prof.self_cpu_time_total / 1000
        
        if args.measure_time:
            elapsed_time = start.elapsed_time(end)
            print(
                "Inference time \
                  (iter {0:5d}):\t\t\t\ttime total {1:.2f}ms".format(
                    i + 1, elapsed_time,
                )
            )
        pred = pred.cpu().numpy()
        softmax_output = softmax_output.cpu().numpy()
        softmax_var_propagated = softmax_var_propagated.data.cpu().numpy()
       

        if args.propagate_mode == "gpu":
            #error propagation: using  norm of the covariance matrix as uncertainty
            UNCERTAINTY = "var_norm"
            agg_uncertainty = calculate_student_agg_var(softmax_var_propagated)
        elif args.propagate_mode == "sample":
            if UNCERTAINTY == "var_std":
                #error propagation: using mutual information as uncertainty 
                agg_uncertainty = calculate_student_agg_var1d(softmax_var_propagated)
            elif UNCERTAINTY == "entropy":
                agg_uncertainty = avg_entropy
            #avg_entropy = avg_entropy.cpu().numpy()
            elif UNCERTAINTY == "mutual_information":
                avg_entropy = avg_entropy.cpu().numpy()
                agg_uncertainty =  calculate_student_mutual_information1d(softmax_output, avg_entropy)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.numpy()

        running_metrics.update(gt, pred)
        running_uncertainty_metrics.update(gt, pred, softmax_output, agg_uncertainty)

    score, class_iou = running_metrics.get_scores()
    score_uncertainty = running_uncertainty_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for k, v in score_uncertainty.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])

    if args.save_results:
        if args.save_results_path is not None:
            result_file = args.save_results_path
        elif 'teacher_ensemble_folder' in cfg['training']: 
            result_file = os.path.join(cfg['training']['teacher_ensemble_folder'], "results.txt")
        elif 'teacher_run_folder' in cfg['training']: 
            result_file = os.path.join(cfg['training']['teacher_run_folder'], "results.txt")
        else:
            raise ValueError("no teacher folder specified in student cfg")
        
        #append to the same file if result file is already created
        if os.path.exists(result_file):
            write_mode = "a"
        else:
            write_mode = "w"
        
        with open(result_file, write_mode) as f:
            string = "\nstudent_folder: {} gt_ratio={} n_sample={} use_teacher_weights={} propagation_method={} uncertainty={}"\
                        .format(os.path.dirname(args.model_path), 
                            cfg['training']['gt_ratio'],
                            cfg['training']['n_sample'],
                            cfg['training']['use_teacher_weights'],
                            args.propagate_mode,
                            UNCERTAINTY)
            print(string, file=f)
            print(" ", file=f)
            for k, v in score.items():
                print(k, v, file=f)
            for k, v in score_uncertainty.items():
                print(k, v, file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--propagate_mode",
        "-p",
        type=str,
        default="gpu",
        help="mode to propagate the variance from logits space to softmax",
    )

    parser.add_argument(
        "--save_results",
        "-s",
        action="store_true",
        help="save results to a summary file",
    )

    parser.add_argument(
        "--save_results_path",
        help="path for a summary file to save results",
    )
    # parser.add_argument(
    #     "--eval_flip",
    #     dest="eval_flip",
    #     action="store_true",
    #     help="Enable evaluation with flipped image |\
    #                           True by default",
    # )
    # parser.add_argument(
    #     "--no-eval_flip",
    #     dest="eval_flip",
    #     action="store_false",
    #     help="Disable evaluation with flipped image |\
    #                           True by default",
    # )
    #parser.set_defaults(eval_flip=True)

    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
