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
from ptsemseg.inference_utils import propogate_logit_uncertainty, mc_inference

import pickle
import os
from functools import partial

pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
torch.backends.cudnn.benchmark = True

def inference_teacher_model(model, images, n_samples=50):
    """
    inference on one image (batch_size =1)
    """
    #monte carlo inference on teacher 
    pred_mean, pred_var_sm, all_sm_output = mc_inference(model, images)
    softmax_output = pred_mean.cpu().numpy().transpose(1,2,0)
    pred = pred_mean.data.max(0)[1].cpu().numpy()
    softmax_var_mc = pred_var_sm.data.cpu().numpy().transpose(1,2,0)
    
    #calculate entropy of teacher 
    all_sm_output = all_sm_output.data.cpu().numpy().transpose(0,2,3,1)
    avg_entropy = np.mean(-np.einsum('nijk, nijl -> nij', all_sm_output, np.log(all_sm_output+1e-15))/all_sm_output.shape[3], axis = 0)
    del all_sm_output
    return pred,softmax_output, softmax_var_mc, avg_entropy 

def calculate_teacher_uncertainty(softmax_output, softmax_var, avg_entropy,method="var_std"):
    """
    given a mc variance for softmax output calculate the aggregate variance per image
    """
    if method == "var_std":
        uncertainty = np.sqrt(np.einsum('ijk, ijk -> ij', softmax_var, softmax_var))
    elif method == "entropy":
        uncertainty = -np.einsum('ijk, ijl -> ij', softmax_var, np.log(softmax_var+1e-15))
    elif method == "mutual_information":
        entropy_x = -np.einsum('ijk, ijl -> ij', softmax_var, np.log(softmax_var+1e-15))
        avg_entropy_xi = avg_entropy
        uncertainty =  entropy_x  - avg_entropy_xi
    elif method =="lc":
        uncertainty = 1- np.max(softmax_output, axis = 2)
    else:
        raise NotImplementedError("haven't implemented this aggregation method")
    return uncertainty

def validate(cfg, args):

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

    n_classes = loader.n_classes - len(ignore_index)

    valloader = data.DataLoader(loader, batch_size=1, num_workers=8)
    running_metrics = runningScore(n_classes, ignore_index=ignore_index[0])
    #setting up uncertainty metrics
    #uncertainty_metrics = ["var_std", "lc", "entropy", "mutual_information"]
    uncertainty_metrics = ["lc"]
    running_uncertainty_metrics = {}
    for mt in uncertainty_metrics:
        if mt == "lc":
            scale = False
        else:
            scale = True
        running_uncertainty_metrics[mt] = runningUncertaintyScore(n_classes, ignore_index=ignore_index[0], name=mt, scale_uncertainty=scale)
    # Setup Model

    model = get_model(cfg["model"], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(valloader):
        start_time = timeit.default_timer()
        images = images.to(device)

        # if args.eval_flip:
        #     pred_mean, pred_logvar = model(images)
        #     outputs = pred_mean

        #     # Flip images in numpy (not support in tensor)
        #     outputs = outputs.data.cpu().numpy()
        #     flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
        #     flipped_images = torch.from_numpy(flipped_images).float().to(device)
            
        #     pred_mean_flipped, pred_logvar_flipped  = model(flipped_images)
        #     outputs_flipped = pred_mean_flipped
            
        #     outputs_flipped = outputs_flipped.data.cpu().numpy()
        #     outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0

        #     pred = np.argmax(outputs, axis=1)
        pred, softmax_output, softmax_var_mc, avg_entropy = inference_teacher_model(model, images)
        uncertainty = {mt: np.expand_dims(calculate_teacher_uncertainty(softmax_output, softmax_var_mc,\
                        avg_entropy, method=mt), axis=0) for mt in uncertainty_metrics}

        pred = np.expand_dims(pred, axis=0)
        softmax_output = np.expand_dims(softmax_output, axis=0)
        # pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.numpy()

        if args.measure_time:
            elapsed_time = timeit.default_timer() - start_time
            print(
                "Inference time \
                  (iter {0:5d}): {1:3.5f} fps".format(
                    i + 1, pred.shape[0] / elapsed_time
                )
            )
        running_metrics.update(gt, pred)
        for mt in running_uncertainty_metrics:
            uc = uncertainty[mt]
            running_uncertainty_metrics[mt].update(gt, pred, softmax_output, uc)
    score, class_iou = running_metrics.get_scores()
    
    for k, v in score.items():
        print(k, v)

    for mt in running_uncertainty_metrics:
        for k,v in running_uncertainty_metrics[mt].get_scores().items():
            print(k, v)


    for i in range(n_classes):
        print(i, class_iou[i])


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
    parser.set_defaults(eval_flip=True)

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
