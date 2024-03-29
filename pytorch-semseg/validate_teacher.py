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
from ptsemseg.inference_utils import mc_inference, ensemble_inference

import pickle
import os
from functools import partial

pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#torch.backends.cudnn.benchmark = True

def inference_teacher_model(model, images,  mode="mc", data_uncertainty=False, n_samples=25):
    """
    inference on one image (batch_size =1)
    n_samples: number of samples for mc dropout (mc mode only)
    """
    #monte carlo or ensemble inference on teacher 
    if mode == "mc":
        pred_mean, pred_var_sm, all_sm_output = mc_inference(model, images, n_samples=n_samples, data_uncertainty=data_uncertainty)
    elif mode == "ensemble":
        # input needs to be 
        assert isinstance(model, list) == True, "input 'model' needs to be a list for ensemble mode"
        pred_mean, pred_var_sm, all_sm_output = ensemble_inference(model, images, data_uncertainty=data_uncertainty)
    elif mode =="deterministic":
        #assert data_uncertainty, "has to have data uncertainty to evalutate an deterministic model"
        #eqivalent to inference on ensemble of 1 
        pred_mean, pred_var_sm, all_sm_output = ensemble_inference([model], images, data_uncertainty=data_uncertainty, n_logits_sample=50)
    else:
        raise NotImplementedError("unrecognized mode: "+ str(mode))

    return pred_mean, pred_var_sm, all_sm_output

def calculate_teacher_uncertainty(softmax_output, softmax_var, avg_entropy,method="var_std"):
    """
    given a mc variance for softmax output calculate the aggregate variance per image
    """
    if method == "var_std":
        uncertainty = np.mean(np.sqrt(softmax_var), axis= -1)
    elif method == "entropy":
        uncertainty = -np.sum(softmax_output * np.log(softmax_output+1e-9), axis=-1)/softmax_output.shape[-1]
    elif method == "mutual_information":
        entropy_x = -np.sum(softmax_output * np.log(softmax_output+1e-9), axis=-1)/softmax_output.shape[-1]
        uncertainty =  entropy_x  - avg_entropy
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

    if 'output_ignored_cls' in cfg['training'] and (cfg["training"]['output_ignored_cls']==True):
        #some model will still output the probability of ignored class
        #tailor for segnet -> sunrgbd with 38 classes (class 0 ignored)
        n_classes = loader.n_classes
    else:
        n_classes = loader.n_classes - len(ignore_index)

    valloader = data.DataLoader(loader, batch_size=1, num_workers=8)
    running_metrics = runningScore(n_classes, ignore_index=ignore_index[0])
    #setting up uncertainty metrics
    #uncertainty_metrics = ["var_std", "lc", "entropy", "mutual_information"]
    uncertainty_metrics = ["var_std",  "mutual_information"]
    running_uncertainty_metrics = {}
    for mt in uncertainty_metrics:
        running_uncertainty_metrics[mt] = runningUncertaintyScore(n_classes, ignore_index=ignore_index[0], name=mt)
    # Setup Model
    def load_model_from_cfg(cfg, model_path, n_classes, device):
        model = get_model(cfg["model"], n_classes).to(device)
        state = convert_state_dict(torch.load(model_path)["model_state"])
        model.load_state_dict(state)
        model.eval()
        model.to(device)
        return model
    
    data_uncertainty = ("output_var" in cfg["model"]) and (cfg["model"]["output_var"])
    
    if args.mode == "mc" or args.mode =="deterministic":
        model = load_model_from_cfg(cfg, args.model_path, n_classes, device)     
    elif args.mode == "ensemble":
        model_file_name = "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"])
        paths = [os.path.join(args.ensemble_folder, str(i+4), model_file_name) for i in range(int(args.ensemble_size))]
        model = []
        for path in paths:
            model.append(load_model_from_cfg(cfg, path, n_classes, device))
    else:
        raise NotImplementedError("unrecognized mode")

    for i, (images, labels) in enumerate(valloader):
        start_time = timeit.default_timer()
        images = images.to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start.record()
        pred_mean, pred_var_sm, all_sm_output\
            = inference_teacher_model(model, images, mode=args.mode, n_samples=args.n_samples, data_uncertainty=data_uncertainty)
        
        with torch.no_grad():
            all_sm_output = all_sm_output.permute(0,2,3,1)
            avg_entropy = -torch.mean(torch.sum(all_sm_output *torch.log(all_sm_output+1e-9), dim=-1)/all_sm_output.size()[3], dim = 0)
            del all_sm_output
        end.record()

        torch.cuda.synchronize()
        
        #cuda_time = sum([evt.cuda_time_total for evt in prof.key_averages()]) / 1000
        #cpu_time = prof.self_cpu_time_total / 1000

        if args.measure_time:
            elapsed_time = start.elapsed_time(end)
            print(
                "Inference time \
                  (iter {0:5d}):\t\t\t\ttime total {1:.2f}ms".format(
                    i + 1, elapsed_time
                )
            )
        
        softmax_output = pred_mean.cpu().numpy().transpose(1,2,0)
        pred = pred_mean.data.max(0)[1].cpu().numpy()
        softmax_var_mc = pred_var_sm.data.cpu().numpy().transpose(1,2,0)
        
        #calculate entropy of teacher 
        avg_entropy = avg_entropy.cpu().numpy()
        uncertainty = {mt: np.expand_dims(calculate_teacher_uncertainty(softmax_output, softmax_var_mc,\
                        avg_entropy, method=mt), axis=0) for mt in uncertainty_metrics}

        pred = np.expand_dims(pred, axis=0)
        softmax_output = np.expand_dims(softmax_output, axis=0)
        # pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.numpy()
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

    if args.save_results:
        if args.save_results_path is not None:
            result_folder = os.path.dirname(args.model_path)
            result_file = args.save_results_path
        elif args.mode == "ensemble": 
            result_folder = args.ensemble_folder
            result_file = os.path.join(result_folder, "results.txt")
        elif args.mode == "mc" or args.mode == "deterministic": 
            result_folder = os.path.dirname(args.model_path)
            result_file = os.path.join(result_folder , "results.txt")
        else:
            raise ValueError("no supporting this mode")
        
        #append to the same file if result file is already created
        if os.path.exists(result_file):
            write_mode = "a"
        else:
            write_mode = "w"
        
        with open(result_file, write_mode) as f:
            string = "\nteacher\n teacher_folder: {}".format(result_folder)
            print(string, file=f)
            print(" ", file=f)
            for k, v in score.items():
                print(k, v, file=f)
            for mt in running_uncertainty_metrics:
                for k,v in running_uncertainty_metrics[mt].get_scores().items():
                    print(k, v, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--mode",
        "-m",
        nargs="?",
        type=str,
        default="mc",
        help="inference mode for teacher",
    )
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--n_samples",
        nargs="?",
        type=int,
        default=50,
        help="n samples for mc dropout",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="(for mc mode) Path to the saved model ",
    )

    parser.add_argument(
        "--ensemble_folder",
        nargs="?",
        type=str,
        default="runs/camvid_test/91245_ensemble",
        help="(ensemble mode) Path to the saved ensemble model folder",
    )

    parser.add_argument(
        "--ensemble_size",
        "-n",
        nargs="?",
        type=str,
        default="5",
        help="(ensemble mode) number of models in the ensmeble",
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
    #parser.set_defaults(eval_flip=True)

    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    # parser.add_argument(
    #     "--no-measure_time",
    #     dest="measure_time",
    #     action="store_false",
    #     help="Disable evaluation with time (fps) measurement |\
    #                           True by default",
    # )
    parser.set_defaults(measure_time=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
