import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function, get_soft_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.utils import convert_state_dict

from validate_student import validate
from tensorboardX import SummaryWriter
from types import SimpleNamespace

from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

def enable_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

def disable_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.eval()

def load_teacher_model(teacher_cfg, teacher_model_path, n_classes, device):
    model = get_model(teacher_cfg["model"], n_classes).to(device)
    state = convert_state_dict(torch.load(teacher_model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

def load_teacher_ensemble(teacher_cfg, cfg, n_classes, device):
    model_file_name = "{}_{}_best_model.pkl".format(teacher_cfg["model"]["arch"], teacher_cfg["data"]["dataset"])
    paths = [os.path.join(cfg['training']['teacher_ensemble_folder'], str(i), model_file_name)\
             for i in range(int(cfg["training"]["n_sample"]))]
    ensemble= []
    for path in paths:
        ensemble.append(load_teacher_model(teacher_cfg, path, n_classes, device))
    return ensemble

def sample_from_teacher(teacher_model, input, n_sample=5):
    assert n_sample > 0
    #monte carlo sampling teacher's preedictions
    #return an output of [n_sample*batch_size, w, h] and expanded input
    teacher_model.apply(enable_dropout)
    all_samples = []
    for i in range(n_sample):
        all_samples.append(teacher_model(input).detach())
    #all_samples = torch.cat(all_samples, 0).to(input.device)
    teacher_model.apply(disable_dropout)
    return all_samples

def sample_from_teacher_ensemble(teacher_ensemble, input, n_sample=5):
    #return an output of [n_sample*batch_size, w, h] and expanded input
    assert isinstance(teacher_ensemble, list) == True, \
        "input 'model' needs to be a list for ensemble mode"
    assert n_sample <= len(teacher_ensemble)
    all_samples = []
    for i in range(n_sample):
        teacher_model = teacher_ensemble[i]
        all_samples.append(teacher_model(input).detach())
    #all_samples = torch.cat(all_samples, 0).to(input.device)
    return all_samples

# def expand_output(output, n_sample=5):
#     assert n_sample > 0
#     #copy the output n_sample times
#     #return an output of [n_sample*batch_size, w, h]
#     all_output = []
#     for i in range(n_sample):
#         all_output.append(output.clone())
#     all_output = torch.cat(all_output).to(output.device)
#     return all_output

def calculate_mc_statistics(teacher_model, input, n_sample=5):
    # calculate mc mean and var in a memory effecient way
    assert n_sample > 0
    teacher_model.apply(enable_dropout)
    mean = teacher_model(input).detach()
    running_s = torch.zeros(mean.size(),  dtype=torch.float32).to(mean.device)

    for i in range(n_sample - 1):
        k = i + 2
        new_sample = teacher_model.predict(input).detach()
        new_mean  =  mean + (new_sample - mean)/k
        running_s +=  (new_sample - new_mean) * (new_sample - mean)
        mean = new_mean
    if n_sample > 1:
        var = running_s/(n_sample - 1)
    teacher_model.apply(disable_dropout)
    return mean, var

def train(teacher_cfg, student_cfg, writer, logger):

    # Setup seeds
    torch.manual_seed(student_cfg.get("seed", 1337))
    torch.cuda.manual_seed(student_cfg.get("seed", 1337))
    np.random.seed(student_cfg.get("seed", 1337))
    random.seed(student_cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = student_cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(teacher_cfg["data"]["dataset"])
    data_path = teacher_cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=teacher_cfg["data"]["train_split"],
        img_size=(teacher_cfg["data"]["img_rows"], teacher_cfg["data"]["img_cols"]),
        augmentations=data_aug,
    )

    ignore_index = []
    if "ignore_index" in teacher_cfg["data"]:
        ignore_index = teacher_cfg["data"]["ignore_index"]

    if 'output_ignored_cls' in teacher_cfg['training'] and (teacher_cfg["training"]['output_ignored_cls']==True):
        #some model will still output the probability of ignored class
        #tailor for segnet -> sunrgbd with 38 classes (class 0 ignored)
        n_classes = t_loader.n_classes
    else:
        n_classes = t_loader.n_classes - len(ignore_index)

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=teacher_cfg["data"]["val_split"],
        img_size=(teacher_cfg["data"]["img_rows"], teacher_cfg["data"]["img_cols"]),
    )

    trainloader = data.DataLoader(
        t_loader,
        batch_size=student_cfg["training"]["batch_size"],
        num_workers=student_cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader, batch_size=student_cfg["training"]["batch_size"], num_workers=student_cfg["training"]["n_workers"]
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes, ignore_index=ignore_index[0])
    # Setup Model
    student_model = get_model(student_cfg["model"], n_classes).to(device)
    if args.mode == "mc":
        teacher_model = load_teacher_model(teacher_cfg, student_cfg['training']['teacher_model_path'], n_classes, device)
        print("teacher model loaded from: {}".format(student_cfg['training']['teacher_model_path']))
    elif args.mode == "ensemble":
        teacher_model = load_teacher_ensemble(teacher_cfg, student_cfg, n_classes, device)
        print("teacher ensemble loaded from: {}".format(student_cfg['training']['teacher_ensemble_folder']))
    else:
        raise NotImplementedError("unrecogized mode")

    if ("use_teacher_weights" in student_cfg["training"]) and (student_cfg["training"]["use_teacher_weights"]):
        print("student model using teacher weights")
        if args.mode == "mc":
            student_model.load_state_dict(teacher_model.state_dict(), strict=False)
        elif args.mode == "ensemble":
            #load weights from the first ensemeble model
            student_model.load_state_dict(teacher_model[0].state_dict(), strict=False)

    student_model = torch.nn.DataParallel(student_model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(student_cfg)
    optimizer_params = {k: v for k, v in student_cfg["training"]["optimizer"].items() if k != "name"}
    optimizer = optimizer_cls(student_model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, student_cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(student_cfg)
    soft_loss_fn = get_soft_loss_function(student_cfg)

    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if student_cfg["training"]["resume"] is not None:
        if os.path.isfile(student_cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(student_cfg["training"]["resume"])
            )
            checkpoint = torch.load(student_cfg["training"]["resume"])
            student_model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    student_cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(student_cfg["training"]["resume"]))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True
    best_iter = 0

    n_sample = student_cfg["training"]["n_sample"]
    gt_ratio = student_cfg["training"]["gt_ratio"]
    
    while i <= student_cfg["training"]["train_iters"] and flag:
        for (images, labels) in trainloader:
            i += 1
            start_ts = time.time()
            student_model.train()
            batch_size = images.size()[0]
            images = images.to(device)
            gt_labels = labels.to(device)

            with torch.no_grad():
                if args.mode == "mc":
                    soft_labels = sample_from_teacher(teacher_model, images, n_sample=n_sample)
                elif args.mode == "ensemble":
                    #here teacher model is a list of loaded models
                    soft_labels = sample_from_teacher_ensemble(teacher_model, images, n_sample=n_sample)
            
            optimizer.zero_grad()
            pred_mean, pred_logvar = student_model(images) 
            
            # pred_mean = expand_output(pred_mean, n_sample=n_sample)
            # pred_logvar = expand_output(pred_logvar, n_sample=n_sample)
            nll_loss = 0
            for soft_label in soft_labels:
                nll_loss += soft_loss_fn(pred_mean=pred_mean, pred_logvar=pred_logvar, soft_target=soft_label, gt_target=gt_labels, ignore_index=ignore_index[0])
            
            #gt_loss = loss_fn(input=pred_mean[:batch_size], target=gt_labels, ignore_index=ignore_index[0])
            gt_loss = loss_fn(input=pred_mean, target=gt_labels, ignore_index=ignore_index[0])
            nll_loss /= float(n_sample)
            loss = nll_loss + gt_ratio * gt_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % student_cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  GT Loss: {:.4f} NLL Loss {:.4f} Total Loss {:.4f} Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    student_cfg["training"]["train_iters"],
                    gt_loss.item(),
                    nll_loss.item(),
                    loss.item(),
                    time_meter.avg / student_cfg["training"]["batch_size"],
                )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % student_cfg["training"]["val_interval"] == 0 or (i + 1) == student_cfg["training"][
                "train_iters"
            ]:
                student_model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        pred_mean, pred_logvar = student_model(images_val)
                        val_loss = loss_fn(input=pred_mean, target=labels_val, ignore_index=ignore_index[0])

                        pred = pred_mean.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    best_iter = i+1
                    state = {
                        "epoch": i + 1,
                        "model_state": student_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(student_cfg["model"]["arch"], student_cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)

            if (i + 1) == student_cfg["training"]["train_iters"]:
                flag = False
                print("best iteration: {}".format(best_iter))
                break
    return save_path          

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--student_cfg",
        nargs="?",
        type=str,
        help="config file for student model",
    )

    parser.add_argument(
        "--mode",
        "-m",
        nargs="?",
        type=str,
        default= "mc",
        help="teacher mode",
    )
    parser.add_argument(
        "--test",
        '-t', 
        dest='test', 
        action='store_true'
    )
    args = parser.parse_args()
    with open(args.student_cfg) as fp:
        student_cfg = yaml.load(fp)
    
    if args.mode == "mc":
        teacher_run_folder = student_cfg['training']['teacher_run_folder']
    elif args.mode == "ensemble":
        teacher_run_folder = student_cfg['training']['teacher_ensemble_folder']
    else:
        raise NotImplementedError("unrecognized mode")

    pf, run_id = os.path.split(teacher_run_folder)
    _, dataset = os.path.split(pf)
    teacher_config_path = os.path.join(teacher_run_folder, dataset + ".yml")
    with open(teacher_config_path) as fp:
        teacher_cfg = yaml.load(fp)

    run_id = int(run_id[:5])
    if args.test:
        student_run_id = 1
    else:
        student_run_id = random.randint(1, 100000)
    #student_run_id = 998
    logdir = os.path.join(teacher_run_folder, "student_"+str(student_run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.student_cfg, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    saved_model_path = train(teacher_cfg, student_cfg, writer, logger)
    val_args = SimpleNamespace(config=args.student_cfg,
                               model_path=saved_model_path, 
                               propagate_mode="sample",
                               measure_time=True,
                               save_results=True,
                               save_results_path=None)
    validate(student_cfg, val_args)