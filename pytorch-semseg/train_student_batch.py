import os
from train_student import train
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from validate_student import validate
from functools import partial
import shutil
import random
import copy
from ptsemseg.utils import get_logger
import itertools


parser = argparse.ArgumentParser(description="config")
parser.add_argument(
    "--template_cfg",
    "-c",
    nargs="?",
    type=str,
    help="config file for student model",
)
parser.add_argument(
    "--teacher_run_folder",
    "-t",
    nargs="?",
    type=str,
    help="teacher_run_folder",
)
parser.add_argument(
    "--mode",
    "-m",
    nargs="?",
    type=str,
    default= "mc",
    help="teacher mode",
)
#search space


#load the template config
args = parser.parse_args()

if args.mode == "ensemble":
    gt_ratio = [1.0]
    n_sample = [8,5]
    #n_logits_sample = [5]
else:
    gt_ratio = [1.0]
    n_sample = [5,8]
    #gt_ratio = [0.0]
    #n_sample = [2]
    #n_logits_sample = [10]


with open(args.template_cfg) as fp:
    student_cfg = yaml.load(fp)

teacher_run_folder = args.teacher_run_folder

if args.mode == "mc":
    student_cfg['training']['teacher_run_folder'] = teacher_run_folder
elif args.mode == "ensemble":
    student_cfg['training']['teacher_ensemble_folder'] = teacher_run_folder

pf, run_id = os.path.split(teacher_run_folder)
run_id = int(run_id.split("_")[0])
_, cfg_name = os.path.split(pf)
teacher_config_path = os.path.join(teacher_run_folder, cfg_name + ".yml")

with open(teacher_config_path) as fp:
    teacher_cfg = yaml.load(fp)

for gr, ns in list(itertools.product(gt_ratio, n_sample)):
    cur_cfg = copy.deepcopy(student_cfg)
    cur_cfg['training']['gt_ratio'] = float(gr)
    cur_cfg['training']['n_sample'] = int(ns)
    print('gt_ratio ={} n_samples ={}'.format(gr, ns))
    random.seed(a=None)
    student_run_id = random.randint(1, 100000)
    logdir = os.path.join(teacher_run_folder, "test_student_"+str(student_run_id))
    writer = SummaryWriter(log_dir=logdir)
    
    _, student_cfg_name = os.path.split(args.template_cfg)
    cur_cfg_path = os.path.join(logdir, student_cfg_name)
    with open(cur_cfg_path, 'w') as yaml_file:
        yaml.dump(cur_cfg, yaml_file, default_flow_style=False)
    print("RUNDIR: {}".format(logdir))

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    saved_model_path = train(teacher_cfg, cur_cfg, writer, logger, seed=student_run_id, mode=args.mode)
    val_args = SimpleNamespace(config=cur_cfg_path,
                               model_path=saved_model_path, 
                               propagate_mode="gpu",
                               measure_time=True,
                               save_results=True,
                               save_results_path=None)
    validate(cur_cfg, val_args)






