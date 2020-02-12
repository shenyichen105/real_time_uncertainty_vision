import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

cmap = plt.cm.viridis
cmap2 = plt.cm.magma

def parse_command():
    model_names = ['resnet18', 'resnet50']
    loss_names = ['gaussian', 'laplace']
    data_names = ['nyudepthv2', 'kitti']
    from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
    sparsifier_names = [x.name for x in [UniformSampling, SimulatedStereo]]
    from models import Decoder
    decoder_names = Decoder.names
    from dataloaders.dataloader import MyDataloader
    modality_names = MyDataloader.modality_names

    import argparse
    parser = argparse.ArgumentParser(description='Sparse-to-Dense')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
    parser.add_argument('--datapath', metavar='DATA', default='~/dataset',
                        help='dataset folder path (default: ~/dataset)')
    parser.add_argument('--decoder', '-d', metavar='DECODER', default='upproj', choices=decoder_names,
                        help='decoder: ' + ' | '.join(decoder_names) + ' (default: upproj)')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                        metavar='LR', help='initial learning rate (default 0.005)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--teacher', '-t', metavar='T', default='results/nyudepthv2.sparsifier=uar.samples=0.modality=rgb.arch=resnet50.decoder=upproj.criterion=l1.lr=0.01.bs=8.pretrained=True.dropout_p=0.2', help='the path to the teacher models folder')
    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false',
                        help='not to use ImageNet pre-trained weights')
    parser.add_argument('--not_use_teacher_weights', dest='use_teacher_weights',action='store_false',
                        help='use teacher weights to start')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')
    parser.add_argument('-n', '--n_sample', default=5, type=int,
                        help='number of teachers predictions to sample per input')
    parser.add_argument('--gr', '--ratio_gt',default=1.0, type=float,
                        help='ratio of ground truth nll loss in the total student loss')
    parser.add_argument('--test', dest='test', action='store_true',
                        help='test or debugg mode')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='laplace', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: laplace)')
    parser.add_argument('--warm_up', "--wm", dest='warm_up', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--mode', "--m", dest='mode', type=str, default='mc',
                        help='teacher_mode: monte carlo or ensemble')
    parser.set_defaults(pretrained=True)
    parser.set_defaults(use_teacher_weights=True)
    args = parser.parse_known_args()[0]
    return args

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_student_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def adjust_learning_rate(optimizer, epoch, lr_init, max_epoch, gamma=0.9, warm_up=0):
    """Sets the learning rate"""
    # stages = [12, 25]
    # if epoch == 1:
    #     lr = 0.00005
    if epoch < warm_up:
        lr = lr_init  * float(epoch)/warm_up
    else:
        lr = ((1 - (epoch-warm_up-1)/float(max_epoch-warm_up)) ** gamma) * lr_init
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # if epoch < warm_up:
    #     lr = lr_init  * float(epoch)/warm_up
    # else:
    #     lr = lr_init * (0.2 ** (np.floor(epoch/10)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("current learning rate: {}".format(lr))

def get_output_directory(args):
    output_directory = os.path.join('results',
        '{}.sparsifier={}.samples={}.modality={}.arch={}.decoder={}.criterion={}.lr={}.bs={}.pretrained={}'.
        format(args.data, args.sparsifier, args.num_samples, args.modality, \
            args.arch, args.decoder, args.criterion, args.lr, args.batch_size, \
            args.pretrained))
    return output_directory

def get_student_output_directory(args):
    teacher_folder = args.teacher
    if args.test:
        output_directory = os.path.join(teacher_folder,'test')
    else:
        output_directory = os.path.join(teacher_folder, 'student.gt_ratio={}.n_sample={}.epochs={}.arch={}.use_teacher_weights={}.criterion={}'.
                format(args.gr, args.n_sample, args.epochs, args.arch, args.use_teacher_weights, args.criterion))
    return output_directory

def colored_depthmap(depth, d_min=None, d_max=None, cmap=cmap):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])
    
    return img_merge

def merge_into_row_only_pred(depth_pred):
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    d_min = min(np.min(depth_pred_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_pred_cpu), np.max(depth_pred_cpu))
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = depth_pred_col
    return img_merge

def merge_into_row_with_confidence(input, depth_target, depth_pred_teacher, depth_pred, conf_teacher, conf_student):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    
    depth_pred_teacher_cpu = np.squeeze(depth_pred_teacher.data.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    
    conf_teacher_cpu = np.squeeze(conf_teacher.data.cpu().numpy())
    conf_student_cpu = np.squeeze(conf_student.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu), np.min(depth_pred_teacher_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu), np.max(depth_pred_teacher_cpu))

    conf_min = min(np.min(conf_teacher_cpu), np.min(conf_student_cpu))
    conf_max =max(np.max(conf_teacher_cpu), np.max(conf_student_cpu))

    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_teacher_col = colored_depthmap(depth_pred_teacher_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    conf_teacher_col = colored_depthmap(conf_teacher_cpu, conf_min, conf_max, cmap=cmap2)
    conf_student_col = colored_depthmap(conf_student_cpu, conf_min, conf_max, cmap=cmap2)
    
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_teacher_col, depth_pred_col, conf_teacher_col, conf_student_col])
    
    return img_merge

def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)

def enable_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

def disable_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.eval()