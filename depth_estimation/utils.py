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
    loss_names = ['l1', 'l2']
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
    parser.add_argument('--data', metavar='DATA', default='nyudepthv2',
                        choices=data_names,
                        help='data: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
    parser.add_argument('--datapath', metavar='DATA', default='~/dataset',
                        help='dataset folder path (default: ~/dataset)')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                        help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
    parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                        help='number of sparse depth samples (default: 0)')
    parser.add_argument('--max-depth', default=-1.0, type=float, metavar='D',
                        help='cut-off depth of sparsifier, negative values means infinity (default: inf [m])')
    parser.add_argument('--sparsifier', metavar='SPARSIFIER', default=UniformSampling.name, choices=sparsifier_names,
                        help='sparsifier: ' + ' | '.join(sparsifier_names) + ' (default: ' + UniformSampling.name + ')')
    parser.add_argument('--decoder', '-d', metavar='DECODER', default='upproj', choices=decoder_names,
                        help='decoder: ' + ' | '.join(decoder_names) + ' (default: upproj)')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run (default: 30)')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default 0.01)')
    parser.add_argument('--dropout_p', default=0.2, type=float,
                        metavar='DR', help='dropout rate (default 0.2)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')
    parser.add_argument('-t', '--test', dest='test', action='store_true',
                        help='test or debugg mode')
    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false',
                        help='not to use ImageNet pre-trained weights')
    parser.add_argument('--data_uncertainty', "--du", dest='data_uncertainty', action='store_true',
                        help='using datauncertainty')
    parser.add_argument('--warmup', "--wm", dest='warmup', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--n_ensemble', "--es", dest='n_ensemble', type=int, default=5,
                        help='ensemble size (for train_teacher_ensemble)')
    parser.set_defaults(pretrained=True)
    args = parser.parse_known_args()[0]
    if args.modality == 'rgb' and args.num_samples != 0:
        print("number of samples is forced to be 0 when input modality is rgb")
        args.num_samples = 0
    if args.modality == 'rgb' and args.max_depth != 0.0:
        print("max depth is forced to be 0.0 when input modality is rgb/rgbd")
        args.max_depth = 0.0
    return args

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def adjust_learning_rate(optimizer, epoch, lr_init, max_epoch, warmup=0, gamma=0.9):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    # stages = [7, 18]
    # if epoch <= stages[0]:
    #     lr = lr_init * 0.2
    # elif epoch > stages[0] and epoch <= stages[1]:
    #     lr = lr_init * 0.04
    # else:
    #     lr = lr_init * 0.008

    #use polynomial learning rate decay
    if epoch < warmup:
        lr = lr_init  * float(epoch)/warmup
    # else:
    #     lr = lr_init * (0.2 ** (epoch//10))
    else:
        lr = ((1 - (epoch-warmup-1)/float(max_epoch-warmup) ) ** gamma) * lr_init
    
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

def get_output_directory_teacher(args):
    if args.test:
        output_directory = os.path.join('results', 'test')
    else:
        output_directory = os.path.join('results',
        '{}.sparsifier={}.samples={}.modality={}.arch={}.decoder={}.criterion={}.lr={}.bs={}.pretrained={}.dropout_p={}.data_uncertainty={}'.
        format(args.data, args.sparsifier, args.num_samples, args.modality, \
            args.arch, args.decoder, args.criterion, args.lr, args.batch_size, \
            args.pretrained, args.dropout_p, args.data_uncertainty))
    return output_directory


def get_output_directory_teacher_ensemble(args):
    if args.test:
        output_directory = os.path.join('results', 'test_ensemble')
    else:
        output_directory = os.path.join('results',
        '[ensemble_size={}]{}.sparsifier={}.samples={}.modality={}.arch={}.decoder={}.criterion={}.lr={}.bs={}.pretrained={}.dropout_p={}.data_uncertainty={}'.
        format(args.n_ensemble, args.data, args.sparsifier, args.num_samples, args.modality, \
            args.arch, args.decoder, args.criterion, args.lr, args.batch_size, \
            args.pretrained, args.dropout_p, args.data_uncertainty))
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

def merge_into_row_w_uncertainty(input, depth_target, depth_pred, depth_uncertainty):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0))
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    depth_uncertainty_cpu = np.squeeze(depth_uncertainty.data.cpu().numpy())
    
    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))

    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    depth_uncertainty_col = colored_depthmap(depth_uncertainty_cpu, cmap=cmap2)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col, depth_uncertainty_col])
    
    return img_merge

def merge_into_row_w_data_uncertainty(input, depth_target, depth_pred, depth_model_uncertainty, depth_data_uncertainty):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0))
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    depth_model_uncertainty_cpu = np.squeeze(depth_model_uncertainty.data.cpu().numpy())
    depth_data_uncertainty_cpu = np.squeeze(depth_data_uncertainty.data.cpu().numpy())
    
    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))

    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    # d_min = min(np.min(depth_model_uncertainty_cpu), np.min(depth_data_uncertainty_cpu))
    # d_max = max(np.max(depth_model_uncertainty_cpu), np.max(depth_data_uncertainty_cpu))

    depth_model_uncertainty_col = colored_depthmap(depth_model_uncertainty_cpu, cmap=cmap2)
    depth_data_uncertainty_col = colored_depthmap(depth_data_uncertainty_cpu, cmap=cmap2)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col, depth_model_uncertainty_col, depth_data_uncertainty_col])
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