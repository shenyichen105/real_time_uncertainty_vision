import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

from models_dropout import ResNet,  ResNetVar
from metrics import AverageMeter, Result, ResultTeacher, AverageMeterTeacher
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
from inference_util import generate_mcdropout_predictions, generate_mcdropout_predictions_w_var
import criteria
import utils
import utils_student

args = utils.parse_command()
print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time', 'ause', 'ece']
best_result = ResultTeacher()
best_result.set_to_worst()

def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join(os.path.expanduser(args.datapath), args.data, 'train')
    valdir = os.path.join(os.path.expanduser(args.datapath), args.data, 'val')
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    if args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=args.num_samples, max_depth=max_depth)

    if args.data == 'nyudepthv2':
        from dataloaders.nyu_dataloader import NYUDataset

        if not args.evaluate:
            train_dataset = NYUDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier)
        val_dataset = NYUDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)

    elif args.data == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args.evaluate:
            train_dataset = KITTIDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier)
        val_dataset = KITTIDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)

    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader

def perform_evaluation(model_path, mc_samples=25):
    assert os.path.isfile(model_path), \
    "=> no best model found at '{}'".format(model_path)
    print("=> loading best model '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    output_directory = os.path.dirname(model_path)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch'] + 1
    best_result = checkpoint['best_result']
    model = checkpoint['model']
    print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
    _, val_loader = create_data_loaders(args)
    args.evaluate = True
    avg, _= validate(val_loader, model, checkpoint['epoch'], write_to_file=False, n_sample=mc_samples)
    result_txt = os.path.join(output_directory, 'result_mc_{}.txt'.format(mc_samples))
    with open(result_txt, 'w') as txtfile:
        print('\n*\n'
        'mc_samples={mc_samples}\n'
        'epoch={start_epoch}\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'
        'ause={average.ause:.3f}\n'
        'ece={average.ece:.3f}\n'        
        .format(average=avg, start_epoch=start_epoch,mc_samples=mc_samples,time=avg.gpu_time), file=txtfile)
    return
    
def main():
    global args, best_result, output_directory, train_csv, test_csv
    # evaluation mode
    start_epoch = 0
    if args.evaluate:
        perform_evaluation(args.evaluate)
        return
    # optionally resume from a checkpoint
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args)
        args.resume = True

    # create new model
    else:
        train_loader, val_loader = create_data_loaders(args)
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        in_channels = len(args.modality)
        if args.data_uncertainty:
            resnet_arch = ResNetVar
        else:
            resnet_arch = ResNet
        if args.arch == 'resnet50':
            model = resnet_arch(layers=50, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained, dropout_p=args.dropout_p)
            
        elif args.arch == 'resnet18':
            model = resnet_arch(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained, dropout_p=args.dropout_p, output_var=args.data_uncertainty)
        print("=> model created.")
        optimizer = torch.optim.SGD(model.parameters(), args.lr, \
            momentum=args.momentum, weight_decay=args.weight_decay)

        # model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
        model = model.cuda()

    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        if args.data_uncertainty:
            criterion = criteria.GaussianNLLloss().cuda()
        else:
            criterion = criteria.MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        if args.data_uncertainty:
            criterion = criteria.LaplaceNLLloss().cuda()
        else:
            criterion = criteria.MaskedL1Loss().cuda()

    # create results folder, if not already exists
    output_directory = utils.get_output_directory_teacher(args)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    elif args.test:
        pass
    else:
        n = 1
        candidate_name = output_directory+"_"+str(n)
        while os.path.exists(candidate_name):
            n +=1
            candidate_name = output_directory+"_"+str(n)
        os.makedirs(candidate_name)
        output_directory = candidate_name
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch,  args.lr, args.epochs)
        train(train_loader, model, criterion, optimizer, epoch) # train for one epoch
        if args.data == 'kitti':
            eval_epoch = 4
        else:
            eval_epoch = 1
        if ((epoch+1) % eval_epoch == 0) or ((epoch+1) == args.epochs):
            result, img_merge = validate(val_loader, model, epoch) # evaluate on validation setW
            # remember best rmse and save checkpoint
            is_best = result.rmse < best_result.rmse
            if is_best:
                best_result = result
                with open(best_txt, 'w') as txtfile:
                    txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                        format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
                if img_merge is not None:
                    img_filename = output_directory + '/comparison_best.png'
                    utils.save_image(img_merge, img_filename)

            utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'arch': args.arch,
                'model': model,
                'best_result': best_result,
                'optimizer' : optimizer,
            }, is_best, epoch, output_directory)

        if args.test:
            break
    return output_directory

def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train() # switch to train mode
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        if args.data_uncertainty:
            pred, pred_logvar = model(input)
            loss = criterion(pred, pred_logvar, target)
        else:
            pred = model(input)
            loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))
        if args.test and (i > args.print_freq):
            break

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})

def validate(val_loader, model, epoch, n_sample=25, write_to_file=True):
    average_meter = AverageMeterTeacher()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        #perform monte carlo dropout
        end = time.time()
        with torch.no_grad():
            if args.data_uncertainty:
                pred_dropout, pred_logvar_dropout = generate_mcdropout_predictions_w_var(model, input, n_sample)
            else:
                pred_dropout = generate_mcdropout_predictions(model, input, n_sample)
            pred_mu = torch.mean(pred_dropout, dim=0)
            pred_model_var = torch.var(pred_dropout, dim=0)
            
            if args.data_uncertainty:
                pred_data_var = torch.mean(torch.exp(pred_logvar_dropout), dim=0)
                pred_data_std = torch.sqrt(pred_data_var)
                pred_var = pred_data_var + pred_model_var
            else:
                pred_var = pred_model_var
            
            pred_model_std = torch.sqrt(pred_model_var)
            pred_std = torch.sqrt(pred_var)
            
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = ResultTeacher()
        result.mc_evaluate(pred_mu.data, pred_std.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50
        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = input
            elif args.modality == 'rgbd':
                rgb = input[:,:3,:,:]
                depth = input[:,3:,:,:]

            if i == 0:
                if args.modality == 'rgbd':
                    img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred_mu)
                else:
                    if args.data_uncertainty:
                        img_merge = utils.merge_into_row_w_data_uncertainty(rgb, target, pred_mu, pred_model_std, pred_data_std)
                    else:
                        img_merge = utils.merge_into_row_w_uncertainty(rgb, target, pred_mu, pred_std)
            elif (i < 8*skip) and (i % skip == 0):
                if args.modality == 'rgbd':
                    row = utils.merge_into_row_with_gt(rgb, depth, target, pred_mu)
                else:
                    if args.data_uncertainty:
                        row = utils.merge_into_row_w_data_uncertainty(rgb, target, pred_mu, pred_model_std, pred_data_std)
                    else:
                        row = utils.merge_into_row_w_uncertainty(rgb, target, pred_mu, pred_std)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8*skip:
                filename = output_directory + '/comparison_' + str(epoch) + '.png'
                print(filename)
                utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '
                  'ause={result.ause:.3f}({average.ause:.3f})'
                  'ece={result.ece:.3f}({average.ece:.3f})'
                .format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

        if args.test and (i>args.print_freq):
            break

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'
        'ause={average.ause:.3f}\n'
        'ece={average.ece:.3f}\n'
        .format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time, 'ause': avg.ause, 'ece':avg.ece})
    return avg, img_merge

if __name__ == '__main__':
    output_directory = main()
    model_path = os.path.join(output_directory, 'model_best.pth.tar')
    perform_evaluation(model_path, mc_samples=50)

