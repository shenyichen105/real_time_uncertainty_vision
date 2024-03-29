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
from inference_util import generate_ensemble_predictions, generate_ensemble_predictions_w_var
import criteria
import utils
import utils_student


def make_output_dir(args):
    output_directory = utils.get_output_directory_teacher_ensemble(args)
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
    return output_directory

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

def perform_evaluation(ensemble_path):
    global args
    output_directory = ensemble_path
    models = []
    for i in range(args.n_ensemble):
        model_path = os.path.join(ensemble_path, str(i), "model_best.pth.tar")
        assert os.path.isfile(model_path), \
        "=> no best model found at '{}'".format(model_path)
        print("=> loading best model '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        
        args = checkpoint['args']
        model = checkpoint['model']
        models.append(model)
        print("=> loaded best model {} from ensemble".format(i))
    _, val_loader = create_data_loaders(args)
    args.evaluate = True
    avg, _= validate_ensemble(val_loader, output_directory, models, checkpoint['epoch'])
    result_txt = os.path.join(output_directory, 'result_ensemble_{}.txt'.format(args.n_ensemble))
    with open(result_txt, 'w') as txtfile:
        print('\n*\n'
        'ensemble_size={ensemble_size}\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'Delta2={average.delta2:.3f}\n'
        'Delta3={average.delta3:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'
        'ause={average.ause:.6f}\n'
        'ece={average.ece:.6f}\n'        
        .format(average=avg, ensemble_size=args.n_ensemble,time=avg.gpu_time), file=txtfile)
    return
    
def train_one_model(output_sub_directory, resume=False, resume_model_name=None):
    global args
    # evaluation mode
    best_result = Result()
    best_result.set_to_worst()
    start_epoch = 0
    if resume:
        chkpt_path = os.path.join(output_sub_directory, 'model_best.pth.tar')
        assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        output_sub_directory = os.path.dirname(os.path.abspath(chkpt_path))
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

    train_csv = os.path.join(output_sub_directory, 'train.csv')
    test_csv = os.path.join(output_sub_directory, 'test.csv')
    best_txt = os.path.join(output_sub_directory, 'best.txt')

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
        train(train_loader, model, criterion, optimizer, epoch, train_csv) # train for one epoch
        eval_epoch = 1
        if args.data == 'kitti':
            if epoch < 20:
                eval_epoch = 4 
            else:
                eval_epoch = 2
        if ((epoch+1) % eval_epoch == 0) or ((epoch+1) == args.epochs):
            result, img_merge = validate(val_loader, model, epoch, output_sub_directory, test_csv) # evaluate on validation setW
            # remember best rmse and save checkpoint
            is_best = result.rmse < best_result.rmse
            if is_best:
                best_result = result
                with open(best_txt, 'w') as txtfile:
                    txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                        format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
                if img_merge is not None:
                    img_filename = output_sub_directory + '/comparison_best.png'
                    utils.save_image(img_merge, img_filename)

            utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'arch': args.arch,
                'model': model,
                'best_result': best_result,
                'optimizer' : optimizer,
            }, is_best, epoch, output_sub_directory)

        if args.test:
            break
    return output_sub_directory

def train(train_loader, model, criterion, optimizer, epoch, train_csv):
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

def validate(val_loader, model, epoch,  output_sub_directory, test_csv, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            if args.data_uncertainty:
                pred, pred_logvar = model(input)
            else:
                pred = model(input)

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
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
                    img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    if args.data_uncertainty:
                        img_merge = utils.merge_into_row_w_uncertainty(rgb, target, pred, torch.exp(0.5*pred_logvar))
                    else:
                        img_merge = utils.merge_into_row(rgb, target, pred)
            elif (i < 8*skip) and (i % skip == 0):
                if args.modality == 'rgbd':
                    row = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    if args.data_uncertainty:
                        row = utils.merge_into_row_w_uncertainty(rgb, target, pred, torch.exp(0.5*pred_logvar))
                    else:
                        row = utils.merge_into_row(rgb, target, pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8*skip:
                filename =  output_sub_directory + '/comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
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
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge

def validate_ensemble(val_loader, output_directory, models, epoch):
    global args
    average_meter = AverageMeterTeacher()
    for model in models:
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
                pred_all, pred_logvar_all = generate_ensemble_predictions_w_var(models, input)
            else:
                pred_all = generate_ensemble_predictions(models, input)
            pred_mu = torch.mean(pred_all, dim=0)
            pred_model_var = torch.var(pred_all, dim=0)
            
            if args.data_uncertainty:
                pred_data_var = torch.mean(torch.exp(pred_logvar_all), dim=0)
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
    return avg, img_merge

if __name__ == '__main__':
    global args, output_directory
    args = utils.parse_command()
    print(args)

    fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                    'delta1', 'delta2', 'delta3',
                    'data_time', 'gpu_time', 'ause', 'ece']
    n_ensemble = args.n_ensemble
    if args.evaluate:
        perform_evaluation(args.evaluate)
    elif args.resume:
        output_directory = args.resume
        for i in range(n_ensemble):
            output_sub_directory = os.path.join(output_directory, str(i))
            train_one_model(output_sub_directory, resume=True)
    else:
        output_directory = make_output_dir(args)
        for i in range(n_ensemble):
            output_sub_directory = os.path.join(output_directory, str(i))
            if not os.path.exists(output_sub_directory):
                os.makedirs(output_sub_directory)
            train_one_model(output_sub_directory)
        perform_evaluation(output_directory)

