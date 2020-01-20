import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True
from models_dropout import ResNet, ResNetStudent
from metrics import AverageMeter, Result, ResultStudent, AverageMeterStudent
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import criteria
import utils
import utils_student

args = utils_student.parse_command()
print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time', 'kl', 'nll_teacher', 'nll_gt']
best_result = Result()
best_result.set_to_worst()

def create_data_loaders(args, teacher_args):
    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join(os.path.expanduser(teacher_args.datapath), teacher_args.data, 'train')
    valdir = os.path.join(os.path.expanduser(teacher_args.datapath), teacher_args.data, 'val')
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = teacher_args.max_depth if teacher_args.max_depth >= 0.0 else np.inf
    if teacher_args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=teacher_args.num_samples, max_depth=max_depth)
    elif teacher_args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=teacher_args.num_samples, max_depth=max_depth)

    if teacher_args.data == 'nyudepthv2':
        from dataloaders.nyu_dataloader import NYUDataset

        if not args.evaluate:
            train_dataset = NYUDataset(traindir, type='train',
                modality=teacher_args.modality, sparsifier=sparsifier)
        val_dataset = NYUDataset(valdir, type='val',
            modality=teacher_args.modality, sparsifier=sparsifier)

    elif teacher_args.data == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args.evaluate:
            train_dataset = KITTIDataset(traindir, type='train',
                modality=teacher_args.modality, sparsifier=sparsifier)
        val_dataset = KITTIDataset(valdir, type='val',
            modality=teacher_args.modality, sparsifier=sparsifier)

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

def load_teacher(args):
    teacher_path = args.teacher
    teacher_model_dict = torch.load(os.path.realpath(os.path.join(teacher_path, "model_best.pth.tar")))
    teacher_model = teacher_model_dict['model']
    teacher_args = teacher_model_dict['args']
    return teacher_model, teacher_args

def main():
    global args, teacher_args, best_result, output_directory, train_csv, test_csv
    model_teacher, teacher_args = load_teacher(args)
    # evaluation mode
    start_epoch = 0
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no best model found at '{}'".format(args.evaluate)
        print("=> loading best model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        output_directory = os.path.dirname(args.evaluate)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model_student = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        _, val_loader = create_data_loaders(args, teacher_args)
        args.evaluate = True
        validate(val_loader, model_student, model_teacher, checkpoint['epoch'], n_samples=25, write_to_file=False)
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
        model_student = checkpoint['model']
        optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args, teacher_args)
        args.resume = True
    # create new model
    else:
        train_loader, val_loader = create_data_loaders(args, teacher_args)
        print("student_args\n", args)
        print("teacher_args\n", teacher_args)
        print("=> creating Model ({}-{}) ...".format(args.arch, teacher_args.decoder))
        in_channels = len(teacher_args.modality)
        if args.arch == 'resnet50':
            model_student = ResNetStudent(layers=50, decoder=teacher_args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=teacher_args.pretrained)
        elif args.arch == 'resnet18':
            model_student = ResNetStudent(layers=18, decoder=teacher_args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=teacher_args.pretrained)
        print("=> model created.")
        #TODO initialize student model with teacher's weights
        if args.use_teacher_weights:
            model_student.load_state_dict(model_teacher.state_dict(), strict=False)
        optimizer = torch.optim.SGD(model_student.parameters(), args.lr, \
            momentum=args.momentum, weight_decay=args.weight_decay)
        # model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
        model_student = model_student.cuda()

    # define loss function (criterion) and optimizer
    criterion = criteria.GaussianNLLloss().cuda()

    # create results folder, if not already exists
    output_directory = utils_student.get_student_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
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
        utils_student.adjust_learning_rate(optimizer, epoch+1, args.lr)
        train_student(train_loader, model_student, model_teacher, criterion, optimizer, epoch, n_samples=args.n_sample, gt_loss_ratio=args.gr) # train for one epoch
        if teacher_args.data == 'kitti':
            eval_epoch = 3
        else:
            eval_epoch = 1
        if ((epoch+1) % eval_epoch == 0) or ((epoch+1) == args.epochs):
            result, img_merge = validate(val_loader, model_student, model_teacher, epoch, n_samples=25) # evaluate on validation set
            # remember best rmse and save checkpoint
            is_best = result.rmse < best_result.rmse
            if is_best:
                best_result = result
                with open(best_txt, 'w') as txtfile:
                    txtfile.write("epoch={}\nnll_gt={:.3f}\nrmse={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nkl={:.3f}t_gpu={:.4f}\n".
                        format(epoch, result.nll_gt, result.rmse, result.mae, result.delta1, result.kl, result.gpu_time))
                if img_merge is not None:
                    img_filename = output_directory + '/comparison_best.png'
                    utils.save_image(img_merge, img_filename)

            utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'arch': args.arch,
                'model': model_student,
                'best_result': best_result,
                'optimizer' : optimizer,
            }, is_best, epoch, output_directory)

def generate_teacher_predictions(model_teacher, input, n_samples):
    #naive implementation, can be slow
    model_teacher.apply(utils_student.enable_dropout) #enable dropout in the inference
    pred_dropout = []
    for i in range(n_samples):
        if teacher_args.data_uncertainty:
            teacher_mean, teacher_logvar = model_teacher(input)
            teacher_sd = (torch.exp(teacher_logvar) + 1e-8)**0.5
            m = torch.distributions.normal.Normal(torch.zeros(teacher_mean.size()), torch.ones(teacher_mean.size()))
            gaussian_samples = m.sample().to(teacher_mean.device)
            pred = teacher_mean + teacher_sd*gaussian_samples
        else:
            pred = model_teacher(input)
        pred_dropout.append(pred)
    pred_dropout = torch.cat(pred_dropout, 0).detach()
    model_teacher.apply(utils_student.disable_dropout)
    return pred_dropout

def expand_output(output, n_sample=5):
    assert n_sample > 0
    #copy the output n_sample times
    #return an output of [n_sample*batch_size, w, h]
    all_output = []
    for i in range(n_sample):
        all_output.append(output.clone())
    all_output = torch.cat(all_output).to(output.device)
    return all_output

def train_student(train_loader, model_student, model_teacher, criterion, optimizer, epoch, n_samples=5, gt_loss_ratio=0.1):
    average_meter = AverageMeterStudent()
    model_student.train() # switch to train mode
    model_teacher.eval() # switch to eval mode for teacher
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        with torch.no_grad():
            pred_dropout = generate_teacher_predictions(model_teacher, input, n_samples)
        pred_mu, pred_logvar = model_student(input)
        pred_mu = expand_output(pred_mu, n_sample=n_samples)
        pred_logvar = expand_output(pred_logvar, n_sample=n_samples)
        
        pred_mu_gt = pred_mu[:input.size()[0],:,:,:]
        pred_logvar_gt = pred_logvar[:input.size()[0],:,:,:]
        
        nll_teacher = criterion(pred_mu, pred_logvar, pred_dropout)
        nll_gt = criterion(pred_mu_gt, pred_logvar_gt, target, mask_zero=True)
        loss = nll_teacher + gt_loss_ratio * nll_gt

        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = ResultStudent()
        result.evaluate(pred_mu.data, pred_logvar.data, target.data, pred_dropout.data, 
                        nll_teacher=nll_teacher.data.mean(), nll_gt=nll_gt.data.mean())
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'nll_teacher={result.nll_teacher:.3f}({average.nll_teacher:.3f})'
                  'nll_gt={result.nll_gt:.3f}({average.nll_gt:.3f})'
                  'kl={result.kl:.3f}({average.kl:.3f})'.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time, 'nll_teacher':avg.nll_gt, 'nll_gt':avg.nll_teacher, 'kl':avg.kl})


def validate(val_loader, model_student, model_teacher, epoch, n_samples=25, write_to_file=True):
    average_meter = AverageMeterStudent()
    model_student.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end
        
        # compute output
        end = time.time()
        with torch.no_grad():
            _, pred_dropout = generate_teacher_predictions(model_teacher, input, n_samples)
            pred_mu, pred_logvar = model_student(input)
            pred_std = torch.exp(0.5*pred_logvar)
            
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        # measure accuracy and record loss
        result = ResultStudent()
        result.evaluate(pred_mu.data, pred_logvar.data, target.data, pred_dropout.data)

        pred_mu_teacher = torch.mean(pred_dropout.view(target.size()[0], -1, target.size()[1], target.size()[2], target.size()[3]), 1)
        pred_std_teacher = torch.std(pred_dropout.view(target.size()[0], -1, target.size()[1], target.size()[2], target.size()[3]), 1)


        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        # TODO create visualization for uncertainty
        skip = 50
        if teacher_args.modality == 'd':
            img_merge = None
        else:
            if teacher_args.modality == 'rgb':
                rgb = input
            elif teacher_args.modality == 'rgbd':
                rgb = input[:,:3,:,:]
                depth = input[:,3:,:,:]

            if i == 0:
                if teacher_args.modality == 'rgbd':
                    img_merge = utils_student.merge_into_row_with_gt(rgb, depth, target, pred_mu)
                else:
                    img_merge = utils_student.merge_into_row_with_confidence(rgb, target, pred_mu_teacher, pred_mu, pred_std_teacher, pred_std)
            elif (i < 8*skip) and (i % skip == 0):
                if teacher_args.modality == 'rgbd':
                    row = utils_student.merge_into_row_with_gt(rgb, depth, target, pred_mu)
                else:
                    row = utils_student.merge_into_row_with_confidence(rgb, target, pred_mu_teacher, pred_mu, pred_std_teacher, pred_std)
                img_merge = utils_student.add_row(img_merge, row)
            elif i == 8*skip:
                filename = output_directory + '/comparison_' + str(epoch) + '.png'
                utils_student.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        'kl={average.kl:.3f}\n'
        'nll_gt={average.nll_gt:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time, 'nll_teacher':avg.nll_teacher, 'nll_gt':avg.nll_gt, 'kl':avg.kl})
    return avg, img_merge

if __name__ == '__main__':
    main()
