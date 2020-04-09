
import argparse
import time
import csv
import datetime

import numpy as np
from scipy.ndimage.interpolation import rotate, zoom

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data
from datasets.sequence_folders import SequenceFolder
import custom_transforms
import models
from utils import *
from logger import TermLogger, AverageMeter
from path import Path
from itertools import chain
from tensorboardX import SummaryWriter
from flowutils.flowlib import flow_to_image
from losses import compute_epe, compute_cossim, multiscale_cossim

epsilon = 1e-8

parser = argparse.ArgumentParser(description='Generating Adversarial Patches for Optical Flow Networks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', dest='data', default='/path/to/the/dataset',
                    help='path to dataset')
parser.add_argument('--kitti-data', dest='kitti_data', default='/path/to/kitti/dataset',
                    help='path to kitti dataset')
parser.add_argument('--patch-path', dest='patch_path', default='',
                    help='Initialize patch from here')
parser.add_argument('--mask-path', dest='mask_path', default='',
                    help='Initialize mask from here')
parser.add_argument('--valset', dest='valset', type=str, default='kitti2015', choices=['kitti2015', 'kitti2012'],
                    help='Optical flow validation dataset')
parser.add_argument('--DEBUG', action='store_true', help='DEBUG Mode')
parser.add_argument('--name', dest='name', type=str, default='demo', required=True,
                    help='name of the experiment, checpoints are stored in checpoints/name')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--max-count', default=2, type=int,
                    help='max count')
parser.add_argument('--epoch-size', default=100, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--flownet', dest='flownet', type=str, default='FlowNetC', choices=['FlowNetS','PWCNet', 'Back2Future', 'FlowNetC', 'SpyNet', 'FlowNet2'],
                    help='flow network architecture. Options: FlowNetS | SpyNet')
parser.add_argument('--alpha', default=0.0, type=float, help='regularization weight')
parser.add_argument('--image-size', type=int, default=384, help='the min(height, width) of the input image to network')
parser.add_argument('--patch-type', type=str, default='circle', help='patch type: circle or square')
parser.add_argument('--patch-size', type=float, default=0.01, help='patch size. E.g. 0.05 ~= 5% of image ')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', type=bool, default=True, help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--norotate', action='store_true', help='will not apply rotation augmentation')
parser.add_argument('--log-terminal', action='store_true', help='will display progressbar at terminal')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=50)

best_error = -1
n_iter = 0


def main():
    global args, best_error, n_iter
    args = parser.parse_args()
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path #/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path)
    output_writer = SummaryWriter(args.save_path/'valid')

    # Data loading code
    flow_loader_h, flow_loader_w = 384, 1280

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(h=256, w=256),
        custom_transforms.ArrayToTensor(),
        ])

    valid_transform = custom_transforms.Compose([custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
                            custom_transforms.ArrayToTensor()])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=3
    )

    if args.valset =="kitti2015":
        from datasets.validation_flow import ValidationFlowKitti2015
        val_set = ValidationFlowKitti2015(root=args.kitti_data, transform=valid_transform)
    elif args.valset =="kitti2012":
        from datasets.validation_flow import ValidationFlowKitti2012
        val_set = ValidationFlowKitti2012(root=args.kitti_data, transform=valid_transform)

    if args.DEBUG:
        train_set.__len__ = 32
        train_set.samples = train_set.samples[:32]

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in valid scenes'.format(len(val_set)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,               # batch size is 1 since images in kitti have different sizes
                    shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    if args.flownet=='SpyNet':
        flow_net = getattr(models, args.flownet)(nlevels=6, pretrained=True)
    elif args.flownet=='Back2Future':
        flow_net = getattr(models, args.flownet)(pretrained='pretrained/b2f_rm_hard.pth.tar')
    elif args.flownet=='PWCNet':
        flow_net = models.pwc_dc_net('pretrained/pwc_net_chairs.pth.tar') # pwc_net.pth.tar')
    else:
        flow_net = getattr(models, args.flownet)()

    if args.flownet in ['SpyNet', 'Back2Future', 'PWCNet']:
        print("=> using pre-trained weights for "+ args.flownet)
    elif args.flownet in ['FlowNetC']:
        print("=> using pre-trained weights for FlowNetC")
        weights = torch.load('pretrained/FlowNet2-C_checkpoint.pth.tar')
        flow_net.load_state_dict(weights['state_dict'])
    elif args.flownet in ['FlowNetS']:
        print("=> using pre-trained weights for FlowNetS")
        weights = torch.load('pretrained/flownets.pth.tar')
        flow_net.load_state_dict(weights['state_dict'])
    elif args.flownet in ['FlowNet2']:
        print("=> using pre-trained weights for FlowNet2")
        weights = torch.load('pretrained/FlowNet2_checkpoint.pth.tar')
        flow_net.load_state_dict(weights['state_dict'])
    else:
        flow_net.init_weights()

    pytorch_total_params = sum(p.numel() for p in flow_net.parameters())
    print("Number of model paramters: " + str(pytorch_total_params))

    flow_net = flow_net.cuda()

    cudnn.benchmark = True
    if args.patch_type == 'circle':
        patch, mask, patch_shape = init_patch_circle(args.image_size, args.patch_size)
        patch_init = patch.copy()
    elif args.patch_type == 'square':
        patch, patch_shape = init_patch_square(args.image_size, args.patch_size)
        patch_init = patch.copy()
        mask = np.ones(patch_shape)
    else:
        sys.exit("Please choose a square or circle patch")

    if args.patch_path:
        patch, mask, patch_shape = init_patch_from_image(args.patch_path, args.mask_path, args.image_size, args.patch_size)
        patch_init = patch.copy()

    if args.log_terminal:
        logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader), attack_size=args.max_count)
        logger.epoch_bar.start()
    else:
        logger=None

    for epoch in range(args.epochs):

        if args.log_terminal:
            logger.epoch_bar.update(epoch)
            logger.reset_train_bar()

        # train for one epoch
        patch, mask, patch_init, patch_shape = train(patch, mask, patch_init, patch_shape, train_loader, flow_net, epoch, logger, training_writer)

        # Validate
        errors, error_names = validate_flow_with_gt(patch, mask, patch_shape, val_loader, flow_net, epoch, logger, output_writer)

        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        #
        if args.log_terminal:
            logger.valid_writer.write(' * Avg {}'.format(error_string))
        else:
            print('Epoch {} completed'.format(epoch))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        torch.save(patch, args.save_path/'patch_epoch_{}'.format(str(epoch)))

    if args.log_terminal:
        logger.epoch_bar.finish()


def train(patch, mask, patch_init, patch_shape, train_loader, flow_net, epoch, logger=None, train_writer=None):
    global args, n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    flow_net.eval()

    end = time.time()

    patch_shape_orig = patch_shape
    for i, (tgt_img, ref_img) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img_var = Variable(tgt_img.cuda())
        ref_past_img_var = Variable(ref_img[0].cuda())
        ref_future_img_var = Variable(ref_img[1].cuda())

        if type(flow_net).__name__ == 'Back2Future':
            flow_pred_var = flow_net(ref_past_img_var, tgt_img_var, ref_future_img_var)
        else:
            flow_pred_var = flow_net(tgt_img_var, ref_future_img_var)
        data_shape = tgt_img.cpu().numpy().shape

        if args.patch_type == 'circle':
            patch, mask, patch_init, rx, ry, patch_shape = circle_transform(patch, mask, patch_init, data_shape, patch_shape, True)
        elif args.patch_type == 'square':
            patch, mask, patch_init, rx, ry = square_transform(patch, mask, patch_init, data_shape, patch_shape, norotate=args.norotate)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        patch_init = torch.FloatTensor(patch_init)

        patch, mask = patch.cuda(), mask.cuda()
        patch_init = patch_init.cuda()
        patch_var, mask_var = Variable(patch), Variable(mask)
        patch_init_var = Variable(patch_init).cuda()

        target_var = Variable(-1*flow_pred_var.data.clone(), requires_grad=True).cuda()
        adv_tgt_img_var, adv_ref_past_img_var, adv_ref_future_img_var, patch_var = attack(flow_net, tgt_img_var, ref_past_img_var, ref_future_img_var, patch_var, mask_var, patch_init_var, target_var=target_var, logger=logger)

        masked_patch_var = torch.mul(mask_var, patch_var)
        patch = masked_patch_var.data.cpu().numpy()
        mask = mask_var.data.cpu().numpy()
        patch_init = patch_init_var.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        new_mask = np.zeros(patch_shape)
        new_patch_init = np.zeros(patch_shape)
        for x in range(new_patch.shape[0]):
            for y in range(new_patch.shape[1]):
                new_patch[x][y] = patch[x][y][ry:ry+patch_shape[-2], rx:rx+patch_shape[-1]]
                new_mask[x][y] = mask[x][y][ry:ry+patch_shape[-2], rx:rx+patch_shape[-1]]
                new_patch_init[x][y] = patch_init[x][y][ry:ry+patch_shape[-2], rx:rx+patch_shape[-1]]

        patch = new_patch
        mask = new_mask
        patch_init = new_patch_init

        patch = zoom(patch, zoom=(1,1,patch_shape_orig[2]/patch_shape[2], patch_shape_orig[3]/patch_shape[3]), order=1)
        mask = zoom(mask, zoom=(1,1,patch_shape_orig[2]/patch_shape[2], patch_shape_orig[3]/patch_shape[3]), order=0)
        patch_init = zoom(patch_init, zoom=(1,1,patch_shape_orig[2]/patch_shape[2], patch_shape_orig[3]/patch_shape[3]), order=1)

        if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:
            train_writer.add_image('train tgt image', transpose_image(tensor2array(tgt_img[0])), n_iter)
            train_writer.add_image('train ref past image', transpose_image(tensor2array(ref_img[0][0])), n_iter)
            train_writer.add_image('train ref future image', transpose_image(tensor2array(ref_img[1][0])), n_iter)
            train_writer.add_image('train adv tgt image', transpose_image(tensor2array(adv_tgt_img_var.data.cpu()[0])), n_iter)
            if type(flow_net).__name__ == 'Back2Future':
                train_writer.add_image('train adv ref past image', transpose_image(tensor2array(adv_ref_past_img_var.data.cpu()[0])), n_iter)
            train_writer.add_image('train adv ref future image', transpose_image(tensor2array(adv_ref_future_img_var.data.cpu()[0])), n_iter)
            train_writer.add_image('train patch', transpose_image(tensor2array(patch_var.data.cpu()[0])), n_iter)
            train_writer.add_image('train patch init', transpose_image(tensor2array(patch_init_var.data.cpu()[0])), n_iter)
            train_writer.add_image('train mask', transpose_image(tensor2array(mask_var.data.cpu()[0])), n_iter)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.log_terminal:
            logger.train_bar.update(i+1)
        if i >= args.epoch_size - 1:
            break

        n_iter += 1

    return patch, mask, patch_init, patch_shape



def attack(flow_net, tgt_img_var, ref_past_img_var, ref_future_img_var, patch_var, mask_var, patch_init_var, target_var, logger):
    global args
    flow_net.eval()

    adv_tgt_img_var = torch.mul((1-mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)
    if type(flow_net).__name__ == 'Back2Future':
        adv_ref_past_img_var = torch.mul((1-mask_var), ref_past_img_var) + torch.mul(mask_var, patch_var)
    adv_ref_future_img_var = torch.mul((1-mask_var), ref_future_img_var) + torch.mul(mask_var, patch_var)

    count = 0
    loss_scalar = 1
    while loss_scalar > 0.1 :
        count += 1
        adv_tgt_img_var = Variable(adv_tgt_img_var.data, requires_grad = True)
        if type(flow_net).__name__ == 'Back2Future':
            adv_ref_past_img_var = Variable(adv_ref_past_img_var.data, requires_grad = True)
        else:
            adv_ref_past_img_var = None
        adv_ref_future_img_var = Variable(adv_ref_future_img_var.data, requires_grad = True)
        just_the_patch = Variable(patch_var.data, requires_grad=True)

        if type(flow_net).__name__ == 'Back2Future':
            adv_flow_out_var = flow_net(adv_ref_past_img_var, adv_tgt_img_var, adv_ref_future_img_var)
        else:
            adv_flow_out_var = flow_net(adv_tgt_img_var, adv_ref_future_img_var)

        loss_data = (1 - nn.functional.cosine_similarity(adv_flow_out_var, target_var )).mean()
        loss_reg = nn.functional.l1_loss(torch.mul(mask_var,just_the_patch), torch.mul(mask_var, patch_init_var))
        loss = (1-args.alpha)*loss_data + args.alpha*loss_reg

        loss.backward()

        adv_tgt_img_grad = adv_tgt_img_var.grad.clone()
        if type(flow_net).__name__ == 'Back2Future':
            adv_ref_past_img_grad = adv_ref_past_img_var.grad.clone()
        adv_ref_future_img_grad = adv_ref_future_img_var.grad.clone()

        adv_tgt_img_var.grad.data.zero_()
        if type(flow_net).__name__ == 'Back2Future':
            adv_ref_past_img_var.grad.data.zero_()
        adv_ref_future_img_var.grad.data.zero_()

        if type(flow_net).__name__ == 'Back2Future':
            patch_var -= torch.clamp(0.5*args.lr*(adv_tgt_img_grad + adv_ref_future_img_grad + adv_ref_past_img_grad), -2, 2)
        else:
            patch_var -= torch.clamp(0.5*args.lr*(adv_tgt_img_grad + adv_ref_future_img_grad), -2, 2)

        adv_tgt_img_var = torch.mul((1-mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)
        if type(flow_net).__name__ == 'Back2Future':
            adv_ref_past_img_var = torch.mul((1-mask_var), ref_past_img_var) + torch.mul(mask_var, patch_var)
        adv_ref_future_img_var = torch.mul((1-mask_var), ref_future_img_var) + torch.mul(mask_var, patch_var)

        adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)
        if type(flow_net).__name__ == 'Back2Future':
            adv_ref_past_img_var = torch.clamp(adv_ref_past_img_var, -1, 1)
        adv_ref_future_img_var = torch.clamp(adv_ref_future_img_var, -1, 1)

        loss_scalar = loss.item()

        if args.log_terminal:
            logger.attack_bar.update(count)

        if count > args.max_count-1:
            break

    return adv_tgt_img_var, adv_ref_past_img_var, adv_ref_future_img_var, patch_var



def validate_flow_with_gt(patch, mask, patch_shape, val_loader, flow_net, epoch, logger, output_writer):
    global args
    batch_time = AverageMeter()
    error_names = ['epe', 'adv_epe', 'cos_sim', 'adv_cos_sim']
    errors = AverageMeter(i=len(error_names))

    flow_net.eval()

    end = time.time()

    for i, (ref_img_past, tgt_img, ref_img_future, flow_gt, _, _, _) in enumerate(val_loader):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        ref_img_past_var = Variable(ref_img_past.cuda(), volatile=True)
        ref_img_future_var = Variable(ref_img_future.cuda(), volatile=True)
        flow_gt_var = Variable(flow_gt.cuda(), volatile=True)

        if type(flow_net).__name__ == 'Back2Future':
            flow_fwd = flow_net(ref_img_past_var, tgt_img_var, ref_img_future_var)
        else:
            flow_fwd = flow_net(tgt_img_var, ref_img_future_var)

        data_shape = tgt_img.cpu().numpy().shape
        if args.patch_type == 'circle':
            patch_full, mask_full, _, _, _, _ = circle_transform(patch, mask, patch.copy(), data_shape, patch_shape)
        elif args.patch_type == 'square':
            patch_full, mask_full, _, _, _ = square_transform(patch, mask, patch.copy(), data_shape, patch_shape, norotate=args.norotate)
        patch_full, mask_full = torch.FloatTensor(patch_full), torch.FloatTensor(mask_full)

        patch_full, mask_full = patch_full.cuda(), mask_full.cuda()
        patch_var, mask_var = Variable(patch_full), Variable(mask_full)

        adv_tgt_img_var = torch.mul((1-mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)
        if type(flow_net).__name__ == 'Back2Future':
            adv_ref_img_past_var = torch.mul((1-mask_var), ref_img_past_var) + torch.mul(mask_var, patch_var)
        adv_ref_img_future_var = torch.mul((1-mask_var), ref_img_future_var) + torch.mul(mask_var, patch_var)

        adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)
        if type(flow_net).__name__ == 'Back2Future':
            adv_ref_img_past_var = torch.clamp(adv_ref_img_past_var, -1, 1)
        adv_ref_img_future_var = torch.clamp(adv_ref_img_future_var, -1, 1)

        if type(flow_net).__name__ == 'Back2Future':
            adv_flow_fwd = flow_net(adv_ref_img_past_var, adv_tgt_img_var, adv_ref_img_future_var)
        else:
            adv_flow_fwd = flow_net(adv_tgt_img_var, adv_ref_img_future_var)

        epe = compute_epe(gt=flow_gt_var, pred=flow_fwd)
        adv_epe = compute_epe(gt=flow_gt_var, pred=adv_flow_fwd)
        cos_sim = compute_cossim(flow_gt_var, flow_fwd)
        adv_cos_sim = compute_cossim(flow_gt_var, adv_flow_fwd)

        errors.update([epe, adv_epe, cos_sim, adv_cos_sim])

        if args.log_output and i % 10 == 0:
            index = int(i//10)
            if epoch == 0:
                output_writer.add_image('val flow Input', transpose_image(tensor2array(tgt_img[0])), 0)
                flow_to_show = flow_gt[0][:2,:,:].cpu()
                output_writer.add_image('val target Flow', transpose_image(flow_to_image(tensor2array(flow_to_show))), epoch)

            val_Flow_Output = transpose_image(flow_to_image(tensor2array(flow_fwd.data[0].cpu()))) / 255.
            val_adv_Flow_Output = transpose_image(flow_to_image(tensor2array(adv_flow_fwd.data[0].cpu()))) / 255.
            val_Diff_Flow_Output = transpose_image(flow_to_image(tensor2array((adv_flow_fwd-flow_fwd).data[0].cpu()))) / 255.
            val_adv_tgt_image = transpose_image(tensor2array(adv_tgt_img_var.data.cpu()[0]))
            if type(flow_net).__name__ == 'Back2Future':
                val_adv_ref_past_image = transpose_image(tensor2array(adv_ref_img_past_var.data.cpu()[0]))
            val_adv_ref_future_image = transpose_image(tensor2array(adv_ref_img_future_var.data.cpu()[0]))
            val_patch = transpose_image(tensor2array(patch_var.data.cpu()[0]))

            if type(flow_net).__name__ == 'Back2Future':
                val_output_viz = np.hstack((val_Flow_Output, val_adv_Flow_Output, val_Diff_Flow_Output, val_adv_ref_past_image, val_adv_tgt_image, val_adv_ref_future_image))
            else:
                val_output_viz = np.hstack((val_Flow_Output, val_adv_Flow_Output, val_Diff_Flow_Output, val_adv_tgt_image, val_adv_ref_future_image))
            output_writer.add_image('val Output viz {}'.format(index), val_output_viz, epoch)


        if args.log_terminal:
            logger.valid_bar.update(i)

        batch_time.update(time.time() - end)
        end = time.time()


    return errors.avg, error_names


if __name__ == '__main__':
    import sys
    with open("experiment_recorder.md", "a") as f:
        f.write('\n python3 ' + ' '.join(sys.argv))
    main()
