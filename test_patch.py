import argparse
import numpy as np
from numpy.linalg import inv
from scipy.misc import imread, imresize
from tqdm import tqdm
from PIL import Image

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data

import cv2

import custom_transforms
import models
from utils import *
from logger import AverageMeter
from path import Path
from tensorboardX import SummaryWriter
from flowutils.flowlib import flow_to_image,interp_gt_flow
from losses import compute_epe, compute_cossim, multiscale_cossim

epsilon = 1e-8

parser = argparse.ArgumentParser(description='Test Adversarial attacks on Optical Flow Networks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', dest='name', default='', required=True,
                    help='path to dataset')
parser.add_argument('--patch_path', dest='patch_path', default='',
                    help='path to dataset')
parser.add_argument('--whole_img', dest='whole_img', default=0.0, type=float,
                    help='Test whole image attack')
parser.add_argument('--compression', dest='compression', default=0.0, type=float,
                    help='Test whole image attack')
parser.add_argument('--example', dest='example', default=0, type=int,
                    help='Test whole image attack')
parser.add_argument('--fixed_loc_x', dest='fixed_loc_x', default=-1, type=int,
                    help='Test whole image attack')
parser.add_argument('--fixed_loc_y', dest='fixed_loc_y', default=-1, type=int,
                    help='Test whole image attack')
parser.add_argument('--mask_path', dest='mask_path', default='',
                    help='path to dataset')
parser.add_argument('--ignore_mask_flow', action='store_true', help='ignore flow in mask region')
parser.add_argument('--valset', dest='valset', type=str, default='kitti2015', choices=['kitti2015', 'kitti2012'],
                    help='Optical flow validation dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
# parser.add_argument('-b', '--batch_size', default=1, type=int,
#                     metavar='N', help='mini-batch size')
parser.add_argument('--flownet', dest='flownet', type=str, default='FlowNetC', choices=['FlowNetS', 'FlowNetC', 'SpyNet', 'FlowNet2', 'PWCNet', 'Back2Future'],
                    help='flow network architecture. Options: FlowNetS | SpyNet')
#parser.add_argument('--image_size', type=int, default=384, help='the min(height, width) of the input image to network')
parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
parser.add_argument('--norotate', action='store_true', help='will display progressbar at terminal')
parser.add_argument('--true_motion', action='store_true', help='use the true motion according to static scene if intrinsics and depth are available')


def main():
    global args
    args = parser.parse_args()
    save_path = Path(args.name)
    args.save_path = 'results'/save_path #/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    output_vis_dir = args.save_path / 'images'
    output_vis_dir.makedirs_p()

    args.batch_size = 1

    output_writer = SummaryWriter(args.save_path/'valid')

    # Data loading code
    flow_loader_h, flow_loader_w = 384, 1280

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5])

    # valid_transform = custom_transforms.Compose([custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
    #                         custom_transforms.ArrayToTensor(), normalize])
    valid_transform = custom_transforms.Compose([custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
                            custom_transforms.ArrayToTensor()])

    if args.valset =="kitti2015":
        # from datasets.validation_flow import ValidationFlowKitti2015MV
        # val_set = ValidationFlowKitti2015MV(root='/ps/project/datasets/AllFlowData/kitti/kitti2015', transform=valid_transform, compression=args.compression, raw_root='/is/rg/avg/jjanai/data/Kitti_2012_2015/Raw', example=args.example, true_motion=args.true_motion)
        from datasets.validation_flow import ValidationFlowKitti2015
        # # val_set = ValidationFlowKitti2015(root='/is/ps2/aranjan/AllFlowData/kitti/kitti2015', transform=valid_transform, compression=args.compression)
        val_set = ValidationFlowKitti2015(root='/ps/project/datasets/AllFlowData/kitti/kitti2015', transform=valid_transform, compression=args.compression, raw_root='/is/rg/avg/jjanai/data/Kitti_2012_2015/Raw', example=args.example, true_motion=args.true_motion)
    elif args.valset =="kitti2012":
        from datasets.validation_flow import ValidationFlowKitti2012
        # val_set = ValidationFlowKitti2012(root='/is/ps2/aranjan/AllFlowData/kitti/kitti2012', transform=valid_transform, compression=args.compression)
        val_set = ValidationFlowKitti2012(root='/ps/project/datasets/AllFlowData/kitti/kitti2012', transform=valid_transform, compression=args.compression, raw_root='/is/rg/avg/jjanai/data/Kitti_2012_2015/Raw')

    print('{} samples found in valid scenes'.format(len(val_set)))

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,               # batch size is 1 since images in kitti have different sizes
                    shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    result_file = open(os.path.join(args.save_path,'results.csv'),'a')
    result_scene_file = open(os.path.join(args.save_path,'result_scenes.csv'),'a')

    # create model
    print("=> fetching model")

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

    flow_net = flow_net.cuda()

    cudnn.benchmark = True

    if args.whole_img == 0 and args.compression == 0:
        print("Loading patch from ", args.patch_path)
        patch = torch.load(args.patch_path)
        patch_shape = patch.shape
        if args.mask_path:
            mask_image = load_as_float(args.mask_path)
            mask_image = imresize(mask_image, (patch_shape[-1], patch_shape[-2]))/256.
            mask = np.array([mask_image.transpose(2,0,1)])
        else:
            if args.patch_type == 'circle':
                mask = createCircularMask(patch_shape[-2], patch_shape[-1]).astype('float32')
                mask = np.array([[mask,mask,mask]])
            elif args.patch_type == 'square':
                mask = np.ones(patch_shape)
    else:
        # add gaussian noise
        mean = 0
        var = 1
        sigma = var**0.5
        patch = np.random.normal(mean,sigma,(flow_loader_h,flow_loader_w,3))
        patch = patch.reshape(3, flow_loader_h, flow_loader_w)
        mask = np.ones(patch.shape) * args.whole_img

    #import ipdb; ipdb.set_trace()
    error_names = ['epe', 'adv_epe', 'cos_sim', 'adv_cos_sim']
    errors = AverageMeter(i=len(error_names))

    # header
    result_file.write("{:>10}, {:>10}, {:>10}, {:>10}\n".format(*error_names))
    result_scene_file.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n".format(*(['scene'] + error_names)))

    flow_net.eval()

    # set seed for reproductivity
    np.random.seed(1337)

    for i, (ref_img_past, tgt_img, ref_img, flow_gt, disp_gt, calib, poses) in enumerate(tqdm(val_loader)):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        ref_past_img_var = Variable(ref_img_past.cuda(), volatile=True)
        ref_img_var = Variable(ref_img.cuda(), volatile=True)
        flow_gt_var = Variable(flow_gt.cuda(), volatile=True)

        if type(flow_net).__name__ == 'Back2Future':
            flow_fwd = flow_net(ref_past_img_var, tgt_img_var, ref_img_var)
        else:
            flow_fwd = flow_net(tgt_img_var, ref_img_var)

        data_shape = tgt_img.cpu().numpy().shape

        margin = 0
        if len(calib) > 0:
            margin = int(disp_gt.max())

        random_x = args.fixed_loc_x
        random_y = args.fixed_loc_y
        if args.whole_img == 0:
            if args.patch_type == 'circle':
                patch_full, mask_full, _, random_x, random_y, _ = circle_transform(patch, mask, patch.copy(), data_shape, patch_shape, margin, norotate=args.norotate, fixed_loc=(random_x, random_y))
            elif args.patch_type == 'square':
                patch_full, mask_full, _, _, _ = square_transform(patch, mask, patch.copy(), data_shape, patch_shape, norotate=args.norotate)
            patch_full, mask_full = torch.FloatTensor(patch_full), torch.FloatTensor(mask_full)
        else:
            patch_full, mask_full = torch.FloatTensor(patch), torch.FloatTensor(mask)

        patch_full, mask_full = patch_full.cuda(), mask_full.cuda()
        patch_var, mask_var = Variable(patch_full), Variable(mask_full)

        patch_var_future = patch_var_past = patch_var
        mask_var_future = mask_var_past = mask_var

        # adverserial flow
        bt, _, h_gt, w_gt = flow_gt_var.shape
        forward_patch_flow = Variable(torch.cat((torch.zeros((bt, 2, h_gt, w_gt)), torch.ones((bt, 1, h_gt, w_gt))), 1).cuda(), volatile=True)

        # project patch into 3D scene
        if len(calib) > 0:
            # #################################### ONLY WORKS WITH BATCH SIZE 1 ####################################
            imu2vel = calib['imu2vel']["RT"][0].numpy()
            imu2cam = calib['P_imu_cam'][0].numpy()
            imu2img = calib['P_imu_img'][0].numpy()

            pose_past = poses[0][0].numpy()
            pose_ref = poses[1][0].numpy()
            inv_pose_ref = inv(pose_ref)
            pose_fut = poses[2][0].numpy()

            # get point in IMU
            patch_disp = disp_gt[0, random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]]
            valid = (patch_disp > 0)
            # set to object or free space disparity
            if False and args.fixed_loc_x > 0 and args.fixed_loc_y > 0:
                # disparity = patch_disp[valid].mean() - 3  # small correction for gps errors
                disparity = patch_disp[valid].mean()
            else:
                subset = patch_disp[valid]
                min_disp = 0
                if len(subset) > 0:
                    min_disp = subset.min()
                max_disp = disp_gt.max()

                disparity = np.random.uniform(min_disp, max_disp)                          # disparity

            # print('Disp from ', min_disp, ' to ', max_disp)
            depth = (calib['cam']['focal_length_x'] * calib['cam']['baseline'] / disparity)
            p_cam0 = np.array([[0], [0], [0], [1]])
            p_cam0[0] = depth * (random_x - calib['cam']['cx']) / calib['cam']['focal_length_x']
            p_cam0[1] = depth * (random_y - calib['cam']['cy']) / calib['cam']['focal_length_y']
            p_cam0[2] = depth

            # transform
            T_p_cam0 = np.eye(4)
            T_p_cam0[0:4,3:4] = p_cam0

            # transformation to generate patch points
            patch_size = -0.25
            pts = np.array([[0,0,0,1],[0,patch_size,0,1], [patch_size,0,0,1], [patch_size,patch_size,0,1]]).T
            pts = inv(imu2cam).dot(T_p_cam0.dot(pts))

            # get points in reference image
            pts_src = pose_ref.dot(pts)
            pts_src = imu2img.dot(pts_src)
            pts_src = pts_src[:3,:] / pts_src[2:3,:].repeat(3,0)

            # get points in past image
            pts_past = pose_past.dot(pts)
            pts_past = imu2img.dot(pts_past)
            pts_past = pts_past[:3,:] / pts_past[2:3,:].repeat(3,0)

            # get points in future image
            pts_fut = pose_fut.dot(pts)
            pts_fut = imu2img.dot(pts_fut)
            pts_fut = pts_fut[:3,:] / pts_fut[2:3,:].repeat(3,0)

            # find homography between points
            H_past, _ = cv2.findHomography(pts_src.T, pts_past.T, cv2.RANSAC)
            H_fut, _ = cv2.findHomography(pts_src.T, pts_fut.T, cv2.RANSAC)

            # import pdb; pdb.set_trace()
            refMtrx = torch.from_numpy(H_fut).float().cuda()
            refMtrx = refMtrx.repeat(args.batch_size,1,1)
            # get pixel origins
            X,Y = np.meshgrid(np.arange(flow_loader_w),np.arange(flow_loader_h))
            X,Y = X.flatten(),Y.flatten()
            XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
            XYhom = np.tile(XYhom,[args.batch_size,1,1]).astype(np.float32)
            XYhom = torch.from_numpy(XYhom).cuda()
            XHom,YHom,Zom = torch.unbind(XYhom,dim=1)
            XHom = XHom.resize_((args.batch_size,flow_loader_h,flow_loader_w))
            YHom = YHom.resize_((args.batch_size,flow_loader_h,flow_loader_w))
            # warp the canonical coordinates
            XYwarpHom = refMtrx.matmul(XYhom)
            XwarpHom,YwarpHom,ZwarpHom = torch.unbind(XYwarpHom,dim=1)
            Xwarp = (XwarpHom/(ZwarpHom+1e-8)).resize_((args.batch_size,flow_loader_h,flow_loader_w))
            Ywarp = (YwarpHom/(ZwarpHom+1e-8)).resize_((args.batch_size,flow_loader_h,flow_loader_w))
            # get forward flow
            u = (XHom - Xwarp).unsqueeze(1)
            v = (YHom - Ywarp).unsqueeze(1)
            flow = torch.cat((u, v), 1)
            flow = nn.functional.upsample(flow, size=(h_gt, w_gt), mode='bilinear')
            flow[:,0,:,:] = flow[:,0,:,:] * (w_gt/flow_loader_w)
            flow[:,1,:,:] = flow[:,1,:,:] * (h_gt/flow_loader_h)
            forward_patch_flow[:,:2,:,:] = flow
            # get grid for resampling
            Xwarp = 2 * ((Xwarp / (flow_loader_w - 1)) - 0.5)
            Ywarp = 2 * ((Ywarp / (flow_loader_h - 1)) - 0.5)
            grid = torch.stack([Xwarp,Ywarp],dim=-1)
            # sampling with bilinear interpolation
            patch_var_future = torch.nn.functional.grid_sample(patch_var,grid,mode="bilinear")
            mask_var_future = torch.nn.functional.grid_sample(mask_var,grid,mode="bilinear")

            # use past homography
            refMtrxP = torch.from_numpy(H_past).float().cuda()
            refMtrx = refMtrx.repeat(args.batch_size,1,1)
            # warp the canonical coordinates
            XYwarpHomP = refMtrxP.matmul(XYhom)
            XwarpHomP,YwarpHomP,ZwarpHomP = torch.unbind(XYwarpHomP,dim=1)
            XwarpP = (XwarpHomP/(ZwarpHomP+1e-8)).resize_((args.batch_size,flow_loader_h,flow_loader_w))
            YwarpP = (YwarpHomP/(ZwarpHomP+1e-8)).resize_((args.batch_size,flow_loader_h,flow_loader_w))
            # get grid for resampling
            XwarpP = 2 * ((XwarpP / (flow_loader_w - 1)) - 0.5)
            YwarpP = 2 * ((YwarpP / (flow_loader_h - 1)) - 0.5)
            gridP = torch.stack([XwarpP,YwarpP],dim=-1)
            # sampling with bilinear interpolation
            patch_var_past = torch.nn.functional.grid_sample(patch_var,gridP,mode="bilinear")
            mask_var_past = torch.nn.functional.grid_sample(mask_var,gridP,mode="bilinear")

        adv_tgt_img_var = torch.mul((1-mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)
        adv_ref_past_img_var = torch.mul((1-mask_var_past), ref_past_img_var) + torch.mul(mask_var_past, patch_var_past)
        adv_ref_img_var = torch.mul((1-mask_var_future), ref_img_var) + torch.mul(mask_var_future, patch_var_future)

        adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)
        adv_ref_past_img_var = torch.clamp(adv_ref_past_img_var, -1, 1)
        adv_ref_img_var = torch.clamp(adv_ref_img_var, -1, 1)

        if type(flow_net).__name__ == 'Back2Future':
            adv_flow_fwd = flow_net(adv_ref_past_img_var, adv_tgt_img_var, adv_ref_img_var)
        else:
            adv_flow_fwd = flow_net(adv_tgt_img_var, adv_ref_img_var)

        # set patch to zero flow!
        mask_var_res = nn.functional.upsample(mask_var, size=(h_gt, w_gt), mode='bilinear')

        # Ignore patch motion if set!
        if args.ignore_mask_flow:
            forward_patch_flow = Variable(torch.cat((torch.zeros((bt, 2, h_gt, w_gt)), torch.zeros((bt, 1, h_gt, w_gt))), 1).cuda(), volatile=True)

        flow_gt_var_adv = torch.mul((1-mask_var_res), flow_gt_var) + torch.mul(mask_var_res, forward_patch_flow)

        # import pdb; pdb.set_trace()
        epe = compute_epe(gt=flow_gt_var, pred=flow_fwd)
        adv_epe = compute_epe(gt=flow_gt_var_adv, pred=adv_flow_fwd)
        cos_sim = compute_cossim(flow_gt_var, flow_fwd)
        adv_cos_sim = compute_cossim(flow_gt_var_adv, adv_flow_fwd)

        errors.update([epe, adv_epe, cos_sim, adv_cos_sim])

        if i % 1 == 0:
            index = i #int(i//10)
            imgs = normalize([tgt_img] + [ref_img_past] + [ref_img])
            norm_tgt_img = imgs[0]
            norm_ref_img_past = imgs[1]
            norm_ref_img = imgs[2]

            patch_cpu = patch_var.data[0].cpu()
            mask_cpu = mask_var.data[0].cpu()

            adv_norm_tgt_img = normalize(adv_tgt_img_var.data.cpu()) #torch.mul((1-mask_cpu), norm_tgt_img) + torch.mul(mask_cpu, patch_cpu)
            adv_norm_ref_img_past = normalize(adv_ref_past_img_var.data.cpu()) # torch.mul((1-mask_cpu), norm_ref_img_past) + torch.mul(mask_cpu, patch_cpu)
            adv_norm_ref_img = normalize(adv_ref_img_var.data.cpu()) #torch.mul((1-mask_cpu), norm_ref_img) + torch.mul(mask_cpu, patch_cpu)

            output_writer.add_image('val flow Input', transpose_image(tensor2array(norm_tgt_img[0])), 0)
            flow_to_show = flow_gt[0][:2,:,:].cpu()
            output_writer.add_image('val target Flow', transpose_image(flow_to_image(tensor2array(flow_to_show))), 0)

            # set flow to zero
            # zero_flow = Variable(torch.zeros(flow_fwd.shape).cuda(), volatile=True)
            # flow_fwd_masked = torch.mul((1-mask_var[:,:2,:,:]), flow_fwd) + torch.mul(mask_var[:,:2,:,:], zero_flow)
            flow_fwd_masked = flow_fwd

            # get ground truth flow
            val_GT_adv = flow_gt_var_adv.data[0].cpu().numpy().transpose(1, 2, 0)
            # val_GT_adv = interp_gt_flow(val_GT_adv[:,:,:2], val_GT_adv[:,:,2])
            val_GT_adv = cv2.resize(val_GT_adv, (flow_loader_w, flow_loader_h), interpolation=cv2.INTER_NEAREST)
            val_GT_adv[:,:,0] = val_GT_adv[:,:,0] * (flow_loader_w/w_gt)
            val_GT_adv[:,:,1] = val_GT_adv[:,:,1] * (flow_loader_h/h_gt)

            # gt normalization for visualization
            u = val_GT_adv[:, :, 0]
            v = val_GT_adv[:, :, 1]
            idxUnknow = (abs(u) > 1e7) | (abs(v) > 1e7)
            u[idxUnknow] = 0
            v[idxUnknow] = 0
            rad = np.sqrt(u ** 2 + v ** 2)
            maxrad = np.max(rad)

            val_GT_adv_Output = flow_to_image(val_GT_adv, maxrad)
            val_GT_adv_Output = cv2.erode(val_GT_adv_Output, np.ones((3,3), np.uint8), iterations = 1) # make points thicker
            val_GT_adv_Output = transpose_image(val_GT_adv_Output) / 255.
            val_Flow_Output = transpose_image(flow_to_image(tensor2array(flow_fwd.data[0].cpu()), maxrad)) / 255.
            val_adv_Flow_Output = transpose_image(flow_to_image(tensor2array(adv_flow_fwd.data[0].cpu()), maxrad)) / 255.
            val_Diff_Flow_Output = transpose_image(flow_to_image(tensor2array((adv_flow_fwd-flow_fwd_masked).data[0].cpu()), maxrad)) / 255.

            val_tgt_image = transpose_image(tensor2array(norm_tgt_img[0]))
            val_ref_image = transpose_image(tensor2array(norm_ref_img[0]))
            val_adv_tgt_image = transpose_image(tensor2array(adv_norm_tgt_img[0]))
            val_adv_ref_image_past = transpose_image(tensor2array(adv_norm_ref_img_past[0]))
            val_adv_ref_image = transpose_image(tensor2array(adv_norm_ref_img[0]))
            val_patch = transpose_image(tensor2array(patch_var.data.cpu()[0]))
            # print(adv_norm_tgt_img.shape)
            # print(flow_fwd.data[0].cpu().shape)

            # if type(flow_net).__name__ == 'Back2Future':
            #     val_output_viz = np.concatenate((val_adv_ref_image_past, val_adv_tgt_image, val_adv_ref_image, val_Flow_Output, val_adv_Flow_Output, val_Diff_Flow_Output), 2)
            # else:
            # val_output_viz = np.concatenate((val_adv_tgt_image, val_adv_ref_image, val_Flow_Output, val_adv_Flow_Output, val_Diff_Flow_Output, val_GT_adv_Output), 2)
            val_output_viz = np.concatenate((val_ref_image, val_adv_ref_image, val_Flow_Output, val_adv_Flow_Output, val_Diff_Flow_Output, val_GT_adv_Output), 2)
            val_output_viz_im = Image.fromarray((255*val_output_viz.transpose(1, 2, 0)).astype('uint8'))
            val_output_viz_im.save(args.save_path/args.name+'viz'+str(i).zfill(3)+'.jpg')
            output_writer.add_image('val Output viz {}'.format(index), val_output_viz, 0)

            #val_output_viz = np.vstack((val_Flow_Output, val_adv_Flow_Output, val_Diff_Flow_Output, val_adv_tgt_image, val_adv_ref_image))
            #scipy.misc.imsave('outfile.jpg', os.path.join(output_vis_dir, 'vis_{}.png'.format(index)))

            result_scene_file.write("{:10d}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(i, epe, adv_epe, cos_sim, adv_cos_sim))


    print("{:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*errors.avg))
    result_file.write("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(*errors.avg))
    result_scene_file.write("{:>10}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(*(["avg"] + errors.avg)))

    result_file.close()
    result_scene_file.close()

if __name__ == '__main__':
    main()
