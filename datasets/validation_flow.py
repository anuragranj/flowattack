# Adapted from https://github.com/ClementPinard/SfmLearner-Pytorch/

import torch.utils.data as data
import numpy as np
from PIL import Image
from path import Path
from flowutils import flow_io
import torch
import os
from skimage import transform as sktransform
from raw import *

import cv2


def load_as_float(path):
    return np.array(Image.open(path)).astype(np.float32)

class ValidationFlowKitti2015MV(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, N=4000, phase='training', compression=0, raw_root=None, example=0, true_motion=False):
        self.root = Path(root)
        self.start = max(0, min(example, N))
        if example > 0:
            self.N = 1
        else:
            self.N = N

        self.transform = transform
        self.phase = phase
        self.compression = compression
        self.raw_root = raw_root

    def __getitem__(self, index):
        index = self.start + index
        scene = index // 20
        frame = index % 20

        tgt_img_path = self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(scene).zfill(6)+'_'+str(frame).zfill(2)+'.png')
        ref_img_past_path = self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(scene).zfill(6)+'_'+str(frame-1).zfill(2)+'.png')
        ref_img_future_path = self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(scene).zfill(6)+'_'+str(frame+1).zfill(2)+'.png')
        gt_flow_path = self.root.joinpath('data_scene_flow', self.phase, 'flow_occ', str(scene).zfill(6)+'_'+str(frame).zfill(2)+'.png')
        gt_disp_path = self.root.joinpath('data_scene_flow', self.phase, 'disp_occ_0', str(scene).zfill(6)+'_'+str(frame).zfill(2)+'.png')

        tgt_img = load_as_float(tgt_img_path)
        ref_img_future = load_as_float(ref_img_future_path)
        if os.path.exists(gt_flow_path):
            ref_img_past = load_as_float(ref_img_past_path)
        else:
            ref_img_past = torch.zeros(tgt_img.shape)

        he, wi, ch = tgt_img.shape

        gtFlow = None
        if os.path.exists(gt_flow_path):
            u,v,valid = flow_io.flow_read_png(str(gt_flow_path))
            gtFlow = np.dstack((u,v,valid))
            gtFlow = torch.FloatTensor(gtFlow.transpose(2,0,1))
        else:
            gtFlow = torch.zeros((3, he, wi))

        # read disparity
        gtDisp = None
        if os.path.exists(gt_flow_path):
            gtDisp = load_as_float(gt_disp_path)
            gtDisp = np.array(gtDisp,  dtype=float) / 256.
        else:
            gtDisp = torch.zeros((he, wi, 1))

        calib = {}
        poses = {}

        if self.transform is not None:
            in_h, in_w, _ = tgt_img.shape
            imgs = self.transform([tgt_img] + [ref_img_past] + [ref_img_future])
            tgt_img = imgs[0]
            ref_img_past = imgs[1]
            ref_img_future = imgs[2]
            _, out_h, out_w = tgt_img.shape

        return ref_img_past, tgt_img, ref_img_future, gtFlow, gtDisp, calib, poses

    def __len__(self):
        return self.N

class ValidationFlowKitti2015(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, N=200, phase='training', compression=0, raw_root=None, example=0, true_motion=False):
        self.root = Path(root)
        self.start = max(0, min(example, N))
        if example > 0:
            self.N = 1
        else:
            self.N = N

        self.transform = transform
        self.phase = phase
        self.compression = compression
        self.raw_root = raw_root

        self.mapping = [None] * N
        if true_motion:
            mapping_file_path = os.path.join(raw_root,'train_mapping.txt')
            if os.path.exists(mapping_file_path):
                with open(mapping_file_path) as mapping_file:
                    lines = mapping_file.readlines()
                    for i, line in enumerate(lines):
                        if line.strip():
                            split = line.split(' ')
                            self.mapping[i] = {'Scene': '', 'Sequence': '', 'Frame': None}
                            self.mapping[i]['Scene'] = split[0]
                            self.mapping[i]['Sequence'] = split[1]
                            self.mapping[i]['Frame'] = int(split[2].strip())

    def __getitem__(self, index):
        index = self.start + index

        tgt_img_path = self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_10.png')
        ref_img_past_path = self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_09.png')
        ref_img_future_path = self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_11.png')
        gt_flow_path = self.root.joinpath('data_scene_flow', self.phase, 'flow_occ', str(index).zfill(6)+'_10.png')
        gt_disp_path = self.root.joinpath('data_scene_flow', self.phase, 'disp_occ_0', str(index).zfill(6)+'_10.png')

        tgt_img = load_as_float(tgt_img_path)
        ref_img_past = load_as_float(ref_img_past_path)
        ref_img_future = load_as_float(ref_img_future_path)
        u,v,valid = flow_io.flow_read_png(str(gt_flow_path))
        gtFlow = np.dstack((u,v,valid))
        gtFlow = torch.FloatTensor(gtFlow.transpose(2,0,1))

        # read disparity
        gtDisp = load_as_float(gt_disp_path)
        gtDisp = np.array(gtDisp,  dtype=float) / 256.
        calib = {}
        poses = {}
        # get calibrations
        if self.mapping[index] is not None:
            path = os.path.join(self.raw_root, self.mapping[index]['Scene'])
            seq = self.mapping[index]['Sequence'][len(self.mapping[index]['Scene'] + '_drive')+1:-5]
            dataset = raw(self.raw_root, self.mapping[index]['Scene'], seq, frames=range(self.mapping[index]['Frame'] - 1, self.mapping[index]['Frame'] + 2), origin=1)
            calib = {}
            calib['cam'] = {}
            calib['vel2cam'] = {}
            calib['imu2vel'] = {}
            # import pdb; pdb.set_trace()
            calib['cam']['P_rect_00'] = dataset.calib.P_rect_00
            # calib['cam']['P_rect_00'] = np.eye(4)
            # calib['cam']['P_rect_00'][0, 3] = dataset.calib.P_rect_00[0, 3] / dataset.calib.P_rect_00[0, 0]
            calib['cam']['R_rect_00'] = dataset.calib.R_rect_00
            calib['vel2cam']['RT'] = dataset.calib.T_cam0_velo_unrect
            calib['imu2vel']['RT'] = dataset.calib.T_velo_imu
            poses = [np.array([])] * 3
            poses[0] = dataset.oxts[0].T_w_imu
            poses[1] = dataset.oxts[1].T_w_imu
            poses[2] = dataset.oxts[2].T_w_imu

            # calib['cam']['baseline'] = 0.54
            calib['cam']['baseline'] = dataset.calib.b_rgb

            # calib = {}
            # calib['cam'] = loadCalib(os.path.join(path, 'calib_cam_to_cam.txt'))
            # calib['vel2cam'] = loadCalib(os.path.join(path, 'calib_velo_to_cam.txt'))
            # calib['imu2vel'] = loadCalib(os.path.join(path, 'calib_imu_to_velo.txt'))

            # # load oxts data
            # oxts = loadOxtsliteData(os.path.join(path, self.mapping[index]['Sequence']), range(self.mapping[index]['Frame'] - 1, self.mapping[index]['Frame'] + 2))

            # # get poses
            # poses = convertOxtsToPose(oxts, origin=1)

        if self.transform is not None:
            in_h, in_w, _ = tgt_img.shape
            imgs = self.transform([tgt_img] + [ref_img_past] + [ref_img_future])
            tgt_img = imgs[0]
            ref_img_past = imgs[1]
            ref_img_future = imgs[2]
            _, out_h, out_w = tgt_img.shape

            # scale projection matrix
            if len(calib) > 0 and (in_h != out_h or in_w != out_w):
                sx = float(out_h) / float(in_h)
                sy = float(out_w) / float(in_w)
                calib['cam']['P_rect_00'][0,0] *= sx
                calib['cam']['P_rect_00'][1,1] *= sy
                calib['cam']['P_rect_00'][0,2] *= sx
                calib['cam']['P_rect_00'][1,2] *= sy

        # set baseline, focal length and principal points
        if len(calib) > 0:
            calib['cam']['focal_length_x'] = calib['cam']['P_rect_00'][0,0]
            calib['cam']['focal_length_y'] = calib['cam']['P_rect_00'][1,1]
            calib['cam']['cx'] = calib['cam']['P_rect_00'][0,2]
            calib['cam']['cy'] = calib['cam']['P_rect_00'][1,2]

            # FROM IMU to IMG00
            calib['P_imu_cam'] = calib['cam']["R_rect_00"].dot(calib['vel2cam']["RT"].dot(calib['imu2vel']["RT"]))
            calib['P_imu_img'] = calib['cam']["P_rect_00"].dot(calib['P_imu_cam'])

        #if self.compression > 0 :
        #    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100 - self.compression]
        #    ref_img_past = cv2.imencode('.jpg', ref_img_past, encode_param)
        #    tgt_img = cv2.imencode('.jpg', tgt_img, encode_param)
        #    ref_img_future = cv2.imencode('.jpg', ref_img_future, encode_param)

        return ref_img_past, tgt_img, ref_img_future, gtFlow, gtDisp, calib, poses

    def __len__(self):
        return self.N

class ValidationFlowKitti2012(data.Dataset):
    """
        Kitti 2012 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, N=194, flow_w=1024, flow_h=384, phase='training', compression=None):
        self.root = Path(root)
        self.N = N
        self.transform = transform
        self.phase = phase
        self.compression = compression
        self.flow_h = flow_h
        self.flow_w = flow_w

    def __getitem__(self, index):
        tgt_img_path =  self.root.joinpath('data_stereo_flow', self.phase, 'colored_0',str(index).zfill(6)+'_10.png')
        ref_img_past_path =  self.root.joinpath('data_stereo_flow', self.phase, 'colored_0',str(index).zfill(6)+'_11.png')
        ref_img_future_path =  self.root.joinpath('data_stereo_flow', self.phase, 'colored_0',str(index).zfill(6)+'_11.png')
        gt_flow_path = self.root.joinpath('data_stereo_flow', self.phase, 'flow_occ', str(index).zfill(6)+'_10.png')

        tgt_img = load_as_float(tgt_img_path)
        ref_img_past = load_as_float(ref_img_past_path)
        ref_img_future = load_as_float(ref_img_future_path)

        u,v,valid = flow_io.flow_read_png(gt_flow_path)
        gtFlow = np.dstack((u,v,valid))
        gtFlow = torch.FloatTensor(gtFlow.transpose(2,0,1))

        if self.transform is not None:
            imgs = self.transform([tgt_img] + [ref_img_past] + [ref_img_future])
            tgt_img = imgs[0]
            ref_img_past = imgs[1]
            ref_img_future = imgs[2]

        #if self.compression is not None:
        #    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.compression]
        #    ref_img_past = cv2.imencode('.jpg', ref_img_past, encode_param)
        #    tgt_img = cv2.imencode('.jpg', tgt_img, encode_param)
        #    ref_img_future = cv2.imencode('.jpg', ref_img_future, encode_param)

        return ref_img_past, tgt_img, ref_img_future, gtFlow, None, None,

    def __len__(self):
        return self.N
