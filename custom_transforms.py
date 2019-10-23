from __future__ import division
import torch
import random
import numpy as np
from utils import imresize

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images):
        if random.random() < 0.5:
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
        else:
            output_images = images
        return output_images


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to a particular size"""
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, images):
        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]
        offset_y = np.random.randint(scaled_h - self.h + 1)
        offset_x = np.random.randint(scaled_w - self.w + 1)
        cropped_images = [im[offset_y:offset_y + self.h, offset_x:offset_x + self.w] for im in scaled_images]
        return cropped_images

class RandomCrop(object):
    """Randomly zooms images up to 15% and crop them to a particular size"""
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, images):
        in_h, in_w, _ = images[0].shape
        offset_y = np.random.randint(in_h - self.h + 1)
        offset_x = np.random.randint(in_w - self.w + 1)
        cropped_images = [im[offset_y:offset_y + self.h, offset_x:offset_x + self.w] for im in images]
        return cropped_images

class Scale(object):
    """Scales images to a particular size"""
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, images):

        in_h, in_w, _ = images[0].shape
        scaled_h, scaled_w = self.h , self.w

        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        return scaled_images
