import argparse
import numpy as np
import torch
from PIL import Image
from utils import *
parser = argparse.ArgumentParser(description='Adversarial attacks on Optical Flow Networks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--patch_path', dest='patch_path', default='',
                    help='path to dataset')
parser.add_argument('--scale', dest='scale',type=int, default=8,
                    help='resize scale')
parser.add_argument('--output_path', dest='output_path', default='results',
                    help='output dir')
parser.add_argument('--output_name', dest='output_name', default='this_is_your_patch',
                    help='output dir')

def main():
    global args
    args = parser.parse_args()
    patch = torch.Tensor(torch.load(args.patch_path))
    patch_clamped = torch.clamp(patch, -1., 1.)
    patch_im = tensor2array(patch_clamped[0])*255.
    
    # make background white
    mask = createCircularMask(patch_im.shape[0], patch_im.shape[1]).astype('float32')
    mask = np.stack((mask,mask,mask), axis=-1)
    patch_im = (1-mask) * np.ones(patch_im.shape)*255 + mask * patch_im

    patch_im = Image.fromarray(patch_im.astype('uint8'))
    sz = patch_im.size
    patch_im = patch_im.resize((sz[0]*args.scale, sz[1]*args.scale))
    patch_im.save('%s/%s.jpg' % (args.output_path, args.output_name))

if __name__ == '__main__':
    main()
