#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch

from model import Generator
from dataset import UnpairedDepthDataset
from PIL import Image
import todos
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, type=str, help='name of this experiment')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Where the model checkpoints are saved')
parser.add_argument('--results_dir', type=str, default='results', help='where to save result images')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--depthroot', type=str, default='', help='dataset of corresponding ground truth depth maps')

parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--midas', type=int, default=0, help='use midas depth map')

parser.add_argument('--n_blocks', type=int, default=3, help='number of resnet blocks for generator')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation', default=True)
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load from')

parser.add_argument('--mode', type=str, default='test', help='train, val, test, etc')
parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

parser.add_argument('--how_many', type=int, default=100, help='number of images to test')

opt = parser.parse_args()
print(opt)

opt.no_flip = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

with torch.no_grad():
    # Networks

    net_G = 0
    net_G = Generator(opt.input_nc, opt.output_nc, opt.n_blocks)
    net_G.cuda()

    net_GB = 0

    # Load state dicts
    net_G.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch)))
    print('loaded', os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch))

    # Set model's test mode
    net_G.eval()
    # (Pdb) net_G -- Generator(...)

    transforms_r = [transforms.Resize(int(opt.size), Image.BICUBIC),
                   transforms.ToTensor()]
    # (Pdb) transforms_r -- [Resize(size=256, interpolation=bicubic, max_size=None, antialias=warn), ToTensor()]

    test_data = UnpairedDepthDataset(opt.dataroot, '', opt, transforms_r=transforms_r, 
                mode=opt.mode, midas=opt.midas>0, depthroot=opt.depthroot)

    dataloader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

    ###################################

    ###### Testing######

    full_output_dir = os.path.join(opt.results_dir, opt.name)

    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    for i, batch in enumerate(dataloader):
        if i > opt.how_many:
            break;

        img_r  = batch['r'].cuda()
        img_depth  = batch['depth'].cuda()

        real_A = img_r

        name = batch['name'][0]
        
        input_image = real_A

        # tensor [input_image] size: [1, 3, 256, 384] , min: 0.003921568859368563 , max: 1.0
        image = net_G(input_image)
        # tensor [image] size: [1, 1, 256, 384] , min: 0.015038051642477512 , max: 1.0
        image = (image - image.min())/(image.max() - image.min() + 1e-5)

        save_image(image.data, full_output_dir+'/%s_out.png' % name)

        sys.stdout.write('\rGenerated images %04d of %04d' % (i, opt.how_many))

    sys.stdout.write('\n')
    ###################################
