import argparse
import datetime
import distutils
import os 
import torch

from distutils.util import strtobool


def get_options(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # main
    parser.add_argument(
        '--input',
        type=str, required=True,
        help='path to input file'
    )
    parser.add_argument(
        '--mode', 
        type=str, default='exec', choices=['exec', 'debug', 'save'],
        help='mode for execution'
    )
    parser.add_argument(
        '--log_root', 
        type=str, default='log',
        help='path to root directory for log'
    )
    parser.add_argument(
        '--gpu_id', 
        type=int, default=0,
        help='id for cuda'
    )

    # resize
    parser.add_argument(
        '--resize_factor', 
        type=float, default=None,
        help='factor of resizing input image'
    )
    
    # segmentation
    parser.add_argument(
        '--semseg_method', 
       type=str, default='DeepLabV3', choices=['DeepLabV3'],
       help='method of semantic segmentation'
    )
    parser.add_argument(
        '--resnet', 
        type=int, default=18,
        help='depth of backborn resnet'
    )

    # mask
    parser.add_argument(
        '--shadow_angle', 
        type=int, default=30,
        help='angle for including shadow to mask'
    )
    parser.add_argument(
        '--kernel_ratio', 
        type=float, default=0.01,
        help='kernel size = image size * kernel_ratio'
    )
    parser.add_argument(
        '--shadow_height_iter', 
        type=int, default=10,
        help='shadow_height = image size * kernel_ratio * shadow_height_iter'
    )
    parser.add_argument(
        '--rough_obj_size', 
        type=strtobool, default=False,
        help='use rough obj size and skip the process for estimating max obj size for making process fast'
    )

    # inpaint
    parser.add_argument(
        '--inpaint_method', 
        type=str, default='EdgeConnect', choices=['EdgeConnect'],
        help='method of image inpainting'
    )
    parser.add_argument(
        '--sigma', 
        type=int, default=2, 
        help='sigma of canny edge detection'
    )
    parser.add_argument(
        '--max_hole_size', 
        type=int, default=100,
        help='maximum hole size when inpainting'
    )

    # parse, print, and return
    opt = parser.parse_args(args)

    # set device
    opt.device = torch.device(
        'cuda:{}'.format(opt.gpu_id) if torch.cuda.is_available() else 'cpu'
    )
    # set log directory
    opt.log = os.path.join(
        opt.log_root,
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    opt.inter_log = None
    if opt.mode != 'debug':
        os.makedirs(opt.log)
        if opt.mode == 'save':
            opt.inter_log = os.path.join(opt.log, 'intermediates')
            os.mkdir(opt.inter_log)

    _print_options(parser, opt)
    return opt


def _print_options(parser, opt):
    message = ''
    message += '---------------------------- Options --------------------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: {}]'.format(str(default))
        message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
    message += '---------------------------- End ------------------------------'
    print(message, '\n')

