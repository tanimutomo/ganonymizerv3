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
        '--output_dir',
        type=str, default=None,
        help='path to directory to save output image or video'
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
    parser.add_argument(
        '--realtime',
        type=strtobool, default=False,
        help='realtime ganonymizerv3 for video processing'
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

    # mask and shadow
    parser.add_argument(
        '--shadow_angle', 
        type=int, default=90,
        help='angle for including shadow to mask'
    )
    parser.add_argument(
        '--base_ksize', 
        type=int, default=7,
        help='base kernel size for morphology transformation'
    )
    parser.add_argument(
        '--shadow_ratio', 
        type=int, default=0.3,
        help='shadow height to image height ratio'
    )
    parser.add_argument(
        '--expand_ratio',
        type=float, default=0.01,
        help='expansion width to image size ratio'
    )
    parser.add_argument(
        '--noise_ratio',
        type=float, default=0.01,
        help='max noise size to image size ratio '
    )
    parser.add_argument(
        '--obj_h_ratio',
        type=float, default=0.5,
        help='max object size to image size ratio '
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

    if opt.mode == 'exec':
        opt = _set_output_path(opt)
    elif opt.mode == 'save':
        opt = _set_log_path(opt)
    
    opt = _set_device(opt)

    _print_options(parser, opt)
    return opt


def _set_output_path(opt):
    is_abspath = True if opt.input[0] == '/' else False
    input_dirpath = '/'.join(opt.input.split('/')[:-1])
    input_filename = opt.input.split('/')[-1]
    output_dirpath = ('/' if is_abspath else '') \
                        + input_dirpath + '.out'
    os.makedirs(output_dirpath, exist_ok=True)
    opt.output_path = os.path.join(output_dirpath, input_filename)
    return opt


def _set_log_path(opt):
    opt.log = os.path.join(
        opt.log_root,
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    opt.inter_log = os.path.join(opt.log, 'intermediates')
    os.makedirs(opt.inter_log)
    return opt


def _set_device(opt):
    opt.device = torch.device(
        'cuda:{}'.format(opt.gpu_id) if torch.cuda.is_available() else 'cpu'
    )
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

