import os
import PIL
import time
import typing

from PIL import Image
from typing import Tuple

from modules.ganonymizer import GANonymizer
from options import get_options


def main(args=None):
    opt = get_options(args)
    img, opt.fname, opt.fext = load_image(opt.input)
    model = GANonymizer(opt)
    _, output = model(img)
    save_image(opt, output)


def load_image(img_path: str) -> Tuple[any, str, str]:
    # Loading input image
    print('[INFO] Loading "{}"'.format(img_path)) 
    fname, fext = img_path.split('/')[-1].split('.')
    img = Image.open(img_path)
    return img, fname, fext


def save_image(opt, image: any) -> None:
    if opt.mode == 'debug': return
    if opt.mode == 'exec':
        image.save(opt.output_path)
    else:
        image.save(os.path.join(opt.log, 'output.png'))


if __name__ == '__main__':
    main()
