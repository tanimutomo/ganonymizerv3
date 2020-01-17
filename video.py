import copy
import cv2
import numpy as np
import os
import torchvision
import typing

import torchvision.transforms.functional as F
from typing import NewType, Tuple

from modules.ganonymizer import GANonymizer
from options import get_options

VideoRead = NewType('VideoRead', cv2.VideoCapture)
VideoWrite = NewType('VideoWrite', cv2.VideoWriter)


def main(args=None):
    opt = get_options(args)
    if opt.mode == 'save':
        raise ValueError('In video processing, "mode" should be "exec" or "debug".')

    print("Loading '{}'".format(opt.input))
    count = 1
    # Load the video
    if opt.realtime:
        cap = cv2.VideoCapture(int(opt.input))
        frames = 0
    else:
        cap, origin_fps, frames, width, height = load_video(opt.input)
    if opt.mode != 'debug' and not opt.realtime:
        writer = video_writer(opt.output_path, origin_fps,
                              width, height, opt.resize_factor)
    model = GANonymizer(opt)

    while(cap.isOpened()):
        print('')
        ret, frame = cap.read()
        if ret:
            print('-----------------------------------------------------')
            print('[INFO] Count: {}/{}'.format(count, frames))

            # process
            input, output = model(F.to_pil_image(frame))
            input, output = np.array(input), np.array(output)
            concat = np.concatenate([input, output], axis=0)
            if opt.mode != 'debug':
                if opt.realtime:
                    cv2.imshow('frame', concat)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    writer.write(concat)
            count += 1
        else:
            break

    # Stop video process
    cap.release()
    if opt.mode != 'debug' and not opt.realtime:
        writer.release()
    cv2.destroyAllWindows()


def load_video(path: str) -> Tuple[VideoRead, int, int, int, int]:
    print('[INFO] Loading "{}" ...'.format(path))
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(frames, fps, W, H))
    return cap, fps, frames, W, H
    

def video_writer(path: str, fps: int, width: int, height: int,
                 resize_factor: float) -> VideoWrite:
    print("[INFO] Save output video in", path)
    if resize_factor is not None:
        width = int(width * resize_factor)
        height = int(height * resize_factor)
        width -= width % 4
        height -= height % 4
    height *= 2
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    writer = cv2.VideoWriter(path, fourcc, fourcc, (width, height))
    return writer


if __name__ == '__main__':
    main()
