import copy
import cv2
import numpy as np
import os
import torchvision

import torchvision.transforms.functional as F

from modules.ganonymizer import GANonymizer
from options import get_options


def main(args=None):
    opt = get_options(args)

    print("Loading '{}'".format(opt.input))
    count = 1
    # Load the video
    if opt.realtime:
        cap = cv2.VideoCapture(int(opt.input))
    else:
        fname, cap, origin_fps, frames, width, height = load_video(opt.input)
    if opt.mode != 'debug' and not opt.realtime:
        writer = video_writer(os.path.join(opt.log, 'output.avi'),
                              origin_fps, width, height, opt.resize_factor)
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


def load_video(path):
    print('[INFO] Loading video')
    fname, fext = path.split('/')[-1].split('.')
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(frames, fps, W, H))
    return fname, cap, fps, frames, W, H
    

def video_writer(path, fps, width, height, resize_factor):
    print("[INFO] Saved Video Path:", path)
    if resize_factor is not None:
        width = int(width * resize_factor)
        height = int(height * resize_factor)
        width -= width % 4
        height -= height % 4
    height *= 2
    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    return writer


if __name__ == '__main__':
    main()