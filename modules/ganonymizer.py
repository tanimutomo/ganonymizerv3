import cv2
import numpy as np
import os
import pickle
import PIL
import time
import torch
import torchvision

from PIL import Image
from torchvision import transforms

from .semantic_segmenter import SemanticSegmenter
from .mask_creater import MaskCreater
from .inpainter import ImageInpainter
from .utils import Debugger, label_img_to_color


class GANonymizer:
    def __init__(self, opt):
        for k, v in opt.__dict__.items():
            setattr(self, k, v)
        self.debugger = Debugger(
            opt.mode,
            save_dir = opt.inter_log if opt.mode == 'save' else None
        )

        print('[INFO] Loading modules')
        self.ss = SemanticSegmenter(opt)
        self.mc = MaskCreater(opt, self.debugger)
        self.ii = ImageInpainter(opt, self.debugger)

        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __call__(self, pil_img):
        """
        Args:
            pil_img : input image (PIL.Image)

        Returns:
            output image (PIL.Image)
        """
        start_time = time.time()
        # resize and convert to torch.Tensor
        img, base_size = self.preprocess(pil_img)

        # semantic segmentation for detecting dynamic objects and creating mask
        label_map = self.detect(img)

        # get object mask
        mask, max_obj_size = self.create_mask(img, label_map)

        # image and edge inpainting
        inpainted = self.inpaint(img, mask, max_obj_size)

        # resize and convert to PIL
        output = self.postprocess(inpainted, base_size)

        print('[INFO] whole elapsed time :', time.time() - start_time)
        return self.to_pil(img), output

    def preprocess(self, img):
        print('===== Preprocess =====')
        print('[INFO] original image size :', img.size)
        start = time.time()
        if self.resize_factor is None:
            print('[INFO] elapsed time :', time.time() - start)
            return self.to_tensor(img), img.size
        new_w = int(img.size[0] * self.resize_factor)
        new_h = int(img.size[1] * self.resize_factor)
        new_w -= new_w % 4
        new_h -= new_h % 4
        img = img.resize((new_w, new_h))
        print('[INFO] resized image size :', (new_w, new_h))
        print('[INFO] elapsed time :', time.time() - start)
        return self.to_tensor(img), (new_w, new_h)

    def detect(self, img):
        # semantic segmentation
        print('===== Semantic Segmentation =====')
        start = time.time()
        label_map = self.ss(img)
        vis, lc_img = label_img_to_color(label_map)
        self.debugger.imsave(vis, 'color_semseg_map.png')
        self.debugger.imsave(lc_img, 'label_color_map.png')
        print('[INFO] elapsed time :', time.time() - start)
        return label_map

    def create_mask(self, img, label_map):
        # create mask image and image with mask
        print('===== Creating Mask Image =====')
        start = time.time()
        mask, max_obj_size = self.mc(label_map) # shape=(h, w) # dtype=torch.float32
        print('[INFO] max_obj_size :', max_obj_size)
        # visualize the mask overlayed image
        mask3c = torch.stack([mask, torch.zeros_like(mask), torch.zeros_like(mask)], dim=0) 
        self.debugger.matrix(mask3c, 'mask3c')
        self.debugger.matrix(img, 'img')
        overlay = img*0.8 + mask3c*0.2
        self.debugger.imsave(mask, 'final_mask.png')
        self.debugger.imsave(overlay, 'img_with_mask.png')
        print('[INFO] elapsed time :', time.time() - start)
        return mask, max_obj_size

    def inpaint(self, img: torch.Tensor, mask: torch.Tensor, max_obj_size: float):
        # inpainter
        print('===== Image Inpainting =====')
        start = time.time()
        inpainted, inpainted_edge, edge = self.ii(img, mask, max_obj_size)
        self.debugger.imsave(edge, 'edge.png')
        self.debugger.imsave(inpainted_edge, 'inpainted_edge.png')
        print('[INFO] elapsed time :', time.time() - start)
        return inpainted

    def postprocess(self, img: torch.Tensor, size: tuple) -> torch.Tensor:
        print('===== Postprocess =====')
        start = time.time()
        out = self.to_pil(img)
        out = out.resize(size)
        print('[INFO] elapsed time :', time.time() - start)
        return out