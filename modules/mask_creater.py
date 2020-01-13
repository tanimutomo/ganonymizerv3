import numpy as np
import PIL
import torch

from PIL import Image, ImageDraw
import torch.nn.functional as F

from .utils import labels


class MaskCreater:
    def __init__(self, opt, debugger):
        self.shadow_angle = opt.shadow_angle
        self.kernel_ratio = opt.kernel_ratio
        self.num_iter_expansion = opt.num_iter_expansion
        self.shadow_height_iter = opt.shadow_height_iter
        self.rough_obj_size = opt.rough_obj_size
        self.device = opt.device

        self.debugger = debugger

        self.prvobj_ids = list()
        self.road_ids = list()
        for label in labels:
            if label.trainId == 19:
                continue
            if label.category in ['vehicle', 'human']:
                self.prvobj_ids.append(label.trainId)
            if label.category == 'flat':
                self.road_ids.append(label.trainId)

    def __call__(self, segmap: torch.Tensor) -> (torch.Tensor, float):
        obj_mask = self.create_base_mask(segmap, self.prvobj_ids)
        road_mask = self.create_base_mask(segmap, self.road_ids)
        self.debugger.imsave(road_mask, 'road_mask.png')

        self.set_base_kernel(segmap.shape)
        obj_mask = obj_mask.reshape(1, 1, obj_mask.shape[0], obj_mask.shape[1])
        obj_mask = self.clean_mask(obj_mask)
        obj_mask = self.expand_mask(obj_mask)
        obj_mask = self.include_shadow(obj_mask, road_mask)
        max_obj_size = self.get_max_obj_size(obj_mask)
        return obj_mask.squeeze().cpu(), max_obj_size

    def set_base_kernel(self, shape: tuple) -> None:
        self.ksize = int(np.mean(shape) * self.kernel_ratio)
        self.kernel = get_circle_kernel(self.ksize).to(self.device)
    
    def create_base_mask(self, segmap: torch.Tensor, label_ids: list) -> torch.Tensor:
        obj_mask = segmap.clone().to(self.device)
        road_mask = segmap.clone().to(self.device)
        for id_ in label_ids:
            obj_mask = torch.where(obj_mask==id_,
                                   torch.full_like(obj_mask, 255),
                                   obj_mask)
        return where(obj_mask==255, 1, 0).to(torch.float32)

    def clean_mask(self, mask: torch.Tensor) -> torch.Tensor:
        out = closing(mask, self.kernel)
        out = opening(out, self.kernel)
        return out

    def expand_mask(self, mask: torch.Tensor) -> torch.Tensor:
        out = mask.clone()
        for _ in range(self.num_iter_expansion):
            out = morph_transform(out, self.kernel, transform='dilation')
        return out

    def include_shadow(self, mask: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
        shadow_kernel = get_circle_kernel(
            self.ksize,
            start = 270 - self.shadow_angle//2,
            end = 285 - self.shadow_angle//2
        ).to(self.device)
        out = mask.clone()
        for _ in range(self.shadow_height_iter):
            out = morph_transform(out, shadow_kernel, transform='dilation')
        shadow = where(out - mask + road_mask == 2, 1, 0)
        out = closing(mask + shadow, self.kernel)
        return out.to(torch.float32)

    def get_max_obj_size(self, mask: torch.Tensor) -> int:
        out = mask.clone()
        for itr in range(100):
            if out.sum() == 0: break
            out = morph_transform(out, self.kernel, transform='erosion')
        return itr * self.ksize


def opening(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    out = morph_transform(img, kernel, transform='erosion')
    out = morph_transform(out, kernel, transform='dilation')
    return out


def closing(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    out = morph_transform(img, kernel, transform='dilation')
    out = morph_transform(out, kernel, transform='erosion')
    return out


def morph_transform(img: torch.Tensor, kernel: torch.Tensor, transform: str) -> torch.Tensor:
    padding = int((kernel.shape[-1] - 1) / 2)
    out = F.conv2d(img, kernel, padding=padding)
    if transform == 'erosion':
        condition = out == kernel.sum()
    elif transform == 'dilation':
        condition = out > 0
    out = where(condition, 1, 0)
    return out.to(torch.float32)


def get_circle_kernel(size: int, start: int =0, end: int =360) -> torch.Tensor:
    if size % 2 == 0:
        size -= 1
    center = (size - 1) // 2
    kernel = np.zeros((size, size), np.uint8)
    kernel = Image.fromarray(kernel)
    draw = ImageDraw.Draw(kernel)
    draw.pieslice((0, 0, size-1, size-1), start=start, end=end, fill=(255))
    kernel = np.array(kernel) / 255
    kernel = torch.from_numpy(kernel).to(dtype=torch.float32)
    return kernel.reshape(1, 1, size, size)


def where(condition: torch.Tensor, true_val: float, false_val: float) -> torch.Tensor:
    return torch.where(condition,
                       torch.full_like(condition, true_val),
                       torch.full_like(condition, false_val))