import numpy as np
import PIL
import torch

from PIL import Image, ImageDraw
import torch.nn.functional as F

from .utils import labels


class MaskCreater:
    def __init__(self, opt):
        self.shadow_angle = opt.shadow_angle
        self.kernel_ratio = opt.kernel_ratio
        self.shadow_height_factor = opt.shadow_height_factor
        self.rough_obj_size = opt.rough_obj_size
        self.device = opt.device

        self.target_ids = list()
        for label in labels:
            if ((label.category is 'vehicle' or label.category is 'human')
                    and
                    label.trainId is not 19):
                self.target_ids.append(label.trainId)

    def __call__(self, segmap: torch.Tensor) -> (torch.Tensor, float):
        mask = self.get_base_mask(segmap)
        mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])
        mask = self.cleaning_mask(mask)
        mask = self.include_shadow(mask)
        max_obj_size = self.get_max_obj_size(mask)
        return mask.squeeze().cpu(), max_obj_size
    
    def get_base_mask(self, segmap: torch.Tensor) -> torch.Tensor:
        obj_mask = segmap.clone().to(self.device)
        for id_ in self.target_ids:
            obj_mask = torch.where(obj_mask==id_,
                                   torch.full_like(obj_mask, 255),
                                   obj_mask)
        return where(obj_mask==255, 1, 0).to(torch.float32)

    def cleaning_mask(self, mask: torch.Tensor) -> torch.Tensor:
        size = int(np.mean(mask.shape[-2:]) * self.kernel_ratio)
        kernel = get_circle_kernel(size).to(self.device)
        out = closing(mask, kernel)
        out = opening(out, kernel)
        return out

    def include_shadow(self, mask: torch.Tensor) -> torch.Tensor:
        size = int(np.mean(mask.shape[-2:]) * self.kernel_ratio) * self.shadow_height_factor
        shadow_kernel = get_circle_kernel(
            size,
            start = 270 - self.shadow_angle//2,
            end = 285 - self.shadow_angle//2
        ).to(self.device)
        return morph_transform(mask, shadow_kernel, transform='dilation')

    def get_max_obj_size(self, mask: torch.Tensor) -> int:
        size = int(np.mean(mask.shape[-2:]) * self.kernel_ratio)
        kernel = get_circle_kernel(size).to(self.device)
        out = mask.clone()
        for itr in range(100):
            out = morph_transform(out, kernel, transform='erosion')
            if out.sum() == 0:
                break
        return itr * size


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
