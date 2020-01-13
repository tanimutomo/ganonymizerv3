import numpy as np
import PIL
import torch
import typing

from PIL import Image, ImageDraw
import torch.nn.functional as F
from typing import Union

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

        self.set_base_kernel(segmap.shape)
        obj_mask = obj_mask.reshape(1, 1, obj_mask.shape[0], obj_mask.shape[1])
        road_mask = road_mask.reshape(1, 1, road_mask.shape[0], road_mask.shape[1])
        obj_mask = self.clean_mask(obj_mask)
        road_mask = self.clean_mask(road_mask)
        self.debugger.imsave(road_mask.squeeze(), 'road_mask.png')

        obj_mask = self.expand_mask(obj_mask)
        self.check_height(obj_mask)
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
        self.debugger.imsave(out.squeeze(), 'mask_expanded.png')
        return out

    def check_height(self, mask:torch.Tensor) -> None:
        # h, w, itr = 5, 1, mask.shape[2] // self.thresh_max_obj // 2
        h, w, itr = 5, 1, mask.shape[2] // 2 // 2
        out = mask.clone()
        hmap = torch.zeros_like(mask)
        vert_kernel = torch.ones(1, 1, h, w).to(device=self.device, dtype=torch.float32)
        for _ in range(itr):
            out = morph_transform(out, vert_kernel, transform='erosion')
            hmap += out * (h-1)//2 * 2
        hmap = hmap.squeeze()
        heights = torch.max(hmap, 0).values
        self.debugger.matrix(heights, 'heights')
        p25, p50, p75 = percentile(heights, 25), percentile(heights, 50), percentile(heights, 75)
        self.debugger.value([p25, p50, p75], 'percentiles')
        self.debugger.imsave(hmap, 'hmap.png')
        hmap_p25 = (((0 < heights) & (heights <= p25)).unsqueeze(0).expand(hmap.shape[0], -1) * hmap) > 1
        self.debugger.imsave(hmap_p25, 'hmap_p25.png')
        hmap_p50 = (((p25 < heights) & (heights <= p50)).unsqueeze(0).expand(hmap.shape[0], -1) * hmap) > 1
        self.debugger.imsave(hmap_p50, 'hmap_p50.png')
        hmap_p75 = (((p50 < heights) & (heights <= p75)).unsqueeze(0).expand(hmap.shape[0], -1) * hmap) > 1
        self.debugger.imsave(hmap_p75, 'hmap_p75.png')
        hmap_p100 = ((p75 < heights).unsqueeze(0).expand(hmap.shape[0], -1) * hmap) > 1
        self.debugger.imsave(hmap_p100, 'hmap_p100.png')

    def include_shadow(self, mask: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
        shadow_kernel = get_circle_kernel(
            self.ksize,
            start = 270 - self.shadow_angle//2,
            end = 270 + self.shadow_angle//2
        ).to(self.device)
        out = mask.clone()
        for _ in range(self.shadow_height_iter):
            out = morph_transform(out, shadow_kernel, transform='dilation')
        self.debugger.imsave(out.squeeze(), 'mask_with_raw_shadow.png')
        shadow = where(out - mask + road_mask == 2, 1, 0)
        self.debugger.imsave(shadow.squeeze(), 'shadow.png')
        out = closing(mask + shadow, self.kernel)
        self.debugger.imsave(out.squeeze(), 'mask_with_shadow_on_road.png')
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
    padding = ((kernel.shape[2] - 1) // 2, (kernel.shape[3] - 1) // 2)
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


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result
