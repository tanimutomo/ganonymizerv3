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
        for k, v in opt.__dict__.items():
            setattr(self, k, v)

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
        self.debugger.matrix(segmap, 'segmap')
        obj_mask = self.create_base_mask(segmap, self.prvobj_ids)
        road_mask = self.create_base_mask(segmap, self.road_ids)
        self.debugger.imsave(road_mask, 'road_mask_raw.png')

        self.set_base_kernel()
        obj_mask = obj_mask.reshape(1, 1, obj_mask.shape[0], obj_mask.shape[1])
        road_mask = road_mask.reshape(1, 1, road_mask.shape[0], road_mask.shape[1])
        obj_mask = self.clean_mask(obj_mask)
        road_mask = self.clean_mask(road_mask)
        self.debugger.imsave(road_mask.squeeze(), 'road_mask.png')

        obj_mask = self.expand_mask(obj_mask)
        obj_heights = self.comp_obj_heights(obj_mask)
        obj_mask = self.include_shadow(obj_heights, obj_mask, road_mask)

        obj_mask = self.clean_mask(obj_mask)
        max_obj_size = self.get_max_obj_size(obj_mask)
        return obj_mask.squeeze().cpu(), max_obj_size

    def set_base_kernel(self) -> None:
        self.ksize = self.base_ksize
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
        num_iter = int(np.mean(mask.shape) * self.noise_ratio) // ((self.ksize - 1)//2)
        self.debugger.imsave(mask.squeeze(), 'a.png')
        out = closing(mask, self.kernel, num_iter)
        self.debugger.imsave(out.squeeze(), 'b.png')
        out = opening(out, self.kernel, num_iter)
        self.debugger.imsave(out.squeeze(), 'c.png')
        return out

    def expand_mask(self, mask: torch.Tensor) -> torch.Tensor:
        out = mask.clone()
        num_iter = int(np.mean(mask.shape) * self.expand_ratio) // ((self.ksize - 1)//2)
        for _ in range(num_iter):
            out = morph_transform(out, self.kernel, transform='dilation')
        self.debugger.imsave(out.squeeze(), 'mask_expanded.png')
        return out

    def comp_obj_heights(self, obj_mask: torch.Tensor) -> torch.Tensor:
        num_iter = int(self.obj_h_ratio * obj_mask.shape[2] // 2)
        vert_kernel = torch.ones(1, 1, self.ksize, 1).to(device=self.device, dtype=torch.float32)
        hmap = torch.zeros_like(obj_mask).to(self.device)
        out = obj_mask.clone()
        for _ in range(num_iter):
            out = morph_transform(out, vert_kernel, transform='erosion')
            hmap += out * (self.ksize-1)//2 * 2
        self.debugger.save_colormap(hmap, 'hmap.png')
        heights = torch.max(hmap.squeeze(), 0).values
        self.debugger.matrix(heights, 'heights')
        return heights

    def include_shadow(self, heights : torch.Tensor, obj_mask: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
        percs = 0, percentile(heights, 25), percentile(heights, 50), percentile(heights, 75), heights.max()
        self.debugger.value(percs, 'percentiles')

        shadows = torch.zeros_like(obj_mask)
        for i in range(len(percs) - 1):
            mask_in_p = ((percs[i] < heights) & (heights <= percs[i+1]))
            mask_in_p = mask_in_p.reshape(1, 1, 1, heights.shape[0])
            mask_in_p = (mask_in_p.expand(-1, -1, obj_mask.shape[2], -1) * obj_mask) == 1
            self.debugger.imsave(mask_in_p, 'mask_in_p{}.png'.format(i))
            mask_in_p_with_s = self._comp_shadow_area(mask_in_p, self.ksize, int(percs[i] + percs[i+1])//2)
            shadow_in_p = mask_in_p_with_s - mask_in_p.to(torch.float32)
            self.debugger.imsave(shadow_in_p, 'shadow_in_p{}.png'.format(i))
            shadows += shadow_in_p
        shadows = where(shadows > 0, 1, 0).to(torch.float32)
        self.debugger.imsave(shadows, 'shadow_raw.png')

        # filtering shadow with road area
        shadow_on_road = where(shadows + road_mask == 2, 1, 0)
        self.debugger.imsave(shadow_on_road.squeeze(), 'shadow_on_road.png')
        out = obj_mask + shadow_on_road
        self.debugger.imsave(out, 'mask_object_with_shadow_on_road.png')
        return out.to(torch.float32)

    def _comp_shadow_area(self, mask: torch.Tensor, k_size: int, obj_height: int) -> torch.Tensor:
        shadow_ratio = 3
        itr = obj_height // shadow_ratio // ((k_size - 1) // 2)
        shadow_kernel = get_circle_kernel(
            k_size,
            start = 270 - self.shadow_angle//2,
            end = 270 + self.shadow_angle//2
        ).to(self.device)
        out = mask.clone()
        for _ in range(itr):
            out = morph_transform(out, shadow_kernel, transform='dilation')
        return out.to(torch.float32)

    def get_max_obj_size(self, mask: torch.Tensor) -> int:
        out = mask.clone()
        for itr in range(100):
            if out.sum() == 0: break
            out = morph_transform(out, self.kernel, transform='erosion')
        return itr * self.ksize


def opening(img: torch.Tensor, kernel: torch.Tensor, num_iter: int) -> torch.Tensor:
    out = img.clone()
    for _ in range(num_iter):
        out = morph_transform(out, kernel, transform='erosion')
    for _ in range(num_iter):
        out = morph_transform(out, kernel, transform='dilation')
    return out


def closing(img: torch.Tensor, kernel: torch.Tensor, num_iter: int) -> torch.Tensor:
    out = img.clone()
    for _ in range(num_iter):
        out = morph_transform(out, kernel, transform='dilation')
    for _ in range(num_iter):
        out = morph_transform(out, kernel, transform='erosion')
    return out


def morph_transform(img: torch.Tensor, kernel: torch.Tensor, transform: str) -> torch.Tensor:
    img = img.to(torch.float32)
    pad_sizes = ((kernel.shape[2] - 1) // 2, (kernel.shape[3] - 1) // 2)
    pad_val = 1 if transform == 'erosion' else 0
    padded_img = padding(img, pad_sizes, pad_val)
    out = F.conv2d(padded_img, kernel)
    if transform == 'erosion':
        condition = out == kernel.sum()
    elif transform == 'dilation':
        condition = out > 0
    out = where(condition, 1, 0)
    return out.to(torch.float32)


def padding(img: torch.Tensor, sizes: tuple, val: int) -> torch.Tensor:
    b, c, h, w = img.shape
    padded_img = torch.full(
        (b, c, h + 2*sizes[0], w + 2*sizes[1]),
        val,
        dtype=img.dtype, device=img.device
    )
    if sizes[0] == 0:
        padded_img[:, :, :, sizes[1]:-sizes[1]] = img
    elif sizes[1] == 0:
        padded_img[:, :, sizes[0]:-sizes[0], :] = img
    else:
        padded_img[:, :, sizes[0]:-sizes[0], sizes[1]:-sizes[1]] = img
    return padded_img


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
