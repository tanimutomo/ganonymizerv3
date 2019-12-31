import os
import torch
import torchvision

import torchvision.transforms.functional as F

from .edge_connect.src.edge_connect import SimpleEdgeConnect


class ImageInpainter():
    def __init__(self, opt, debugger):
        self.max_hole_size = opt.max_hole_size
        self.debugger = debugger

        if opt.inpaint_method == 'EdgeConnect':
            self.model = self.set_edge_connect(opt)

    def __call__(self, img: torch.Tensor, mask: torch.Tensor, max_obj_size: float):
        resize_factor = min(self.max_hole_size / max_obj_size, 1)
        img_resized = self.resize(img, resize_factor)
        mask_resized = self.resize(mask.unsqueeze(0), resize_factor).squeeze()
        inpainted_resized, inpainted_edge, edge = self.model.inpaint(img_resized, mask_resized)
        print('[INFO] inpainted image size :', inpainted_resized.shape)
        inpainted = self.replace_original(img, mask, inpainted_resized)
        return inpainted, inpainted_edge, edge

    def resize(self, img: torch.Tensor, resize_factor: float) -> torch.Tensor:
        new_h = int(img.shape[1] * resize_factor)
        new_w = int(img.shape[2] * resize_factor)
        new_h -= new_h % 4
        new_w -= new_w % 4
        img = F.to_pil_image(img).resize((new_w, new_h))
        return F.to_tensor(img)

    def replace_original(self, img: torch.Tensor, mask: torch.Tensor,
                         inpainted_resized: torch.Tensor) -> torch.Tensor:
        inpainted = F.to_pil_image(inpainted_resized).resize((img.shape[-1], img.shape[-2]))
        return torch.where(mask == 1, F.to_tensor(inpainted), img)

    def set_edge_connect(self, opt):
        inpaint_ckpt = 'modules/edge_connect/checkpoints'
        model = SimpleEdgeConnect(inpaint_ckpt, 
                                  opt.sigma,
                                  opt.device)
        return model