import os
import torch
import torchvision

import torchvision.transforms.functional as F

from .edge_connect.src.edge_connect import SimpleEdgeConnect


class ImageInpainter():
    def __init__(self, opt):
        self.max_hole_size = opt.max_hole_size
        if opt.inpaint_method == 'EdgeConnect':
            self.model = self.set_edge_connect(opt)

    def __call__(self, img: torch.Tensor, mask: torch.Tensor, max_obj_size: float):
        resize_factor = min(self.max_hole_size / max_obj_size, 1)
        img = self.resize(img, resize_factor)
        mask = self.resize(mask.unsqueeze(0), resize_factor).squeeze()
        return self.model.inpaint(img, mask)

    def resize(self, img: torch.Tensor, resize_factor: float) -> torch.Tensor:
        new_h = int(img.shape[1] * resize_factor)
        new_w = int(img.shape[2] * resize_factor)
        new_h -= new_h % 4
        new_w -= new_w % 4
        img = F.to_pil_image(img).resize((new_w, new_h))
        return F.to_tensor(img)

    def set_edge_connect(self, opt):
        inpaint_ckpt = 'modules/edge_connect/checkpoints'
        model = SimpleEdgeConnect(inpaint_ckpt, 
                                  opt.sigma,
                                  opt.device)
        return model

