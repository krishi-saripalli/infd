import random
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import datasets
from datasets import register
from utils.geometry import make_coord_grid


class BaseWrapperCAE:

    def __init__(self, dataset, resize_inp, ret_gt=True, resize_gt_lb=None, resize_gt_ub=None,
                 final_crop_gt=None, p_whole=0.0, p_max=0.0, input_is_tensor=False):
        self.dataset = datasets.make(dataset)
        self.resize_inp = resize_inp
        self.ret_gt = ret_gt
        self.resize_gt_lb = resize_gt_lb
        self.resize_gt_ub = resize_gt_ub
        self.final_crop_gt = final_crop_gt
        self.p_whole = p_whole
        self.p_max = p_max
        self.input_is_tensor = input_is_tensor

        self.pil_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
        self.interpolation_mode = 'bilinear'

    def process(self, data):
        ret = {}

        
        if not self.input_is_tensor:
            img = data
            assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}"
            assert img.size[0] == img.size[1], "Image must be square"
            img_size = img.size[0]

            inp_pil = img.resize((self.resize_inp, self.resize_inp), Image.LANCZOS)
            inp = self.pil_transform(inp_pil)
            ret.update({'inp': inp})

            if self.ret_gt:
                if self.resize_gt_lb is None:
                    gt_resized_pil = img
                    r = img_size
                else:
                    if random.random() < self.p_whole:
                        r = img_size
                    elif random.random() < self.p_max:
                        r = min(img_size, self.resize_gt_ub)
                    else:
                        r = random.randint(self.resize_gt_lb, min(img_size, self.resize_gt_ub))
                    gt_resized_pil = img.resize((r, r), Image.LANCZOS)

                gt_resized_norm = self.pil_transform(gt_resized_pil)

                p = self.final_crop_gt
                gt_h, gt_w = gt_resized_norm.shape[-2:]
                if gt_h < p or gt_w < p:
                    padding_h = max(0, p - gt_h)
                    padding_w = max(0, p - gt_w)
                    pad_top = padding_h // 2
                    pad_bottom = padding_h - pad_top
                    pad_left = padding_w // 2
                    pad_right = padding_w - pad_left
                    gt_resized_norm = F.pad(gt_resized_norm, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
                    gt_h, gt_w = gt_resized_norm.shape[-2:]

                ii = random.randint(0, gt_h - p)
                jj = random.randint(0, gt_w - p)
                gt_patch = gt_resized_norm[:, ii: ii + p, jj: jj + p]

                x0, y0 = ii / gt_h, jj / gt_w
                x1, y1 = (ii + p) / gt_h, (jj + p) / gt_w
                coord = make_coord_grid((p, p), range=((x0, x1), (y0, y1)), device=gt_patch.device)
                coord = 2 * coord - 1
                cell = torch.tensor([2 / gt_h, 2 / gt_w], dtype=torch.float32, device=gt_patch.device)
                cell = cell.view(1, 1, 2).expand(p, p, -1)
                ret.update({
                    'gt': gt_patch,
                    'gt_coord': coord,
                    'gt_cell': cell,
                })

        else:
            if isinstance(data, tuple):
                img_tensor = data[0]
            else:
                img_tensor = data

            assert isinstance(img_tensor, torch.Tensor), f"Expected Tensor, got {type(img_tensor)}"
            if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
                img_tensor = img_tensor.squeeze(0)
            assert img_tensor.dim() == 3, f"Expected tensor shape (C, H, W), got {img_tensor.shape}"
            assert img_tensor.shape[1] == img_tensor.shape[2], "Tensor image must be square"
            img_size = img_tensor.shape[1]

            if img_size == self.resize_inp:
                 inp = img_tensor
            else:
                 inp = F.interpolate(img_tensor.unsqueeze(0),
                                     size=(self.resize_inp, self.resize_inp),
                                     mode=self.interpolation_mode,
                                     align_corners=False, antialias=True).squeeze(0)
            ret.update({'inp': inp})

            if self.ret_gt:
                gt_base_tensor = img_tensor

                if self.resize_gt_lb is None:
                    r = img_size
                else:
                    if random.random() < self.p_whole:
                         r = img_size
                    elif random.random() < self.p_max:
                        r = min(img_size, self.resize_gt_ub)
                    else:
                        r = random.randint(self.resize_gt_lb, min(img_size, self.resize_gt_ub))

                if r == img_size:
                    gt_resized_norm = gt_base_tensor
                else:
                    gt_resized_norm = F.interpolate(gt_base_tensor.unsqueeze(0), size=(r, r),
                                               mode=self.interpolation_mode, align_corners=False, antialias=True).squeeze(0)

                p = self.final_crop_gt
                gt_h, gt_w = gt_resized_norm.shape[-2:]
                if gt_h < p or gt_w < p:
                    padding_h = max(0, p - gt_h)
                    padding_w = max(0, p - gt_w)
                    pad_top = padding_h // 2
                    pad_bottom = padding_h - pad_top
                    pad_left = padding_w // 2
                    pad_right = padding_w - pad_left
                    gt_resized_norm = F.pad(gt_resized_norm, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
                    gt_h, gt_w = gt_resized_norm.shape[-2:]

                ii = random.randint(0, gt_h - p)
                jj = random.randint(0, gt_w - p)
                gt_patch = gt_resized_norm[:, ii: ii + p, jj: jj + p]

                x0, y0 = ii / gt_h, jj / gt_w
                x1, y1 = (ii + p) / gt_h, (jj + p) / gt_w
                coord = make_coord_grid((p, p), range=((x0, x1), (y0, y1)), device=gt_patch.device)
                coord = 2 * coord - 1
                cell = torch.tensor([2 / gt_h, 2 / gt_w], dtype=torch.float32, device=gt_patch.device)
                cell = cell.view(1, 1, 2).expand(p, p, -1)
                ret.update({
                    'gt': gt_patch,
                    'gt_coord': coord,
                    'gt_cell': cell,
                })

        return ret

@register('wrapper_cae')
class WrapperCAE(BaseWrapperCAE, Dataset):
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return self.process(data)
