#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import torch.nn.functional as F


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


class DoGFilter(nn.Module):
    def __init__(self, channels, sigma1):
        super(DoGFilter, self).__init__()
        self.channels = channels
        self.sigma1 = sigma1
        self.sigma2 = 2 * sigma1
        self.kernel_size1 = int(2 * round(3 * self.sigma1) + 1)
        self.kernel_size2 = int(2 * round(3 * self.sigma2) + 1)
        self.padding1 = (self.kernel_size1 - 1) // 2
        self.padding2 = (self.kernel_size2 - 1) // 2
        self.weight1 = self.get_gaussian_kernel(self.kernel_size1, self.sigma1)
        self.weight2 = self.get_gaussian_kernel(self.kernel_size2, self.sigma2)

    def get_gaussian_kernel(self, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        kernel = torch.exp(-(xy_grid - mean).pow(2).sum(dim=-1) / (2 * variance))
        kernel = kernel / kernel.sum()
        kernel = kernel.repeat(self.channels, 1, 1, 1)

        return kernel

    @torch.no_grad()
    def forward(self, x):
        gaussian1 = F.conv2d(x, self.weight1.to(x.device), bias=None, stride=1, padding=self.padding1,
                             groups=self.channels)
        gaussian2 = F.conv2d(x, self.weight2.to(x.device), bias=None, stride=1, padding=self.padding2,
                             groups=self.channels)
        return gaussian1 - gaussian2


def apply_dog_filter(batch, freq=50, scale_factor=0.5):

    if batch.size(1) == 3:
        batch = torch.mean(batch, dim=1, keepdim=True)


    downscaled = F.interpolate(batch, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    channels = downscaled.size(1)


    sigma1 = 0.1 + (100 - freq) * 0.1 if freq >= 50 else 0.1 + freq * 0.1

    dog_filter = DoGFilter(channels, sigma1)
    mask = dog_filter(downscaled)

    def dilate(image, kernel_size=3, padding=1):

        dilation_kernel = torch.ones((1, 1, kernel_size, kernel_size)).cuda()


        with torch.no_grad():

            image_unsqueezed = image.float()


            dilation_result = F.conv2d(image_unsqueezed, dilation_kernel, padding=padding)


            dilation_result = (dilation_result >= 1).float()


        return dilation_result


    upscaled_mask = F.interpolate(mask, size=batch.shape[-2:], mode='bilinear', align_corners=False)

    upscaled_mask = upscaled_mask - upscaled_mask.min()
    upscaled_mask = upscaled_mask / upscaled_mask.max() if freq >= 50 else 1.0 - upscaled_mask / upscaled_mask.max()

    upscaled_mask = (upscaled_mask >= 0.5).to(torch.float)

    upscaled_mask = dilate(upscaled_mask, kernel_size=11, padding=5)

    return upscaled_mask[:, 0, ...]