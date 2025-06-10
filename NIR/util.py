import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def get_mgrid(sidelen, vmin=-1, vmax=1):      #生成一个torch.tensor
    if type(vmin) is not list:
        vmin = [vmin for _ in range(len(sidelen))]
    if type(vmax) is not list:
        vmax = [vmax for _ in range(len(sidelen))]
    tensors = tuple([torch.linspace(vmin[i], vmax[i], steps=sidelen[i]) for i in range(len(sidelen))])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(sidelen))
    return mgrid

def apply_homography(x, h):
    h = torch.cat([h, torch.ones_like(h[:, [0]])], -1)
    h = h.view(-1, 3, 3)
    x = torch.cat([x, torch.ones_like(x[:, 0]).unsqueeze(-1)], -1).unsqueeze(-1)
    o = torch.bmm(h, x).squeeze(-1)
    o = o[:, :-1] / o[:, [-1]]
    return o

def jacobian(y, x):
    B, N = y.shape
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = torch.autograd.grad(y,
                                      x,
                                      grad_outputs=v,
                                      retain_graph=True,
                                      create_graph=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=1).requires_grad_()
    return jacobian


class VideoFitting(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()  #图像的帧数。通道数量，
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        shuffle = torch.randperm(len(self.pixels))
        self.pixels = self.pixels[shuffle]      #像素
        self.coords = self.coords[shuffle]      #像素的坐标

    def get_video_tensor(self):      #读取路径下面的所有图片堆叠一个视频张量
        frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(self.path, frames[i]))
            img = self.transform(img)
            video.append(img)
        return torch.stack(video, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels