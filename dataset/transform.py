import torch
import random
import numpy as np
from torchvision.transforms import Compose
import torchvision.transforms.functional as TF


# PyTorch 原生的 transforms 对 segmentation 任务支持较差，尤其在同时变换图像和 mask 时需要你自定义 transform 类

# 保留
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask

# 保留
class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        return img, mask

class Resize:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img, mask):
        img = TF.resize(img, self.size)
        mask = TF.resize(mask, self.size)
        return img, mask


# 保留
class ToTensor:
    def __call__(self, img, mask):
        img = TF.to_tensor(img)  # (C, H, W) 缩放到0-1
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)    # (H, W) 数值不缩放
        return img, mask

# TODO： 改写这个归一化类 
class Normalize:
    def __call__(self, img, mask):
        img = (img * 2) - 1  # 归一化到 [-1,1]
        # mask 不处理
        return img, mask

# 继承 Compose 类，覆写__call__方法
class MyCompose(Compose):
    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

def get_train_transform():
    return MyCompose([
        # Resize((128, 128)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
        Normalize(),
    ])

def get_eva_transform():
    return MyCompose([
        ToTensor(),
        Normalize(),
    ])


