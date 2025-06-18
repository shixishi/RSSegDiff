import os
import glob
from PIL import Image
from torch.utils.data import Dataset


class LoveDADataset(Dataset):
    def __init__(self, root_dir, split, transform, train_flag):
        '''
        root_dir: 数据集的根目录
        split: 'Train' 或 'Test'
        transform: 图像变换操作（用于图像）
        label_transform: 标签变换操作 （用于mask）
        训练阶段和评估阶段的 label_transform 不同
        '''
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'new_new_masks')
        self.image_paths = sorted([os.path.join(self.image_dir, fname) for fname in os.listdir(self.image_dir)])
        self.label_paths = sorted([os.path.join(self.label_dir, fname) for fname in os.listdir(self.label_dir)])
        self.transform = transform
        self.name_list = sorted(os.listdir(self.label_dir))
        self.train_flag = train_flag

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # print("*"*40)
        # print(idx)
        # print("*"*40)
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label_path = self.label_paths[idx]
        label = Image.open(label_path).convert('L')  # 标签是单通道
        label_name = self.name_list[idx]  # 获取标签名  保存生成的mask时需要的文件名

        if self.transform:
            image, label = self.transform(image, label)
        
        if self.train_flag:  # 训练阶段, 不需要返回label_name
            return image, label
        else:               # 推理阶段，需要返回label_name，因为需要保存mask用到，同名
            return image, label, label_name
    
       
