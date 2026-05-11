import os
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import random

SEED = 42

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def transform_perm_to_label(x):
    """转换渗透率到标签"""
    return (np.log10(np.clip(x, 1e-25, None)) + 11) / 2

def transform_label_to_perm(x):
    """转换标签到渗透率"""
    return 10 ** (x * 2 - 11)

class PorousMediaDataset(Dataset):
    """多孔介质数据集"""
    def __init__(self, data_folder, excel_path, use_extra_features=True, transform=None):
        self.transform = transform
        self.use_extra_features = use_extra_features
        
        self.data_info = pd.read_excel(excel_path).values
        self.file_list = sorted([f for f in os.listdir(data_folder) if f.endswith('.mat')])
        self.data_folder = data_folder

        self.labels_real = self.data_info[:, 4].astype(np.float32)
        self.labels = transform_perm_to_label(self.labels_real)
        
        if use_extra_features:
            self.extra_features = self.data_info[:, 6].astype(np.float32)
            self.extra_mean = np.mean(self.extra_features)
            self.extra_std = np.std(self.extra_features)
            self.extra_features = (self.extra_features - self.extra_mean) / (self.extra_std + 1e-8)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_folder, self.file_list[idx])
        data = sio.loadmat(file_path)['tiffStack'].astype(np.float32)
        
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        data = torch.tensor(data).unsqueeze(0)  # 添加通道维度
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            data = self.transform(data)
        
        if self.use_extra_features:
            extra_feature = torch.tensor(self.extra_features[idx], dtype=torch.float32)
            return data, label, extra_feature
        else:
            return data, label

def create_dataloader(data_folder, excel_path, batch_size=32, 
                      shuffle=False, use_extra_features=True, num_workers=4):
    """创建数据加载器"""
    set_seed(SEED)
    
    dataset = PorousMediaDataset(
        data_folder, 
        excel_path, 
        use_extra_features=use_extra_features
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        generator=generator
    )