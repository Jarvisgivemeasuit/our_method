import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import Dataset
from PIL import Image


class IFOOD(Dataset):
    '''
    Generating a torch dataset using Imagenet for training or validation.
    Args:
        mode: Specifies the dataset to train or test.
        data_domain: Determines this dataset as IND or OOD.
    '''
    def __init__(self, mode, domain, data_dir, k, transform) -> None:
        super().__init__()
        assert mode in ['train', 'val']

        self.mode = mode
        self.ifood_path = data_dir
        self.data_list = pd.read_csv(os.path.join(self.ifood_path, f'{mode}_labels.csv'))
        self.img_list = []
        self.transform = transform

        if domain == 'onlyin':
            with open(f'/home/ljl/Documents/our_method/dataset/ind_ifood_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls_idx = int(line.strip())
                    cls_img_list = self.data_list[self.data_list['label'] == cls_idx]
                    cls_img_list = np.array(cls_img_list)
                    cls_img_list[:, 1] = idx
                    cls_img_list = cls_img_list.tolist()
                    self.img_list = self.img_list + cls_img_list

        elif domain == 'onlyout':
            with open(f'/home/ljl/Documents/our_method/dataset/ood_{self.mode}_ifood_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls_idx = int(line.strip())
                    cls_img_list = self.data_list[self.data_list['label'] == cls_idx]
                    cls_img_list = np.array(cls_img_list)
                    cls_img_list[:, 1] = -1
                    cls_img_list = cls_img_list.tolist()
                    self.img_list = self.img_list + cls_img_list

        elif domain == 'both':
            with open(f'/home/ljl/Documents/our_method/dataset/ind_ifood_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls_idx = int(line.strip())
                    cls_img_list = self.data_list[self.data_list['label'] == cls_idx]
                    cls_img_list = np.array(cls_img_list)
                    cls_img_list[:, 1] = idx
                    cls_img_list = cls_img_list.tolist()
                    self.img_list = self.img_list + cls_img_list

            with open(f'/home/ljl/Documents/our_method/dataset/ood_{self.mode}_ifood_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls_idx = int(line.strip())
                    cls_img_list = self.data_list[self.data_list['label'] == cls_idx]
                    cls_img_list = np.array(cls_img_list)
                    cls_img_list[:, 1] = -1
                    cls_img_list = cls_img_list.tolist()
                    self.img_list = self.img_list + cls_img_list

    def __getitem__(self, idx):
        img_name, cls_label = self.img_list[idx]
        domain_label = 1 if cls_label == -1 else 0

        img = Image.open(os.path.join(self.ifood_path, f'{self.mode}_set', img_name)).convert('RGB')
        img = self.transform(img)

        return img, cls_label, domain_label

    def __len__(self):
        return len(self.img_list)

    def train_transformations(self, image):
        compose = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.CLAHE(),
            A.Blur(blur_limit=3, p=.5),
            A.GridDistortion(p=.5),
            A.Resize(500, 500, p=1),
            A.RandomSizedCrop((300, 500), 224, 224, p=1),

            A.Normalize()
        ])
        return compose(image=image)['image']

    def valid_transformations(self, image):
        norm = A.Compose([
            A.Resize(224, 224, p=1),
            A.Normalize()
        ])
        return norm(image=image)['image']
