import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import Dataset
from PIL import Image


class INATURALIST(Dataset):
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
        self.inat_path = data_dir
        self.data_list = []
        with open(os.path.join(self.inat_path, f'{mode}_list_310.txt'), 'r') as f:
            for line in f:
                self.data_list.append(line.strip())
        
        self.img_list = []
        self.transform = transform

        if domain == 'onlyin':
            with open(f'/home/ljl/Documents/our_method/dataset/ind_inat_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls = line.strip()
                    cls_img_list = []
                    for file_name in self.data_list:
                        if cls in file_name:
                            cls_img_list.append([file_name, idx])
                    self.img_list = self.img_list + cls_img_list

        elif domain == 'onlyout':
            with open(f'/home/ljl/Documents/our_method/dataset/ood_{self.mode}_inat_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls = line.strip()
                    cls_img_list = []
                    for file_name in self.data_list:
                        if cls in file_name:
                            cls_img_list.append([file_name, -1])
                    self.img_list = self.img_list + cls_img_list

        elif domain == 'both':
            with open(f'/home/ljl/Documents/our_method/dataset/ind_inat_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls = line.strip()
                    cls_img_list = []
                    for file_name in self.data_list:
                        if cls in file_name:
                            cls_img_list.append([file_name, idx])
                    self.img_list = self.img_list + cls_img_list

            with open(f'/home/ljl/Documents/our_method/dataset/ood_{self.mode}_inat_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls = line.strip()
                    cls_img_list = []
                    for file_name in self.data_list:
                        if cls in file_name:
                            cls_img_list.append([file_name, -1])
                    self.img_list = self.img_list + cls_img_list

    def __getitem__(self, idx):
        img_name, cls_label = self.img_list[idx]
        domain_label = 1 if cls_label == -1 else 0

        img = Image.open(os.path.join(self.inat_path, 'Aves', img_name)).convert('RGB')
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
