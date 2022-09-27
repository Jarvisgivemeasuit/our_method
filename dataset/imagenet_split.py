import os
import cv2
import time
import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class Imagenet(Dataset):
    '''
    Generating a torch dataset using Imagenet for training or validation.
    Args:
        mode: Specifies the dataset to train or test.
        data_domain: Determines this dataset as IND or OOD.
    '''
    def __init__(self, mode, domain, data_path, k, transform=None) -> None:
        super().__init__()
        assert mode in ['train', 'val']

        self.mode = mode
        self.transform = transform
        self.imagenet_path = os.path.join(data_path, mode)
        self.classes, self.img_list = {}, []

        if domain == 'onlyin':
            with open(f'dataset/ind_imagenet_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls_name = line.strip()
                    self.classes[cls_name] = [idx, 0]

                    cls_img_list = os.listdir(os.path.join(self.imagenet_path, cls_name))
                    cls_img_list = [os.path.join(cls_name, k) for k in cls_img_list]
                    self.img_list = self.img_list + cls_img_list

        elif domain == 'onlyout':
            with open(f'dataset/ood_{self.mode}_imagenet_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls_name = line.strip()
                    self.classes[cls_name] = [idx, 1]

                    cls_img_list = os.listdir(os.path.join(self.imagenet_path, cls_name))
                    cls_img_list = [os.path.join(cls_name, k) for k in cls_img_list]
                    self.img_list = self.img_list + cls_img_list

        elif domain == 'both':
            with open(f'dataset/ind_imagenet_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls_name = line.strip()
                    self.classes[cls_name] = [idx, 0]

                    cls_img_list = os.listdir(os.path.join(self.imagenet_path, cls_name))
                    cls_img_list = [os.path.join(cls_name, k) for k in cls_img_list]
                    self.img_list = self.img_list + cls_img_list

            with open(f'dataset/ood_{self.mode}_imagenet_{k}.txt', 'r') as f:
                for idx, line in enumerate(f):
                    cls_name = line.strip()
                    self.classes[cls_name] = [idx, 1]

                    cls_img_list = os.listdir(os.path.join(self.imagenet_path, cls_name))
                    cls_img_list = [os.path.join(cls_name, k) for k in cls_img_list]
                    self.img_list = self.img_list + cls_img_list

    def __getitem__(self, idx):

        img_name = self.img_list[idx]

        cls_name = img_name.split('/')[0]
        cls_label, domain_label = self.classes[cls_name]

        img = Image.open(os.path.join(self.imagenet_path, img_name)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = self.train_transforms(img) if self.mode == 'train' else \
                self.val_transforms(img)

        return img, cls_label, domain_label

    def __len__(self):
        return len(self.img_list)

    def train_transforms(self, image):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(224, (0.6, 1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return transform(image)

    def val_transforms(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return transform(image)

    # def train_transformations(self, image):
    #     compose = A.Compose([
    #         A.RandomRotate90(p=0.5),
    #         A.Transpose(p=0.5),
    #         A.CLAHE(),
    #         A.ShiftScaleRotate(shift_limit=0.0625, 
    #                         scale_limit=0.50, 
    #                         rotate_limit=45, p=.75),
    #         A.Blur(blur_limit=3, p=.5),
    #         A.OpticalDistortion(p=.5),
    #         A.GridDistortion(p=.5),
    #         A.HueSaturationValue(p=.5),
    #         A.Resize(500, 500, p=1),
    #         A.RandomSizedCrop((300, 500), 224, 224, p=1),

    #         A.Normalize()
    #     ])
    #     return compose(image=image)['image']

    # def valid_transformations(self, image):
    #     norm = A.Compose([
    #         A.Resize(224, 224, p=1),
    #         A.Normalize()
    #     ])
    #     return norm(image=image)['image']

