import ttach as tta
from typing import Callable, List, Tuple
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM

from torchvision.models import resnet50
from torchvision import transforms

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


image_path = '/home/ljl/Datasets/ImageNet/train/n03874293/n03874293_133.JPEG'
image = Image.open(image_path)

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

model = resnet50().cuda()


class KLOutputTarget:
    def __init__(self, category, centers):
        self.category = category
        self.center = centers[self.category]

    def __call__(self, model_output):
        return F.kl_div(F.log_softmax(model_output,dim=-1), F.softmax(self.center, dim=-1))