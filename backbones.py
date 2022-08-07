import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import torchvision_resnet
from wrn import WideResNet
import vision_transformer as vits

def select_backbone(backbone, pretrained=True, num_classes=50, patch_size=None, is_student=False):
    assert backbone in ['resnet50', 'wideresnet',
                        'vit_tiny', 'vit_small',
                        'vit_base']

    if 'resnet50' in backbone:
        return Resnet(backbone, num_classes, pretrained)
    elif 'wideresnet' in backbone:
        return WideResNet(40, num_classes, 2, dropRate=0.3)
    elif 'vit' in backbone:
        return vits.__dict__[backbone](patch_size=patch_size, is_student=is_student)


class Resnet(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True):
        super().__init__()
        model = getattr(torchvision_resnet, backbone)(pretrained)
        self.embed_dim = list(model.children())[-1].in_features

        self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # self.change_dilation([1, 1, 1, 2, 4])
        self.query = nn.Linear(2048, 8192)
        self.classifier = nn.Linear(2048, num_classes)

        if not pretrained:
            initialize_weights(self)
            for m in self.modules():
                if isinstance(m, torchvision_resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision_resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        query = self.query(x)
        output = self.classifier(x)
        return output, query

    def get_query(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        query = self.query(x)
        return query

    def get_feats(self, x):
        feat0 = self.layer0(x)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        return feat0, feat1, feat2, feat3, feat4

    def change_dilation(self, params):
        assert isinstance(params, (tuple, list))
        assert len(params) == 5

        self._change_stage_dilation(self.layer0, params[0])
        self._change_stage_dilation(self.layer1, params[1])
        self._change_stage_dilation(self.layer2, params[2])
        self._change_stage_dilation(self.layer3, params[3])
        self._change_stage_dilation(self.layer4, params[4])

    def _change_stage_dilation(self, stage, param):
        for m in stage.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (param, param)
                    m.dilation = (param, param)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight = torch.nn.Parameter(bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0]))


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)