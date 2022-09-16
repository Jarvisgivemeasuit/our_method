import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models as torchvision_models
from torchvision import transforms

import utils
from dataset.imagenet import Imagenet
import vision_transformer as vits
from progress.bar import Bar


def calculate_ind_acc(args):
    utils.init_distributed_mode(args)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = get_dataset('val', 'onlyin', args.data_path, 0, transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    dataloader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

     # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    model = utils.MultiCropWrapper(
        model,
        vits.DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    model.cuda()
    model.eval()
    # load weights to evaluate
    centers, gmm_weights = load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    clses, kers, dims = centers.shape
    acc = Accuracy()

    num_batch = len(dataloader)
    bar = Bar('Calculating ind_acc:', max=num_batch)

    for idx, (x, cls, _) in enumerate(dataloader):
        x, cls = x.cuda(), cls.cuda()
        with torch.no_grad():
            _, q = model(x)
        bs = q.shape[0]

        centers_ = centers.unsqueeze(0).expand(bs, clses, kers, dims).reshape(-1, dims).cuda()
        q = q.unsqueeze(1).expand(bs, clses, kers * dims).reshape(-1, dims)

        results = torch.cosine_similarity(q, centers_, dim=1).reshape(-1, clses, kers)
        results = (results * gmm_weights.cuda()).sum(-1)
        # results = torch.pairwise_distance(q, centers).reshape(-1, 100)
        results = torch.argmax(results, 1)
        # print(results, cls)

        acc.update(results, cls)
        bar.suffix = f'{idx+1}/{num_batch}, acc:{acc.get_top1()}'
        bar.next()
    bar.finish()

        
class Accuracy:
    def __init__(self, eps=1e-7):
        self.num_correct = 0
        self.num_instance = 0

        self.eps = eps

    def update(self, pred, target):

        self.num_correct += (pred == target).sum().item()
        self.num_instance += target.shape[0]

    def get_top1(self):
        return self.num_correct / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")

        if 'center_loss' in state_dict.keys():
            centers = state_dict['center_loss']['centers']
            gmm_weights = state_dict['center_loss']['gmm_weights']

        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        return centers, gmm_weights


def get_dataset(mode, domain, data_path, k, transform):
    if 'ImageNet' in data_path:
        return Imagenet(mode, domain, data_path, k, transform)
    elif 'ifood' in data_path:
        return IFOOD(mode, domain, data_path, k, transform)
    else:
        return INATURALIST(mode, domain, data_path, k, transform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=40, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    args = parser.parse_args()
    calculate_ind_acc(args)