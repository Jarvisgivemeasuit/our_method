import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

import torch
from torch import nn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms

import utils
from dataset.imagenet import Imagenet
from dataset.ifood import IFOOD
from dataset.inat import INATURALIST
from backbones import select_backbone

import vision_transformer as vits
from progress.bar import Bar


def get_dataset(mode, domain, data_path, k, transform):
    if 'ImageNet' in data_path:
        return Imagenet(mode, domain, data_path, k, transform)
    elif 'ifood' in data_path:
        return IFOOD(mode, domain, data_path, k, transform)
    else:
        return INATURALIST(mode, domain, data_path, k, transform)

        
def calculate_var(args, gamma=0.9):
    utils.init_distributed_mode(args)
    transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224), interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = get_dataset('train', 'onlyin', args.data_path, args.k, transform=transform)
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
    if 'center' in args.pretrained_weights:
        model = select_backbone(args.arch, False, args.num_labels)
        state_dict = torch.load(args.pretrained_weights)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        centers = torch.load(args.pretrained_weights.replace('.pt', '-center.pth'))
        print('this is supervised model')
    elif args.arch in vits.__dict__.keys():
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
        model = utils.MultiCropWrapper(
        model,
        vits.DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        )

        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        centers = state_dict['center_loss']['centers'][:args.num_labels]

    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    model.cuda()
    model.eval()

    var = torch.zeros(centers.shape[0], centers.shape[-1], centers.shape[-1]).cuda()
    print(var.shape)

    num_batch = len(dataloader)
    bar = Bar('Calculating variance:', max=num_batch)

    for idx, (x, cls, _) in enumerate(dataloader):
        x, cls = x.cuda(), cls.cuda()
        with torch.no_grad():
            if 'center' in args.pretrained_weights:
                q = model.get_query(x)
            else:
                q = model(x)
        for i in range(centers.shape[0]): # Update variance of class i 
            if len(cls[cls == i]) == 0:
                continue
            else:
                q_ = q[cls == i].unsqueeze(1)
                k = centers[i].reshape(1, 1 ,-1).expand_as(q_).cuda()
            var_ = torch.bmm((q_ - k).permute(0, 2, 1), q_ - k)
            var_ = torch.mean(torch.abs(var_), axis=0)
            
            # calculate Σ^-1/2
            l, Q = torch.linalg.eig(var_)
            L = torch.diag(l ** -0.5)
            sigma_inv_half = torch.mm(torch.mm(Q, L), Q.inverse())

            var[i] = var[i] * gamma + sigma_inv_half * (1 - gamma)

        bar.suffix = f'{idx+1}/{num_batch}'
        bar.next()
    bar.finish()
    torch.save(var, os.path.join(args.output_dir, 'var_inv_half.pt'))
    print('Variance calculating complete.')

    # results = torch.zeros(var.shape[0], var.shape[1])
    # bar = Bar('Spliting var:', max=var.shape[0])

    # for i in range(var.shape[0]):
    #     U, S, Vh = torch.linalg.svd(var[i])
    #     results[i] = U.mm(torch.sqrt(S.reshape(-1, 1))).squeeze(1)
    #     bar.suffix = f'{i+1}/{var.shape[0]}'
    #     bar.next()
    # bar.finish()
    # torch.save(results, os.path.join(args.output_dir, 'std_stat.pt'))
    return var


def cal_split():
    val = torch.load('/home/et21-lijl/Documents/dino/model_saving/var.pt')
    results = torch.zeros(val.shape[0], val.shape[1])
    bar = Bar('Spliting val:', max=val.shape[0])
    for i in range(val.shape[0]):
        U, S, Vh = torch.linalg.svd(val[i])
        results[i] = U.mm(torch.sqrt(S.reshape(-1, 1))).squeeze(1)
        bar.suffix = f'{i+1}/{val.shape[0]}'
        bar.next()
    bar.finish()
    torch.save(results, './model_saving/std.pt')


# def calculate_center(args, num_classes=50):
#     utils.init_distributed_mode(args)
#     transform = pth_transforms.Compose([
#         pth_transforms.Resize((224, 224), interpolation=3),
#         pth_transforms.ToTensor(),
#         pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     dataset_train = get_dataset('train', 'onlyin', args.data_path, args.k, transform=transform)
#     sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
#     dataloader = torch.utils.data.DataLoader(
#         dataset_train,
#         sampler=sampler,
#         batch_size=args.batch_size_per_gpu,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )

#      # ============ building network ... ============
#     # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
#     if args.arch in vits.__dict__.keys():
#         model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
#         embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
#     # if the network is a XCiT
#     elif "xcit" in args.arch:
#         model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
#         embed_dim = model.embed_dim
#     # otherwise, we check if the architecture is in torchvision models
#     elif args.arch in torchvision_models.__dict__.keys():
#         model = torchvision_models.__dict__[args.arch]()
#         embed_dim = model.fc.weight.shape[1]
#         model.fc = nn.Identity()
#     else:
#         print(f"Unknow architecture: {args.arch}")
#         sys.exit(1)

#     model = utils.MultiCropWrapper(
#         model,
#         vits.DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
#     )
#     model.cuda()
#     model.eval()
#     # load weights to evaluate

#     num_batch = len(dataloader)
#     bar = Bar('Calculating center:', max=num_batch)
#     centers = torch.zeros(num_classes, args.out_dim).cuda()
#     counts = torch.zeros(num_classes)

#     for idx, (x, cls) in enumerate(dataloader):
#         x, cls = x.cuda(), cls.cuda()
#         with torch.no_grad():
#             q = model(x)
        
#         for i in range(num_classes):
#             q_ = q[cls == i]
#             counts[cls] += q_.shape[0]
#             centers[cls] += torch.sum(q_, dim=0)

#         bar.suffix = f'{idx+1}/{num_batch}'
#         bar.next()
#     bar.finish()
#     centers = (centers.t() / counts.cuda()).t()
#     print(centers)
#     torch.save(centers, './model_saving/ifood_surr/centers.pt')

#     # calculate_var
#     gamma = 0.9
#     var = torch.zeros(centers.shape[0], centers.shape[-1], centers.shape[-1])
#     bar = Bar('Calculating variance:', max=num_batch)

#     for idx, (x, cls) in enumerate(dataloader):
#         x, cls = x.cuda(), cls.cuda()
#         with torch.no_grad():
#             q = model(x)
#         for i in range(centers.shape[0]): # Update variance of class i 
#             if len(cls[cls == i]) == 0:
#                 continue
#             else:
#                 q_ = q[cls == i].unsqueeze(1).cpu()
#                 k = centers[i].reshape(1, 1 ,-1).expand_as(q_).cpu()
#             var_ = torch.bmm((q_ - k).permute(0, 2, 1), q_ - k)
#             var_ = torch.mean(var_, axis=0)

#             var[i] = var[i] * gamma + var_ * (1 - gamma)
#             # if len(cls[cls == i]) == 0:
#             #     continue
#             # else:
#             #     q_ = q[cls == i].cpu()
#             #     k = centers[i].unsqueeze(0).expand_as(q_)
#             # var_ = (q_ - k) ** 2
#             # var_ = torch.sqrt(torch.mean(var_, axis=0))

#             # var[i] = var[i] * gamma + var_ * (1 - gamma)
#         bar.suffix = f'{idx+1}/{num_batch}'
#         bar.next()
#     bar.finish()
#     torch.save(var, './model_saving/ifood_surr/var_stat.pt')
#     print('Variance saving complete.')

#     results = torch.zeros(var.shape[0], var.shape[1])
#     bar = Bar('Spliting var:', max=var.shape[0])
#     for i in range(var.shape[0]):
#         U, S, Vh = torch.linalg.svd(var[i])
#         results[i] = U.mm(torch.sqrt(S.reshape(-1, 1))).squeeze(1)
#         bar.suffix = f'{i+1}/{var.shape[0]}'
#         bar.next()
#     bar.finish()
#     torch.save(results, './model_saving/ifood_surr/std_stat.pt')


# def calc_var(centers, cls, q, var, gamma):
#     for i in range(centers.shape[0]): # Update variance of class i 
#         if len(cls[cls == i]) == 0:
#             continue
#         else:
#             q_ = q[cls == i].unsqueeze(1).cpu()
#             k = centers[i].reshape(1, 1 ,-1).expand_as(q_)
#         var_ = torch.bmm((q_ - k).permute(0, 2, 1), q_ - k)
#         var_ = torch.mean(var_, axis=0)

#         var[i] = var[i] * gamma + var_ * (1 - gamma)


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
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--k', default=1, type=int, help='Number of random split')
    args = parser.parse_args()
    calculate_var(args)