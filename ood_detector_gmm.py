# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import math
import random

import torch
from torch import TorchDispatchMode, nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision import models as torchvision_models

import utils
from dataset.imagenet import Imagenet
from dataset.ifood import IFOOD
from dataset.inat import INATURALIST
from dataset.tinyimagenet import TinyImages

import vision_transformer as vits
from progress.bar import Bar
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score


def ood_detector(args):
    utils.init_distributed_mode(args)
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(224),
            # transforms.RandomResizedCrop(224, (0.6, 1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = get_dataset('train', args.data_path, args.k, train_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_in = get_dataset('val', 'onlyin', args.data_path, 0, val_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(val_in)
    val_in_loader = torch.utils.data.DataLoader(
        val_in,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train imgs and {len(val_in)} val imgs.")

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
    mu, sigma, gmm_weights = utils.load_pretrained_weights(model, 
                                            os.path.join(args.pretrained_weights, 'checkpoint0400.pth'), 
                                            args.checkpoint_key, 
                                            args.arch, args.patch_size).cuda()
    det_sigma, sigma_inv = get_det_and_inv(sigma)
    print(f"Model {args.arch} built.")

    classifier = Classifier(args.out_dim, num_labels=1).cuda()
    classifier = classifier.cuda()
    classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu])

    # set optimizer
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=5e-5, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_auroc": 0., 'best_fpr':99.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_auroc = to_restore["best_auroc"]
    best_fpr = to_restore['best_fpr']
    auroc = utils.AverageMeter()
    fpr95 = utils.AverageMeter()
    acc = utils.Accuracy()

    for epoch in range(start_epoch, args.epochs):
        # train_loader.sampler.set_epoch(epoch)

        train(model, classifier, optimizer, train_loader, 
                epoch, args.n_last_blocks, args.avgpool_patchtokens, 
                mu, det_sigma, sigma_inv, gmm_weights, auroc, fpr95, acc)
        scheduler.step()

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            au, fpr = validate_network(val_loader, model, classifier, args.n_last_blocks, args.avgpool_patchtokens, centers, variance, auroc, fpr95, acc)
            print(f"Auroc at epoch {epoch} of the network on the {len(dataset_val)} test images: {au:.2f}%")
            best_auroc = max(best_auroc, au)
            best_fpr = min(best_fpr, fpr)
            print(f'Max auroc so far: {best_auroc:.2f}%, min fpr so far:{best_fpr:.2f}')
            # log_stats = {**{k: v for k, v in log_stats.items()},
            #              **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            # with (Path(args.output_dir) / "log.txt").open("a") as f:
            #     f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_au": best_auroc,
                'best_fpr': best_fpr
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test auroc: {acc:.1f}".format(acc=best_auroc))


def train(model, classifier, optimizer, loader, epoch, n, avgpool, mu, det_sigma, sigma_inv, gmm_weights, auroc, fpr95, acc):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    classifier.train()
    num_iter = len(loader)
    results, targets = [], []
    acc.reset()

    for idx, (inp, target, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                _, output = model(inp)

        energy = get_energy_score(mu, det_sigma, sigma_inv, gmm_weights, output)
        energy, labels = dropout_energy(targets, energy)
        output = classifier(energy)

        # compute cross entropy loss
        loss = F.binary_cross_entropy_with_logits(output, labels.float())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        output, labels = output.cpu(), labels.cpu()
        acc.update(torch.sigmoid(output) >= 0.5, labels)

        output = torch.sigmoid(output).data.numpy()
        labels = labels.numpy()
        results.extend(output)
        targets.extend(labels)

        auroc = calc_auroc(output, labels)
        fpr95 = calc_fpr(output, labels)

        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=acc.get_top1())
        metric_logger.update(auroc=auroc)
        metric_logger.update(fpr95=fpr95)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    auroc = calc_auroc(results, targets)
    fpr95 = calc_fpr(results, targets)
    print(f"auroc:{auroc}, fpr95:{fpr95}")


@torch.no_grad()
def validate_network(val_loader, model, classifier, n, avgpool, mu, det_sigma, sigma_inv, gmm_weights):
    classifier.eval()
    results, targets = [], []

    for idx, (inp, _, target) in enumerate(val_loader):
        
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), 
                                torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                _, output = model(inp)
        
        energy = get_energy_score(mu, det_sigma, sigma_inv, gmm_weights, output)
        energy, labels = dropout_energy(targets, energy)
        output = classifier(output)
        # loss = nn.CrossEntropyLoss()(output, target)
        loss = F.binary_cross_entropy_with_logits(output, target.float())

        output, labels = output.cpu(), labels.cpu()

        output = torch.sigmoid(output).data.numpy()
        labels = labels.numpy()
        results.extend(output)
        targets.extend(labels)

        auroc = calc_auroc(output, labels)
        fpr95 = calc_fpr(output, labels)

    auroc = calc_auroc(results, targets)
    fpr95 = calc_fpr(results, targets)
    print(f"auroc:{auroc}, FPR95:{fpr95}")
    return auroc * 100, fpr95 * 100


def get_energy_score(mu, det_sigma, sigma_inv, gmm_weights, x):
    '''
    Args:
        x: features of input with shape (kernels, dimensions)
        mu: centers of gmm of all classes with shape (classes, kernels, dimensions)
        det_sigma: Determinant of covariance matrix with shape (classes, kernels)
        sigma_inv: The inverse matrix of sigma which has shape (classes, kernels, dimensions, dimensions)
        gmm_weights: weights of gmm with shape (classes, kernels)
    '''
    cls, kers, dims = mu.shape
    # expand x with the same shape with mu
    x = x.unsqueeze(0).expand_as(mu)

    # reshape for calculation
    x = x.reshape(-1, 1, dims)
    mu = mu.reshape(-1, 1, dims)
    det_sigma = det_sigma.reshape(-1, 1)
    sigma_inv = sigma_inv.reshape(-1, dims, dims)
    gmm_weights = gmm_weights.reshape(-1, 1)

    # calculate the denominator of gmm: √|2πΣ|
    denom = torch.sqrt(torch.abs(det_sigma) * torch.pi * 2)

    # calculate the numerator of gmm: exp(-(x-μ)Σ^(-1)(x-μ)^T)
    numer = torch.bmm((x - mu), sigma_inv)
    numer = torch.exp(-torch.bmm(numer, (x - mu).permute(0, 2, 1))).squeeze(-1)
    
    # calculate the energy score: -log(sum(w * N(μ, Σ)))
    energy = (numer / denom * gmm_weights).reshape(cls, kers).sum(-1)
    energy = -torch.log(energy)

    return energy


def get_det_and_inv(sigma):
    '''
    Args:
        sigma: variances of gmm of all classes with shape (classes, kernels, dimensions, dimensions)
    '''
    cls, kers, dims, _ = sigma.shape
    sigma = sigma.reshape(-1, dims, dims)
    det_sigma = torch.det(sigma).reshape(cls, kers)
    sigma_inv = torch.linalg.inv(sigma).reshape(cls, kers, dims, dims)
    return det_sigma, sigma_inv


def dropout_energy(targets, inputs):
    drop = nn.Dropout(p=1)
    classes = torch.unique(targets).tolist()
    id_cls = torch.tensor(random.sample(classes, k=len(classes) // 2))
    id_cls_ = id_cls.repeat(inputs.shape[0]).reshape(-1, len(classes) // 2)
    
    bin_labels = (id_cls_.T == targets).sum(0)
    inputs[:, id_cls] = drop(inputs[:, id_cls])
    return inputs, bin_labels


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if 'center_loss' in state_dict.keys():
            centers = state_dict['center_loss']['centers']
            gmm_weights = state_dict['center_loss']['centers']

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


def get_dataset(mode, data_path, k, transform):
    if 'ImageNet' in data_path:
        return Imagenet(mode, data_path, k, transform)   


class Classifier(nn.Module):
    def __init__(self, embed_dim, num_labels):
        super().__init__()
        self.embed_dim = embed_dim
        # self.model = vits.vit_onelayer(embed_dim=self.embed_dim)

        self.classifier = nn.Sequential(nn.Linear(embed_dim, embed_dim),
        nn.LeakyReLU(),
        nn.Linear(embed_dim, embed_dim * 3),
        nn.LeakyReLU(),
        nn.Linear(embed_dim * 3, embed_dim),
        nn.LeakyReLU(),
        nn.Linear(embed_dim, num_labels))

    def forward(self, x):
        # x = self.model(x)
        # x = x.unsqueeze(1)
        # xt = x.permute(0, 2, 1).detach()

        # x = torch.bmm(x, xt)
        out = self.classifier(x).reshape(x.shape[0])
        return out


def calc_fpr(scores, trues):
    tpr95=0.95
    fpr, tpr, thresholds = roc_curve(trues, scores)
    fpr0=0
    tpr0=0
    for i,(fpr1,tpr1) in enumerate(zip(fpr,tpr)):
        if tpr1>=tpr95:
            break
        fpr0=fpr1
        tpr0=tpr1
    fpr95 = ((tpr95-tpr0)*fpr1 + (tpr1-tpr95)*fpr0) / (tpr1-tpr0)
    return fpr95

def calc_auroc(scores, trues):
    #calculate the AUROC
    result = roc_auc_score(trues, scores)

    return result


def get_lr(reset_times, epochs, iterations, iters, lr_init, lr_min, warm_up_epoch=0):
        warm_step, lr_gap = iterations * warm_up_epoch, lr_init - lr_min
        if iters < warm_step:
            lr = lr_gap / warm_step * iters
        else:
            lr_lessen = int((epochs - warm_up_epoch) / reset_times * iterations)
            lr = 0.5 * ((math.cos((iters + 1 - warm_step) % lr_lessen / lr_lessen * math.pi)) + 1) * lr_gap  + lr_min
        
        return lr / lr_init


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
    parser.add_argument("--lr", default=0.05, type=float, help="""Learning rate at the beginning of
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
    ood_detector(args)
    # calculate_var_inverse(args)
