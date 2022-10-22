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
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from progress.bar import Bar
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import models as torchvision_models
from torchvision import transforms

import utils
import vision_transformer as vits
from dataset.ifood import IFOOD
from dataset.imagenet_split import Imagenet
from dataset.inat import INATURALIST
from dataset.tinyimagenet import TinyImages


def ood_detector(args):
    utils.init_distributed_mode(args)
    cudnn.benchmark = True

    # ============ preparing data ... ============
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_val = get_dataset('val', 'both', args.data_path, args.k, val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=True,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:
        utils.load_pretrained_linear_weights(classifier, args.arch, args.patch_size)
        test_stats = validate_network(val_loader, model, classifier, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(224),
            # transforms.RandomResizedCrop(224, (0.6, 1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = get_dataset('train', 'both', args.data_path, args.k, train_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

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
    centers = utils.load_pretrained_weights(model, 
                                            os.path.join(args.pretrained_weights, 'checkpoint0400.pth'), 
                                            args.checkpoint_key, 
                                            args.arch, args.patch_size)[:args.num_labels].cuda()
    variance = torch.load(os.path.join(args.pretrained_weights, 'var_inv_half.pt')).cuda()

    assert centers is not None
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
    # lr = lambda iters: get_lr(reset_times=1, 
    #                                     epochs=args.epochs, 
    #                                     iterations=len(train_loader), 
    #                                     iters=iters, 
    #                                     lr_init=args.lr,
    #                                     lr_min=1e-6, 
    #                                     warm_up_epoch=3)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)

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

        train(model, classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens, centers, variance, auroc, fpr95, acc)
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


def train(model, classifier, optimizer, loader, epoch, n, avgpool, centers, variance, auroc, fpr95, acc):
    classifier.train()
    num_iter = len(loader)
    bar = Bar('Training classifier:', max=num_iter)
    results, targets = [], []
    acc.reset()

    for idx, (inp, _, target) in enumerate(loader):
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
                output = model(inp)

        # output = surrogate_norm(centers, variance, output)
        output = classifier(output)

        # compute cross entropy loss
        # loss = nn.CrossEntropyLoss()(output, target)
        loss = F.binary_cross_entropy_with_logits(output, target.float())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        # output = torch.argmax(output, axis=1).cpu()
        # target = target.cpu()
        # auroc.update(roc_auc_score(target, output))
        # fpr95.update(utils.calc_fpr(target, output))
        # acc.update(output, target)

        output, target = output.cpu(), target.cpu()
        acc.update(torch.sigmoid(output) >= 0.5, target)

        output = torch.sigmoid(output).data.numpy()
        target = target.numpy()
        results.extend(output)
        targets.extend(target)

        auroc = calc_auroc(output, target)
        fpr95 = calc_fpr(output, target)

        bar.suffix = '({batch}/{size}) Epoch:{epoch} | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f} | Acc:{acc:.4f} | AUROC:{auroc:.4f} | FPR95:{fpr:.4f} ｜ LR:{lr:.4f}'.format(
            epoch=epoch,
            batch=idx + 1,
            size=num_iter,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss,
            acc=acc.get_top1(),
            auroc=auroc,
            fpr=fpr95,
            lr=optimizer.param_groups[0]['lr']
            )
        bar.next()
    bar.finish()

    auroc = calc_auroc(results, targets)
    fpr95 = calc_fpr(results, targets)
    print(f"auroc:{auroc}, fpr95:{fpr95}")


@torch.no_grad()
def validate_network(val_loader, model, classifier, n, avgpool, centers, variance, auroc, fpr95, acc):
    classifier.eval()
    num_iter = len(val_loader)
    bar = Bar('Validating:', max=num_iter)
    results, targets = [], []
    acc.reset()

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
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        
        # output = surrogate_norm(centers, variance, output)
        output = classifier(output)
        # loss = nn.CrossEntropyLoss()(output, target)
        loss = F.binary_cross_entropy_with_logits(output, target.float())

        output, target = output.cpu(), target.cpu()
        acc.update(torch.sigmoid(output) >= 0.5, target)

        output = torch.sigmoid(output).data.numpy()
        target = target.numpy()
        results.extend(output)
        targets.extend(target)

        auroc = calc_auroc(output, target)
        fpr95 = calc_fpr(output, target)
        bar.suffix = '({batch}/{size}) | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f} | Acc:{acc:.4f} | AUROC:{auroc:.4f} | FPR95:{fpr:.4f}'.format(
            batch=idx + 1,
            size=num_iter,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss,
            acc=acc.get_top1(),
            auroc=auroc,
            fpr=fpr95,
            )
        bar.next()
    bar.finish()

    auroc = calc_auroc(results, targets)
    fpr95 = calc_fpr(results, targets)
    print(f"auroc:{auroc}, FPR95:{fpr95}")
    return auroc * 100, fpr95 * 100


def surrogate_norm(centers, variance_inv, x):
    x = x.unsqueeze(1).expand(x.shape[0], centers.shape[0], x.shape[1])
    centers = centers.unsqueeze(0).expand_as(x)
    cls_input = torch.bmm((x - centers).permute(1, 0, 2), variance_inv)
    cls_input = cls_input.permute(1, 0, 2)
    # return cls_input
    return x


def surrogate_norm_simple(centers, variance, x):
    x = x.unsqueeze(1).expand(x.shape[0], centers.shape[0], x.shape[1])
    centers = centers.unsqueeze(0).expand_as(x)
    cls_input = (x - centers) @ variance
    return cls_input
    # return x


def get_dataset(mode, domain, data_path, k, transform):
    if 'ImageNet' in data_path:
        return Imagenet(mode, domain, data_path, k, transform)
    elif 'ifood' in data_path:
        return IFOOD(mode, domain, data_path, k, transform)
    elif 'inat' in data_path:
        return INATURALIST(mode, domain, data_path, k, transform)
    elif 'cifar10' in data_path and mode == 'train':
        return datasets.CIFAR10(data_path, train=True, transform=transform)
    elif 'cifar10' in data_path:
        return datasets.CIFAR10(data_path, train=False, transform=transform)
    elif 'cifar100' in data_path and mode == 'train':
        return datasets.CIFAR100(data_path, train=True, transform=transform)
    elif 'cifar100' in data_path:
        return datasets.CIFAR100(data_path, train=False, transform=transform)
    elif 'tiny-imagenet-200' in data_path:
        return datasets.ImageFolder(os.path.join(data_path, mode), transform=transform)


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
        nn.Linear(embed_dim, embed_dim),
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
