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
from socket import timeout
import sys
import argparse
import math
import numpy as np

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision import models as torchvision_models

import utils
from dataset.imagenet_split import Imagenet
from dataset.ifood import IFOOD
from dataset.inat import INATURALIST

from backbones import select_backbone
from progress.bar import Bar
from sklearn.metrics import roc_curve, roc_auc_score


def ood_detector(args):
    # utils.init_distributed_mode(args)
    # cudnn.benchmark = True

    # ============ preparing data ... ============
    # CIFAR100
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])

    val_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                        transforms.RandomCrop(32),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])

    dataset_val_in = get_dataset('val', 'onlyin', args.data_path_in, 0, val_transform)
    dataset_train_in = get_dataset('train', 'onlyin', args.data_path_in, 0, train_transform)
    dataset_val_out = get_dataset('val', 'onlyout', args.data_path_out, 0, val_transform)
    dataset_train_out = get_dataset('train', 'onlyout', args.data_path_out, 0, train_transform)

    val_in_loader = torch.utils.data.DataLoader(dataset_val_in, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        timeout=2
                        )
    val_out_loader = torch.utils.data.DataLoader(dataset_val_out, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        timeout=2
                        )

    train_in_loader = torch.utils.data.DataLoader(dataset_train_in, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        timeout=2
                        )
    train_out_loader = torch.utils.data.DataLoader(dataset_train_out, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        timeout=2
                        )

    print(f'''Data loaded with {len(dataset_train_in)} in-distribution train images,
              {len(dataset_train_out)} out-of-distribution train images,
              {len(dataset_val_in)} in-distribuion val images and
              {len(dataset_val_out)} out-of-distribution val images''')

    # ============ building network ... ============
    model = select_backbone(args.arch, num_classes=args.num_labels).cuda()
    state_dict = torch.load(args.pretrained_weights)
    model.load_state_dict(state_dict)
    print(f"Model {args.arch} built.")

    classifier = Classifier(args.out_dim, num_labels=1).cuda()
    classifier = classifier.cuda()
    # classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu])

    print('Classifier built.')

    # if args.evaluate:
    #     utils.load_pretrained_linear_weights(classifier, args.arch, args.patch_size)
    #     test_stats = validate_network(val_loader, model, classifier, args.n_last_blocks, args.avgpool_patchtokens)
    #     print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #     return None

    # set optimizer
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=5e-5, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    make_sure_path_exists(args.output_dir)

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
    acc = utils.Accuracy()

    for epoch in range(start_epoch, args.epochs):
        train(model, classifier, optimizer, train_in_loader, train_out_loader, epoch, acc)
        scheduler.step()

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            au, fpr = validate_network(model, classifier, val_in_loader, val_out_loader, acc)
            print(f"Auroc at epoch {epoch} of the network: {au:.2f}%")

            best_auroc, best_fpr = max(best_auroc, au), min(best_fpr, fpr)
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


def train(model, classifier, optimizer, loader_in, loader_out, epoch, acc):
    classifier.train()
    num_iter = len(loader_in)
    bar = Bar('Training classifier:', max=num_iter)
    results, targets = [], []
    acc.reset()
    # loader_out.dataset.offset = np.random.randint(len(loader_out.dataset))
    idx = 0
    for data_in, data_out in zip(loader_in, loader_out):
        # move to gpu
        inp = torch.cat((data_in[0], data_out[0]), 0)
        target = torch.cat((torch.zeros(data_in[1].shape), torch.ones(data_out[1].shape)))

        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model.intermediate_forward(inp)
            output = output.reshape(output.shape[0], -1)
        output = classifier(output)

        # compute cross entropy loss
        loss = F.binary_cross_entropy_with_logits(output, target.float())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        output, target = output.cpu(), target.cpu()
        output, target = torch.sigmoid(output).data.numpy(), target.numpy()

        acc.update(output >= 0.5, target)

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
        idx += 1
    bar.finish()

    auroc = calc_auroc(results, targets)
    fpr95 = calc_fpr(results, targets)
    print(f"auroc:{auroc}, fpr95:{fpr95}")


@torch.no_grad()
def validate_network(model, classifier, loader_in, loader_out, acc):
    classifier.eval()
    num_iter = len(loader_in)
    bar = Bar('Validating:', max=num_iter)
    results, targets = [], []
    acc.reset()

    idx = 0
    for data_in, data_out in zip(loader_in, loader_out):
        
        # move to gpu
        inp = torch.cat((data_in[0], data_out[0]), 0)
        target = torch.cat((torch.zeros(data_in[1].shape), torch.ones(data_out[1].shape)))

        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model.intermediate_forward(inp)
            output = output.reshape(output.shape[0], -1)
            output = classifier(output)

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
        idx += 1
    bar.finish()

    auroc = calc_auroc(results, targets)
    fpr95 = calc_fpr(results, targets)
    print(f"auroc:{auroc}, FPR95:{fpr95}")
    return auroc * 100, fpr95 * 100


def get_dataset(mode, domain, data_path, k, transform):
    if 'ImageNet' in data_path:
        return Imagenet(mode, domain, data_path, k, transform)
    elif 'ifood' in data_path:
        return IFOOD(mode, domain, data_path, k, transform)
    elif 'inat' in data_path:
        return INATURALIST(mode, domain, data_path, k, transform)
    elif 'cifar10/' in data_path and mode == 'train':
        return datasets.CIFAR10(data_path, train=True, transform=transform)
    elif 'cifar10/' in data_path:
        return datasets.CIFAR10(data_path, train=False, transform=transform)
    elif 'cifar100/' in data_path and mode == 'train':
        return datasets.CIFAR100(data_path, train=True, transform=transform)
    elif 'cifar100/' in data_path:
        return datasets.CIFAR100(data_path, train=False, transform=transform)
    elif 'tiny-imagenet-200' in data_path:
        return datasets.ImageFolder(os.path.join(data_path, mode), transform=transform)


class Classifier(nn.Module):
    def __init__(self, embed_dim, num_labels):
        super().__init__()
        self.embed_dim = embed_dim
        # self.model = vits.vit_onelayer(embed_dim=self.embed_dim)

        self.classifier = nn.Sequential(nn.Linear(embed_dim * 8, embed_dim),
        nn.LeakyReLU(),
        nn.Linear(embed_dim, embed_dim * 3),
        nn.LeakyReLU(),
        nn.Linear(embed_dim * 3, embed_dim),
        nn.LeakyReLU(),
        nn.Linear(embed_dim, embed_dim),
        nn.LeakyReLU(),
        nn.Linear(embed_dim, num_labels))

    def forward(self, x):
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


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


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
    parser.add_argument('--data_path_in', default='/path/to/imagenet/', type=str)
    parser.add_argument('--data_path_out', default='/path/to/imagenet/', type=str)
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
