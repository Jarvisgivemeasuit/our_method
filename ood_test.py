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
from backbones import select_backbone
from progress.bar import Bar
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score


def test(args):
    # ============ building network ... ============
    model = select_backbone(args.arch, num_classes=args.num_labels).cuda()
    state_dict = torch.load(args.pretrained_weights)
    model.load_state_dict(state_dict)
    print(f"Model {args.arch} built.")

    classifier = Classifier(args.out_dim, num_labels=1).cuda()
    classifier = classifier.cuda()
    checkpoint = torch.load(args.classifier_weights, map_location="cpu")
    classifier.load_state_dict(checkpoint['state_dict'], strict=False)
    print('Classifier built.')

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    in_data = get_dataset('test', args.data_path_in, test_transform)
    in_loader = torch.utils.data.DataLoader(in_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        timeout=2
                        )

    print(f'Data loaded with {len(in_data)} in-distribuion test images.')
    results_in = get_scores(model, classifier, in_loader)

    # Texture dataset
    ood_data = datasets.ImageFolder('/home/ljl/Datasets/dtd/images', transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    print(f'Data loaded with {len(ood_data)} out-of-distribuion test images.')
    auroc, fpr95, aupr = get_results(model, classifier, ood_loader, results_in)
    print("Texture Detection")
    print(f"AUROC:{auroc:.2f}, FPR95:{fpr95:.2f}, AUPR:{aupr:.2f}")
    print()

    # SVHN dataset
    ood_data = datasets.SVHN('/home/ljl/Datasets/SVHN', split='test', transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    print(f'Data loaded with {len(ood_data)} out-of-distribuion test images.')
    auroc, fpr95, aupr = get_results(model, classifier, ood_loader, results_in)
    print("SVHN Detection")
    print(f"AUROC:{auroc:.2f}, FPR95:{fpr95:.2f}, AUPR:{aupr:.2f}")
    print()

    # LSUN-Crop dataset
    ood_data = datasets.ImageFolder('/home/ljl/Datasets/LSUN', transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    print(f'Data loaded with {len(ood_data)} out-of-distribuion test images.')
    auroc, fpr95, aupr = get_results(model, classifier, ood_loader, results_in)
    print("LSUN-Crop Detection")
    print(f"AUROC:{auroc:.2f}, FPR95:{fpr95:.2f}, AUPR:{aupr:.2f}")
    print()

    # LSUN-Resize dataset
    ood_data = datasets.ImageFolder('/home/ljl/Datasets/LSUN_resize', transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    print(f'Data loaded with {len(ood_data)} out-of-distribuion test images.')
    auroc, fpr95, aupr = get_results(model, classifier, ood_loader, results_in)
    print("LSUN-Resize Detection")
    print(f"AUROC:{auroc:.2f}, FPR95:{fpr95:.2f}, AUPR:{aupr:.2f}")
    print()

    # iSUN dataset
    ood_data = datasets.ImageFolder('/home/ljl/Datasets/iSUN', transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    print(f'Data loaded with {len(ood_data)} out-of-distribuion test images.')
    auroc, fpr95, aupr = get_results(model, classifier, ood_loader, results_in)
    print("iSUN Detection")
    print(f"AUROC:{auroc:.2f}, FPR95:{fpr95:.2f}, AUPR:{aupr:.2f}")
    print()


@torch.no_grad()
def get_scores(model, classifier, loader):
    classifier.eval()
    num_iter = len(loader)

    bar = Bar('Getting results:', max=num_iter)
    results = []

    for idx, (inp, _) in enumerate(loader):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        # forward
        with torch.no_grad():
            output = model.intermediate_forward(inp)
            output = output.reshape(output.shape[0], -1)
            output = classifier(output)

        output = torch.sigmoid(output).cpu().data.numpy()
        results.extend(output)

        bar.suffix = '({batch}/{size}) | Total:{total:} | ETA:{eta:}'.format(
            batch=idx + 1,
            size=num_iter,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            )
        bar.next()
    bar.finish()
    return results


def get_results(model, classifier, loader, results_in):
    results_out = get_scores(model, classifier, loader)

    results = []
    results.extend(results_in)
    results.extend(results_out)

    targets = np.hstack((np.zeros_like(results_in), np.ones_like(results_out)))

    auroc = calc_auroc(results, targets)
    fpr95 = calc_fpr(results, targets)
    aupr = average_precision_score(targets, results)

    return auroc * 100, fpr95 * 100, aupr * 100


def get_maha_score(mu, cov_inv, gmm_weights, x):
    '''
    Args:
        x: features of input with shape (num_samples, kernels, dimensions)
        mu: centers of gmm of all classes with shape (classes, kernels, dimensions)
        det_sigma: Determinant of covariance matrix with shape (classes, kernels)
        cov_inv: The inverse matrix of sigma which has shape (classes, kernels, dimensions, dimensions)
        gmm_weights: weights of gmm with shape (classes, kernels)
    '''
    cls, kers, dims = mu.shape
    num = x.shape[0]
    # expand x with the same shape with mu
    x = x.unsqueeze(1).expand(num, cls, kers, dims)
    mu = mu.expand_as(x)
    cov_inv = cov_inv.expand(num, *cov_inv.shape[:])
    # print(x.shape, mu.shape, cov_inv.shape)

    # reshape for calculation
    x = x.reshape(-1, 1, dims)
    mu = mu.reshape(-1, 1, dims)
    # cov_inv = cov_inv.reshape(-1, dims, dims)

    for i in range(cls):
    # calculate the maha distance: (x-μ)Σ^(-1)(x-μ)^T
        maha = torch.bmm((x - mu), cov_inv[i])
        maha =  0.5 * torch.bmm(maha, (x - mu).permute(0, 2, 1)).reshape(num, 1, kers)
        maha = (maha.cpu() * gmm_weights).sum(-1)
        if i == 1:
            mahas = maha
        else:
            mahas = torch.cat([mahas, maha], dim=1)
    min_maha = mahas.min(1)

    return min_maha

    
def get_gaussian(pretrained_weights):
    gau_path = (os.path.join(*pretrained_weights.split('/')[:-1]))
    means = torch.load(os.path.join(gau_path, 'means.pt'))
    covs_inv = torch.load(os.path.join(gau_path, 'covs_inv.pt'))
    return means, covs_inv


def get_dataset(mode, data_path, transform):
    if 'cifar10/' in data_path and mode == 'train':
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
    parser.add_argument('--classifier_weights', default='', type=str, help="Path to classifier pretrained weights to evaluate.")
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
    test(args)
    # calculate_var_inverse(args)
