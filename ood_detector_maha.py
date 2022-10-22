import argparse
import os
from time import process_time_ns, time

import numpy as np
import torch
import torch.nn.functional as F
from progress.bar import Bar
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torchvision import datasets
from torchvision import models as torchvision_models
from torchvision import transforms

import utils
import vision_transformer as vits
from dataset.imagenet import Imagenet


def calculate_ind_acc(args):
    utils.init_distributed_mode(args)

    model = torchvision_models.__dict__[args.arch]()
    embed_dim = model.fc.weight.shape[1]
    model = utils.MultiCropWrapper(
        model,
        vits.DINOHead(embed_dim, args.out_dim, False),
    )

    _, gmm_weights = load_pretrained_weights(model, args.pretrained_weights, 'teacher')
    means, covs_inv = get_gaussian(args.pretrained_weights)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    in_data = get_dataset('val', args.in_data_path, args.num_labels, transform)
    sampler = torch.utils.data.distributed.DistributedSampler(in_data)
    in_loader = torch.utils.data.DataLoader(
        in_data,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f'Data loaded with {len(in_data)} in-distribuion test images.')
    results_in = get_scores(model, in_loader, means, covs_inv, gmm_weights, args.threshold)

    # Texture dataset
    ood_data = datasets.ImageFolder('/home/ljl/Datasets/dtd/images', transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    print(f'Data loaded with {len(ood_data)} out-of-distribuion test images.')
    results_out = get_scores(model, ood_loader, means, covs_inv, gmm_weights, args.threshold)
    auroc, fpr95, aupr = get_results(results_in, results_out)
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


def get_results(res_in, res_out):
    tar_in, tar_out = np.zeros(len(res_in)), np.ones(len(res_out))
    res, tar = [], []
    res.extend(res_in)
    res.extend(res_out)
    tar.extend(tar_in.tolist())
    tar.extend(tar_out.tolist())
    
    auroc = calc_auroc(res, tar)
    fpr95 = calc_fpr(res, tar)
    aupr = average_precision_score(tar, res)
    return auroc, fpr95, aupr
    

def get_scores(model, loader, means, covs_inv, gmm_weights, threshold):
    num_iter = len(loader)
    bar = Bar('Getting results:', max=num_iter)
    results = []

    for idx, (inp, _) in enumerate(loader):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        # forward
        with torch.no_grad():
            _, q = model(inp)
        maha = get_maha_score(means, covs_inv, gmm_weights, q.reshape(-1, 32, 256))
        output = (maha > threshold).int().cpu().tolist()
        print(maha)
        results.extend(output)

        bar.suffix = '({batch}/{size}) | Total:{total:} | ETA:{eta:}'.format(
            batch=idx + 1,
            size=num_iter,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            )
        bar.next()
    bar.finish()
    print(1 - sum(results) / len(results))
    return results


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

    for i in range(cls):
        # expand mean and cov+inv
        mu_ = mu[i:i+1].expand(num, kers, dims)
        cov_inv_ = cov_inv[i:i+1].expand(num, kers, dims, dims)

        # reshape for calculation
        x = x.reshape(-1, 1, dims).double()
        mu_ = mu_.reshape(-1, 1, dims).double()
        cov_inv_ = cov_inv_.reshape(-1, dims, dims)

        # calculate the maha distance: (x-μ)Σ^(-1)(x-μ)^T
        maha = torch.bmm((x - mu_), cov_inv_)
        maha =  0.5 * torch.bmm(maha, (x - mu_).permute(0, 2, 1)).reshape(num, 1, kers)
        maha = (maha.cpu() * gmm_weights[i]).sum(-1)
        if i == 0:
            mahas = maha
        else:
            mahas = torch.cat([mahas, maha], dim=1)
    min_maha, _ = mahas.min(1)
    # print(min_maha)

    return min_maha

    
def get_gaussian(pretrained_weights):
    gau_path = ('/' + os.path.join(*pretrained_weights.split('/')[:-1]))
    means = torch.load(os.path.join(gau_path, 'means.pt')).cuda()
    covs_inv = torch.load(os.path.join(gau_path, 'covs_inv.pt')).cuda()
    return means, covs_inv


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
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
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    return centers, gmm_weights


def get_dataset(mode, data_path, num_cls, transform):
    if 'ImageNet' in data_path:
        return Imagenet(mode, data_path, num_cls, transform)
    

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--in_data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--threshold', default=500, type=int, help='The threshold for discriminating OOD samples')
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    args = parser.parse_args()
    calculate_ind_acc(args)