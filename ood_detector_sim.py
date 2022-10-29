import os
import argparse
import numpy as np
from time import process_time_ns, time
from progress.bar import Bar

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import models as torchvision_models
from torchvision import transforms

import utils
import vision_transformer as vits
from dataset.imagenet import Imagenet
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score


def calculate_ind_acc(args):
    utils.init_distributed_mode(args)

    model = torchvision_models.__dict__[args.arch]()
    embed_dim = model.fc.weight.shape[1]
    model = utils.MultiCropWrapper(
        model,
        vits.DINOHead(embed_dim, args.out_dim, False),
    )

    centers, gmm_weights = load_pretrained_weights(model, args.pretrained_weights, 'teacher')
    # means, covs_inv = get_gaussian(args.pretrained_weights)
    centers, gmm_weights = centers.cuda(), gmm_weights.cuda()
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
    results_in = get_scores(model, in_loader, centers, gmm_weights)

    aurocs, fpr95s, auprs = [], [], []

    # Texture dataset
    ood_data = datasets.ImageFolder('/home/ljl/Datasets/dtd/images', transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    print(f'Data loaded with {len(ood_data)} out-of-distribuion test images.')
    results_out = get_scores(model, ood_loader, centers, gmm_weights)
    auroc, fpr95, aupr = get_results(results_in, results_out)
    aurocs.append(auroc)
    fpr95s.append(fpr95)
    auprs.append(aupr)
    print("Texture Detection")
    print(f"AUROC:{auroc * 100:.2f}, FPR95:{fpr95 * 100:.2f}, AUPR:{aupr * 100:.2f}")
    print()

    # iNaturalist dataset
    ood_data = datasets.ImageFolder('/home/ljl/Datasets/iNaturalist', transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    print(f'Data loaded with {len(ood_data)} out-of-distribuion test images.')
    results_out = get_scores(model, ood_loader, centers, gmm_weights)
    auroc, fpr95, aupr = get_results(results_in, results_out)
    aurocs.append(auroc)
    fpr95s.append(fpr95)
    auprs.append(aupr)
    print("iNaturalist Detection")
    print(f"AUROC:{auroc * 100:.2f}, FPR95:{fpr95 * 100:.2f}, AUPR:{aupr * 100:.2f}")
    print()

    # Places dataset
    ood_data = datasets.ImageFolder('/home/ljl/Datasets/Places', transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    print(f'Data loaded with {len(ood_data)} out-of-distribuion test images.')
    results_out = get_scores(model, ood_loader, centers, gmm_weights)
    auroc, fpr95, aupr = get_results(results_in, results_out)
    aurocs.append(auroc)
    fpr95s.append(fpr95)
    auprs.append(aupr)
    print("Places Detection")
    print(f"AUROC:{auroc * 100:.2f}, FPR95:{fpr95 * 100:.2f}, AUPR:{aupr * 100:.2f}")
    print()

    # SUN dataset
    ood_data = datasets.ImageFolder('/home/ljl/Datasets/SUN', transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    print(f'Data loaded with {len(ood_data)} out-of-distribuion test images.')
    results_out = get_scores(model, ood_loader, centers, gmm_weights)
    auroc, fpr95, aupr = get_results(results_in, results_out)
    aurocs.append(auroc)
    fpr95s.append(fpr95)
    auprs.append(aupr)
    print("SUN Detection")
    print(f"AUROC:{auroc * 100:.2f}, FPR95:{fpr95 * 100:.2f}, AUPR:{aupr * 100:.2f}")
    print()

    # # ImageNet-O dataset
    # ood_data = datasets.ImageFolder('/home/ljl/Datasets/imagenet-o', transform=transform)
    # ood_loader = torch.utils.data.DataLoader(ood_data, 
    #                     batch_size=args.batch_size_per_gpu, 
    #                     shuffle=True,
    #                     num_workers=args.num_workers,
    #                     pin_memory=True
    #                     )
    # print(f'Data loaded with {len(ood_data)} out-of-distribuion test images.')
    # results_out = get_scores(model, ood_loader, centers, gmm_weights)
    # auroc, fpr95, aupr = get_results(results_in, results_out)
    # aurocs.append(auroc)
    # fpr95s.append(fpr95)
    # auprs.append(aupr)
    # print("ImageNet-O Detection")
    # print(f"AUROC:{auroc:.2f}, FPR95:{fpr95:.2f}, AUPR:{aupr:.2f}")
    # print()
    print('AVERAGE')
    print(f"AUROC:{np.mean(aurocs) * 100:.2f}, FPR95:{np.mean(fpr95s) * 100:.2f}, AUPR:{np.mean(auprs) * 100:.2f}")


def get_results(res_in, res_out):
    tar_in, tar_out = np.ones(len(res_in)), np.zeros(len(res_out))
    res, tar = [], []
    res.extend(res_in)
    res.extend(res_out)
    tar.extend(tar_in.tolist())
    tar.extend(tar_out.tolist())
    
    auroc = calc_auroc(res, tar)
    fpr95 = calc_fpr(res, tar)
    aupr = average_precision_score(tar, res)
    return auroc, fpr95, aupr
    

def get_scores(model, loader, means, gmm_weights):
    num_iter = len(loader)
    bar = Bar('Getting results:', max=num_iter)
    results = []

    for idx, (inp, _) in enumerate(loader):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        # forward
        with torch.no_grad():
            _, q = model(inp)
        # scores = get_similarity_score(means, gmm_weights, q.reshape(-1, 32, 256))
        scores = get_distance_score(means, gmm_weights, q.reshape(-1, 32, 256))
        output = scores.cpu().tolist()
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


def get_distance_score(mu, gmm_weights, x):
    '''
    Args:
        x: features of input with shape (num_samples, kernels, dimensions)
        mu: centers of gmm of all classes with shape (classes, kernels, dimensions)
        det_sigma: Determinant of covariance matrix with shape (classes, kernels)
        gmm_weights: weights of gmm with shape (classes, kernels)
    '''
    cls, kers, dims = mu.shape
    num = x.shape[0]

    for i in range(cls):
        # expand mean
        mu_ = mu[i:i+1].expand(num, kers, dims)

        # reshape for calculation
        x = x.reshape(-1, dims)
        mu_ = mu_.reshape(-1, dims)

        # calculate the euclidean distance
        dist = F.pairwise_distance(x, mu_, p=2).reshape(num, kers)
        dist = (dist * gmm_weights[i]).sum(-1)
        if i == 0:
            scores = dist.unsqueeze(1)
        else:
            scores = torch.cat([scores, dist.unsqueeze(1)], dim=1)
    min_score, _ = scores.min(1)
    # print(min_maha)

    return min_score


def get_similarity_score(mu, gmm_weights, x):
    '''
    Args:
        x: features of input with shape (num_samples, kernels, dimensions)
        mu: centers of gmm of all classes with shape (classes, kernels, dimensions)
        det_sigma: Determinant of covariance matrix with shape (classes, kernels)
        gmm_weights: weights of gmm with shape (classes, kernels)
    '''
    cls, kers, dims = mu.shape
    num = x.shape[0]

    for i in range(cls):
        # expand mean
        mu_ = mu[i:i+1].expand(num, kers, dims)

        # reshape for calculation
        x = x.reshape(-1, 1, dims).double()
        mu_ = mu_.reshape(-1, 1, dims).double()

        # calculate the maha distance: (x-μ)Σ^(-1)(x-μ)^T
        cos_sim = torch.cosine_similarity(x, mu_, dim=-1).reshape(num, kers)
        cos_sim = (cos_sim * gmm_weights[i]).sum(-1)
        if i == 0:
            scores = cos_sim.unsqueeze(1)
        else:
            scores = torch.cat([scores, cos_sim.unsqueeze(1)], dim=1)
    max_score, _ = scores.max(1)
    # print(min_maha)

    return max_score

    
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
    parser.add_argument('--threshold', default=500, type=float, help='The threshold for discriminating OOD samples')
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    args = parser.parse_args()
    calculate_ind_acc(args)