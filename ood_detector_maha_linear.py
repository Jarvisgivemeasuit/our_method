
import argparse
from distutils.version import LooseVersion
import os
import random
from time import process_time_ns, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from progress.bar import Bar
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import Dataset
from torchvision import models as torchvision_models
from torchvision import transforms
from torchvision import datasets

import utils
import vision_transformer as vits


def calculate_ind_acc(args):
    utils.init_distributed_mode(args)

    model = torchvision_models.__dict__[args.arch]()
    embed_dim = model.fc.weight.shape[1]
    model = utils.MultiCropWrapper(
        model,
        vits.DINOHead(embed_dim, args.out_dim, False),
    )

    _, gmm_weights = load_pretrained_weights(model, args.pretrained_weights, 'teacher')
    means, covs, covs_inv = get_gaussian(args.pretrained_weights)
    gmm_weights = gmm_weights.cuda()
    model.cuda()
    model.eval()

    classifier = vits.vit_onelayer(embed_dim=args.num_labels, num_label=1)
    classifier.cuda()
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    acc = utils.Accuracy()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    in_data = get_dataset('train', args.in_data_path, transform)
    in_loader = torch.utils.data.DataLoader(
        in_data,
        # sampler=sampler,
        shuffle=True,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f'Data loaded with {len(in_data)} in-distribuion train images.')

    val_in_data = get_dataset('val', args.in_data_path, transform)
    val_in_loader = torch.utils.data.DataLoader(val_in_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        )

    print(f'Data loaded with {len(val_in_data)} in-distribuion ind test images.')

    # Texture dataset
    ood_data = datasets.ImageFolder('/home/ljl/Datasets/dtd/images', transform=transform)
    val_out_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )

    print(f'Data loaded with {len(ood_data)} in-distribuion ood test images.')
    for epoch in range(args.epochs):
        train_classifier(model, classifier, optimizer, 
                         in_loader, means, covs, covs_inv, gmm_weights, 
                         epoch, args.epochs, acc)
        scheduler.step()
        auroc, fpr95, aupr = validation(model, classifier, val_in_loader, val_out_loader,
                         means, covs_inv, gmm_weights)
        torch.save(classifier.state_dict(), os.path.join(args.output_dir, 'classifier.pt'))
        if epoch % 5 == 0:
            torch.save(classifier.state_dict(), os.path.join(args.output_dir, f'classifier{epoch:03}.pt'))

    

def train_classifier(model, classifier, optim, loader, 
                    means, covs_inv, gmm_weights, 
                    epoch, epochs, acc):
    # picov = utils.get_picov(covs)
    num_iter = len(loader)
    bar = Bar(f'Training classifier ({epoch:03d} / {epochs}):', max=num_iter)
    classifier.train()
    results, targets = [], []

    for idx, (inp, tar) in enumerate(loader):
        inp   = inp.cuda(non_blocking=True)
        cls   = means.shape[0]
        batch = inp.shape[0]

        # forward
        with torch.no_grad():
            _, q = model(inp)
            q = q.reshape(-1, 32, 256)

        maha   = utils.get_maha_score(means, covs_inv, gmm_weights, q)
        # target = (tar != cls_idx).float().cuda(non_blocking=True)
        # fvs    = utils.get_fvs(q, maha, means, covs_inv, gmm_weights, picov)

        # # Get the maha distance of the species to which all samples belong.
        # pos_maha = utils.get_cls_maha_score(means, covs_inv, gmm_weights, q, tar)
        # pos_tar  = torch.zeros_like(tar).float().cuda(non_blocking=True)

        # # Randomly select all categories to which the sample does not belong.
        # wait2del = [i * cls + tar[i] for i in range(len(tar))]
        # neg_tar  = np.array([range(cls)] * batch).reshape(batch, cls)
        # neg_tar  = np.delete(neg_tar, wait2del).reshape(batch, cls - 1)
        # neg_tar = torch.tensor([random.choice(tar_pool) for tar_pool in neg_tar]).cuda(non_blocking=True)

        # # Get the maha distance between the sample and the selected category, marked as a negative sample.
        # neg_maha = utils.get_cls_maha_score(means, covs_inv, gmm_weights, q, neg_tar)
        # neg_tar  = torch.ones_like(neg_tar).float().cuda(non_blocking=True)

        # maha   = torch.cat([pos_maha, neg_maha], dim=0)
        # target = torch.cat([pos_tar, neg_tar], dim=0)

        output = classifier(maha.float()).flatten()
        loss   = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        loss = loss.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        output, target = output.cpu(), target.cpu()
        output = (torch.sigmoid(output) >= 0.5).int()

        acc.update(output, target)

        output, target = output.tolist(), target.tolist()
        results.extend(output)
        targets.extend(target)

        auroc, fpr, aupr = get_results(results, targets)

        bar.suffix = '({batch}/{size}) | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f} | Acc:{acc:.4f} | AUROC:{auroc:.4f} | FPR95:{fpr:.4f} |AUPR:{aupr:.4f} | LR:{lr:.4f}'.format(
        # bar.suffix = '({batch:2d}/{size}) | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f} | Acc:{acc:.4f} | LR:{lr:.4f}'.format(
            batch=idx + 1,
            size=num_iter,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss,
            acc=acc.get_top1(),
            auroc=auroc,
            fpr=fpr,
            aupr=aupr,
            lr=optim.param_groups[0]['lr']
            )
        bar.next()
    bar.finish()
    # auroc, fpr, aupr = get_results(results, target)
    # print(f'AUROC:{auroc:.3f}, FPR95:{fpr:.3f}, AUPR:{aupr:.3f}')

    
def validation(model, classifier, 
               val_in_loader, val_out_loader, 
               means, covs_inv, gmm_weights):
    classifier.eval()
    results_in  = classifier_inference(model, classifier, val_in_loader, 
                                      means, covs_inv, gmm_weights)
    results_out = classifier_inference(model, classifier, val_out_loader, 
                                      means, covs_inv, gmm_weights)

    auroc, fpr95, aupr = get_test_results(results_in, results_out)
    print(f"AUROC:{auroc:.2f}, FPR95:{fpr95:.2f}, AUPR:{aupr:.2f}")
    return auroc, fpr95, aupr

    
def classifier_inference(model, classifier, loader, means, covs_inv, gmm_weights):
    num_iter = len(loader)

    bar = Bar('Getting results:', max=num_iter)
    results = []

    for idx, (inp, _) in enumerate(loader):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        # forward
        with torch.no_grad():
            _, q = model(inp)
        
            # Get the maha distance of the species to which all samples belong.
            maha = utils.get_maha_score(means, covs_inv, gmm_weights, q)
            _, min_idx = torch.min(maha.mean(-1), dim=-1)
            maha = maha[torch.tensor(range(maha.shape[0])), min_idx]
            output = classifier(maha.float())

            output = torch.sigmoid(output).cpu().numpy() >= 0.5
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


def get_gaussian(pretrained_weights):
    gau_path = ('/' + os.path.join(*pretrained_weights.split('/')[:-1]))
    means    = torch.load(os.path.join(gau_path, 'means.pt')).cuda()
    covs     = torch.load(os.path.join(gau_path, 'covs.pt')).cuda()
    covs_inv = torch.load(os.path.join(gau_path, 'covs_inv.pt')).cuda()
    return means, covs, covs_inv


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


def get_dataset(mode, data_path, transform):
    if 'ImageNet' in data_path:
        return Imagenet(mode, data_path, transform)


def get_results(res, tar):    
    auroc = calc_auroc(res, tar)
    fpr95 = calc_fpr(res, tar)
    aupr = average_precision_score(tar, res)
    return auroc, fpr95, aupr


def get_test_results(results_in, results_out):
    results = []
    results.extend(results_in)
    results.extend(results_out)

    targets = np.concatenate((np.zeros_like(results_in), np.ones_like(results_out)), axis=0)

    auroc = calc_auroc(results, targets)
    fpr95 = calc_fpr(results, targets)
    aupr = average_precision_score(targets, results)

    return auroc * 100, fpr95 * 100, aupr * 100


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

    
class Imagenet(Dataset):
    def __init__(self, mode, data_path, transform=None) -> None:
        super().__init__()
        assert mode in ['train', 'val']

        self.mode = mode
        self.transform = transform
        self.imagenet_path = os.path.join(data_path, mode)
        self.classes, self.img_list = {}, []

        with open(f'/home/ljl/Documents/our_method/dataset/ind_imagenet_100cls.txt', 'r') as f:
            for idx, line in enumerate(f):
                cls_name = line.strip()
                self.classes[cls_name] = idx

                cls_img_list = os.listdir(os.path.join(self.imagenet_path, cls_name))
                cls_img_list = [os.path.join(cls_name, k) for k in cls_img_list]
                self.img_list = self.img_list + cls_img_list

    def __getitem__(self, idx):
        img_name = self.img_list[idx]

        cls_name = img_name.split('/')[0]
        cls_label = self.classes[cls_name]

        img = Image.open(os.path.join(self.imagenet_path, img_name)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = self.train_transforms(img) if self.mode == 'train' else \
                self.val_transforms(img)

        return img, cls_label

    def __len__(self):
        return len(self.img_list)


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
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    args = parser.parse_args()
    calculate_ind_acc(args)