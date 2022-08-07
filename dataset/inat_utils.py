import numpy as np
import os
import random
from time import time
from shutil import copytree
from progress.bar import Bar


def run_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        func(*args, **kwargs)                  
        cost_time = time() - start
        print("func run time: {:.3f}s.".format(cost_time))
    return wrapper


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


@run_time
def get_cls_file(num_classes, k=1):
    cls_list = []
    with open('/home/ljl/Datasets/inat/class_list.txt', 'r') as f:
        for line in f:
            cls_list.append(line.strip())
    cls_list = np.array(cls_list)

    cls_index = range(310)
    ind_cls_index = random.sample(cls_index, k=num_classes)
    ood_cls_index = np.delete(cls_index, ind_cls_index)
    ind_cls_list, ood_cls_list = cls_list[ind_cls_index].tolist(), cls_list[ood_cls_index].tolist()

    with open(f'ind_inat_{k}.txt', 'w') as f:
        for cls in ind_cls_list:
            f.write(cls+'\n')

    with open(f'ood_inat_rest.txt', 'w') as f:
        for cls in ood_cls_list:
            f.write(cls+'\n')


@run_time
def make_ood_dataset(num_classes, k=1):
    cls_list = []

    with open(f'ood_cls_inat_rest.txt', 'r') as f:
        for line in f:
            cls_list.append(line.strip())
    cls_list = np.array(cls_list)

    cls_index = range(210)
    ood_train_index = random.sample(cls_index, k=num_classes)
    ood_valid_index = np.delete(cls_index, ood_train_index)
    ood_train_list, ood_valid_list = cls_list[ood_train_index].tolist(), cls_list[ood_valid_index].tolist()

    with open(f'ood_train_inat_{k}.txt', 'w') as f:
        for cls in ood_train_list:
            f.write(cls+'\n')
        f.close()

    with open(f'ood_valid_inat_{k}.txt', 'w') as f:
        for cls in ood_valid_list:
            f.write(cls+'\n')
        f.close


if __name__ == '__main__':
    k = 2
    get_cls_file(100, k=k)
    make_ood_dataset(100, k=k)