# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, MultiCropAugmentation
from semilearn.datasets.utils import split_ssl_data

# Algorithms that consume JEPAMatch-style local/global multi-crop views.
JEPA_ALGORITHMS = {'jepamatch'}

mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]


def get_cifar(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    # weak_strong_source (jepamatch only; default 'fixmatch' everywhere else):
    #   'fixmatch' -- standard FixMatch/FlexMatch recipe: weak = mild flip+crop,
    #                 strong = RandAugment(3,5). Well-validated for pseudo-labeling.
    #   'lejepa'   -- weak/strong instead built like the local-crop augmentation
    #                 (RandomResizedCrop + colorjitter/grayscale/solarize), with
    #                 weak using a larger crop scale (scale_weak) and strong a
    #                 smaller one (scale_strong), so they play the same "milder
    #                 vs. more aggressive" roles via crop scale instead of color.
    weak_strong_source = getattr(args, 'weak_strong_source', 'fixmatch') if alg in JEPA_ALGORITHMS else 'fixmatch'

    if weak_strong_source == 'lejepa':
        scale_weak = tuple(getattr(args, 'scale_weak', (0.8, 1.0)))
        scale_strong = tuple(getattr(args, 'scale_strong', (0.5, 0.8)))
        transform_weak = MultiCropAugmentation(num_crops=1, size=crop_size, scale=scale_weak, mean=mean[name], std=std[name]).view_transform
        transform_strong = MultiCropAugmentation(num_crops=1, size=crop_size, scale=scale_strong, mean=mean[name], std=std[name]).view_transform
    else:
        transform_weak = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[name], std[name])
        ])

        transform_strong = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean[name], std[name])
        ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])
    transform_local, transform_global = None, None
    if alg in JEPA_ALGORITHMS:
        scale_l = tuple(getattr(args, 'scale_l', (0.2, 0.5)))
        transform_local = MultiCropAugmentation(num_crops=args.num_local, size=args.local_size, scale=scale_l, mean=mean[name], std=std[name])

        # global_source (default 'reuse'): 'separate' builds an independent global-crop
        # batch (x_ulb_g) for the representation loss, instead of reusing weak/strong.
        if getattr(args, 'global_source', 'reuse') == 'separate':
            scale_g = tuple(getattr(args, 'scale_g', (0.5, 1.0)))
            transform_global = MultiCropAugmentation(num_crops=getattr(args, 'num_global', 2), size=crop_size, scale=scale_g, mean=mean[name], std=std[name])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])
    
    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))
    # lb_count = lb_count / lb_count.sum()
    # ulb_count = ulb_count / ulb_count.sum()
    # args.lb_class_dist = lb_count
    # args.ulb_class_dist = ulb_count

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets
        # if len(ulb_data) == len(data):
        #     lb_data = ulb_data 
        #     lb_targets = ulb_targets
        # else:
        #     lb_data = np.concatenate([lb_data, ulb_data], axis=0)
        #     lb_targets = np.concatenate([lb_targets, ulb_targets], axis=0)
    
    # output the distribution of labeled data for remixmatch
    # count = [0 for _ in range(num_classes)]
    # for c in lb_targets:
    #     count[c] += 1
    # dist = np.array(count, dtype=float)
    # dist = dist / dist.sum()
    # dist = dist.tolist()
    # out = {"distribution": dist}
    # output_file = r"./data_statistics/"
    # output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    # if not os.path.exists(output_file):
    #     os.makedirs(output_file, exist_ok=True)
    # with open(output_path, 'w') as w:
    #     json.dump(out, w)

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False, transform_local, transform_global)
    
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    test_data, test_targets = dset.data, dset.targets
    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, None, False)

    return lb_dset, ulb_dset, eval_dset
