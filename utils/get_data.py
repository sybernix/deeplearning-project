import random

import torch
import numpy as np
from torchvision.transforms import transforms
from OfficeDataset import OfficeDataset
from utils.randaugment import RandAugmentMC


def get_data(args):
    source_annotation_path = './data/annotations/labeled_source_images_webcam.txt'
    labled_target_annotation_path = './data/annotations/labeled_target_images_amazon_1.txt'
    unlabled_target_annotation_path = './data/annotations/unlabeled_target_images_amazon_1.txt'
    data_dir = './data/office'

    # define transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'strong': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

    }

    # load required data
    source_dataset = OfficeDataset(source_annotation_path, data_dir, transform=data_transforms.get('train'))
    # source_data_loader = DataLoader(source_dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    labled_target_dataset = OfficeDataset(labled_target_annotation_path, data_dir, transform=data_transforms.get('train'))
    # labled_target_data_loader = DataLoader(labled_target_dataset, batch_size=batch_size, num_workers=0, shuffle=True,
    #                                        drop_last=True)

    labeled_dataset = torch.utils.data.ConcatDataset([source_dataset, labled_target_dataset])

    unlabled_target_dataset = OfficeDataset(unlabled_target_annotation_path, data_dir,
                                            transform=data_transforms.get('train'),
                                            strong_transform=data_transforms.get('strong'))
    return labeled_dataset, unlabled_target_dataset

# class RandomAugmentation(object):
#     def __init__(self, numAugs, m):
#         assert numAugs >= 1
#         assert 1 <= m <= 10
#         self.numAugs = numAugs
#         self.m = m
#         self.augmentation_pool = augmentation_pool()
#
#     def __call__(self, img):
#         operations = random.choices(self.augmentation_pool, k=self.numAugs)
#         for operation, max_v, bias in operations:
#             v = np.random.randint(1, self.m)
#             if random.random() < 0.5:
#                 img = operation(img, v=v, max_v=v, bias=bias)
#         img = CutoutRegion(img, 16)
#         return img
#
# def augmentation_pool():
#     augmentations = [()]