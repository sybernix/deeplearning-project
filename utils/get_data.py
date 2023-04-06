import torch
from torchvision.transforms import transforms
from OfficeDataset import OfficeDataset
from utils.randaugment import RandAugmentMC


def get_data(args):
    source_annotation_path = './data/annotations/labeled_source_images_webcam.txt'
    labled_target_annotation_path = './data/annotations/labeled_target_images_amazon_3.txt'
    unlabled_target_annotation_path = './data/annotations/unlabeled_target_images_amazon_3.txt'
    val_target_annotation_path = './data/annotations/validation_target_images_amazon_3.txt'
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
    labled_target_dataset = OfficeDataset(labled_target_annotation_path, data_dir, transform=data_transforms.get('train'))
    target_dataset_val = OfficeDataset(val_target_annotation_path, data_dir, transform=data_transforms.get('val'),
                                       strong_transform=data_transforms.get('strong'))

    # labeled_dataset = torch.utils.data.ConcatDataset([source_dataset, labled_target_dataset])

    unlabled_target_dataset = OfficeDataset(unlabled_target_annotation_path, data_dir,
                                            transform=data_transforms.get('train'),
                                            strong_transform=data_transforms.get('strong'))
    return source_dataset, labled_target_dataset, unlabled_target_dataset, target_dataset_val
