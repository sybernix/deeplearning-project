import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from OfficeDataset import OfficeDataset
from model.basenet import Predictor
from utils import get_classlist

batch_size = 4
resnet_output_vector_size = 512 # ??
temperature = 0.05
device = torch.device("mps")
learning_rate = 0.01
logs_file = '' # ??
checkpath = '' # ??
source_annotation_path = 'data/annotations/labeled_source_images_webcam.txt'
labled_target_annotation_path = 'data/annotations/labeled_target_images_amazon_1.txt'
unlabled_target_annotation_path = 'data/annotations/unlabeled_target_images_amazon_1.txt'
data_dir = 'data/office'

class_list = get_classlist(source_annotation_path)
num_class = len(class_list)

feature_extractor = resnet34(pretrained=True)
predictor = Predictor(num_class=num_class, input_vector_size=resnet_output_vector_size, norm_factor=temperature)
nn.init.xavier_normal_(predictor.fc.weight)
# nn.init.zeros_(F.fc.bias)

feature_extractor = nn.DataParallel(feature_extractor)
predictor = nn.DataParallel(predictor)
feature_extractor = feature_extractor.to(device)
predictor = predictor.to(device)

opt = {}
opt['logs_file'] = logs_file
opt['checkpath'] = checkpath
opt['class_list'] = class_list

# load required data
train_transform = transforms.Resize((256, 256))
source_dataset = OfficeDataset(source_annotation_path, data_dir, transform=train_transform)
source_data_loader = DataLoader(source_dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
labled_target_dataset = OfficeDataset(labled_target_annotation_path, data_dir, transform=train_transform)
labled_target_data_loader = DataLoader(labled_target_dataset, batch_size=batch_size, num_workers=0, shuffle=True,
                                       drop_last=True)
unlabled_target_dataset = OfficeDataset(unlabled_target_annotation_path, data_dir, transform=train_transform)
unlabled_target_data_loader = DataLoader(unlabled_target_dataset, batch_size=batch_size, num_workers=0, shuffle=True,
                                         drop_last=True)



print("hi")

def train() :
    feature_extractor.train()
    predictor.train()

    optimizer_feature_extractor = optim.SGD(feature_extractor.parameters(), lr=learning_rate)

