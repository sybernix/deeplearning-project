import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet34
from torch.utils.data import DataLoader

# params
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

G = resnet34(pretrained=True)
F = Predictor(num_class=num_class, input_vector_size=resnet_output_vector_size, norm_factor=temperature)
nn.init.xavier_normal_(F.fc.weight)
# nn.init.zeros_(F.fc.bias)

G = nn.DataParallel(G)
F = nn.DataParallel(F)
G = G.to(device)
F = F.to(device)

opt = {}
opt['logs_file'] = logs_file
opt['checkpath'] = checkpath
opt['class_list'] = class_list

# load required data

source_data_loader = DataLoader(OfficeDataset(source_annotation_path, data_dir))
labled_target_data_loader = DataLoader(OfficeDataset(labled_target_annotation_path, data_dir))
unlabled_target_data_loader = DataLoader(OfficeDataset(unlabled_target_annotation_path, data_dir))

def train() :
    G.train()
    F.train()



print("hi")