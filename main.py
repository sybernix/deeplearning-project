import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34
from torch.utils.data import DataLoader
from model.basenet import Predictor
from utils import get_classlist

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--temperature', type=float, default=0.05)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--train_steps', type=int, default=10)
logs_file = '' # ??
checkpath = '' # ??


args = parser.parse_args()

device = torch.device("mps")
resnet_output_vector_size = 1000 # ??

class_list = get_classlist(source_annotation_path)
num_class = len(class_list)

feature_extractor = resnet34(pretrained=True)
predictor = Predictor(num_class=num_class, input_vector_size=resnet_output_vector_size, norm_factor=args.temperature)
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



labeled_dataloader = DataLoader(labeled_datasets, batch_size=args.batch_size, num_workers=0, shuffle=True,
                                drop_last=True)
unlabled_target_data_loader = DataLoader(unlabled_target_dataset, batch_size=args.batch_size, num_workers=0,
                                         shuffle=True, drop_last=True)

labeled_data_iter = iter(labeled_dataloader)
unlabled_data_iter = iter(unlabled_target_data_loader)

print("hi")


def train() :
    feature_extractor.train()
    predictor.train()

    optimizer_feature_extractor = optim.SGD(feature_extractor.parameters(), lr=args.learning_rate, momentum=0.9,
                                            weight_decay=0.0005, nesterov=True)
    optimizer_predictor = optim.SGD(predictor.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005,
                                    nesterov=True)

    criterion = nn.CrossEntropyLoss().to(device)

    for step in range(args.train_steps):
        labeled_data_iter_next = next(labeled_data_iter)
        labeled_data_images = labeled_data_iter_next[0].type(torch.FloatTensor).to(device)
        labeled_data_labels = labeled_data_iter_next[1].to(device)
        features = feature_extractor(labeled_data_images)
        predictions = predictor(features)

        cross_entropy_loss = criterion(predictions, labeled_data_labels)

        cross_entropy_loss.backward(retain_graph=True)
        optimizer_feature_extractor.step()
        optimizer_predictor.step()
        optimizer_feature_extractor.zero_grad()
        optimizer_predictor.zero_grad()

        # calculate loss for unlabeled target data

        print("hi")


train()
