import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet34

# params
from model.basenet import Predictor

batch_size = 4
resnet_output_vector_size = 512 # ??
num_class = 0 # ??
temperature = 0.05
device = torch.device("mps")
learning_rate = 0.01
logs_file = '' # ??
checkpath = '' # ??
class_list = '' # ??

G = resnet34(pretrained=True)
F = Predictor(num_class=num_class, input_vector_size=resnet_output_vector_size, norm_factor=temperature)
nn.init.xavier_normal_(F.weight)
nn.init.zeros_(F.bias)

G = nn.DataParallel(G)
F = nn.DataParallel(F)
G = G.to(device)
F = F.to(device)

opt = {}
opt['logs_file'] = logs_file
opt['checkpath'] = checkpath
opt['class_list'] = class_list

print("hi")