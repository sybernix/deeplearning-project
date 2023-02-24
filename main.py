import numpy as np
import torch
from torchvision.models import resnet34

batch_size = 4
G = resnet34(pretrained=True)

resnet_output_vector_size = 512 #?

print("hi")