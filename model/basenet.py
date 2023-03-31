import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, output_grad):
        input_grad = -output_grad
        return input_grad


def grad_reverse(x, lambd=1.0):
    return GradReverse().apply(x) * lambd


class Predictor(nn.Module):
    def __init__(self, num_class, input_vector_size, norm_factor):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(input_vector_size, num_class, bias=False)
        self.num_class = num_class
        self.norm_factor = norm_factor

    def forward(self, x, reversed=False, lambd=0.1):
        if reversed:
            x = grad_reverse(x, lambd)
        x = F.normalize(x)
        x = self.fc(x)
        x = x / self.norm_factor
        return x
