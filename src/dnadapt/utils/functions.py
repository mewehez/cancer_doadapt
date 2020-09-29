import numpy as np

import torch
from torch import nn
from torch.autograd import grad, Function

from dnadapt.globals import device


def l2_loss(x, eps=1e-8):
    loss = torch.sqrt(torch.mean(x ** 2) + eps)
    return loss


@torch.no_grad()
def correct_pred(z: torch.Tensor, labels: torch.Tensor)->int:
    _, pred = torch.max(z.data, 1)
    correct = (pred == labels).sum()
    return correct.item()


@torch.no_grad()
def accuracy(z: torch.Tensor, labels: torch.Tensor)->float:
    _, pred = torch.max(z.data, 1)
    acc = (pred == labels).float().mean()
    return acc.item()


def truncated_normal_(tensor, mean=0, std=1, lower_boud=-2, upper_bound=2):
    """Code from this discussion:
    https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/19
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < upper_bound) & (tmp > lower_boud)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


@torch.no_grad()
def init_bias(bias, cte=0.1):
    nn.init.constant_(bias, cte)


@torch.no_grad()
def init_weight(weight):
    input_size = weight.size(0)
    std = 1. / np.sqrt(input_size / 2.)
    truncated_normal_(weight, std=std)


def init_linear_weight(linear, cte=0.1):
    init_weight(linear.weight)
    init_bias(linear.bias, cte)


def init_weights(model, cte=0.1):
    for name, param in model.named_parameters():
        if 'weight' in name:
            init_weight(param)
        if 'bias' in name:
            init_bias(param, cte=cte)


class GradReverse(Function):
    lambd = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        return GradReverse.lambd * grad_outputs.neg()


def flip_gradient(x, lambd=1.0):
    GradReverse.lambd = lambd
    return GradReverse.apply(x)
