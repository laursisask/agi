import math

import torch


def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)


def normal_log_probs(mean, std, value):
    assert mean.shape == std.shape == value.shape

    return -torch.log(std) - math.log(math.sqrt(2 * math.pi)) - (value - mean) ** 2 / (2 * std ** 2)


def inverse_log_prob(x):
    return torch.log(1 - torch.exp(x))


def categorical_entropy(log_probs):
    return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)


def bernoulli_entropy(log_probs):
    inverse = inverse_log_prob(log_probs)
    return -(torch.exp(log_probs) * log_probs + torch.exp(inverse) * inverse)


def normal_entropy(std):
    return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)
