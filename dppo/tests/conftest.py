import random

import torch


def pytest_runtest_setup(item):
    random.seed(1)
    torch.manual_seed(1)
