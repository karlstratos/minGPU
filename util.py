import numpy as np
import os
import random
import sys
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = -1
        local_rank = -1
        world_size = -1
    return rank, local_rank, world_size


class Logger:

    def __init__(self, on=True):
        self.on = on

    def log(self, string, newline=True, force=False):
        if self.on or force:
            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()
