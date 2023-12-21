import numpy as np
import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset


# Environment variables set by torchrun:
# https://pytorch.org/docs/stable/elastic/run.html#environment-variables
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


def get_device(logger=None):
    _, local_rank, _ = check_distributed()
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group('nccl')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if logger is not None:
        logger(f'Using device: {str(device)}', ['underline'], force=True)

    return device


class MyDataset(Dataset):

    def __init__(self, num_examples, dim, num_labels):
        self.examples = torch.randn(num_examples, dim)
        self.labels = torch.randint(num_labels, (num_examples,))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index], self.labels[index], index


class LinearNet(nn.Module):

    def __init__(self, dim, num_labels):
        super().__init__()
        self.scorer = nn.Linear(dim, num_labels, bias=False)

    def forward(self, examples):  # (num_examples, dim)
        scores = self.scorer(examples)  # (num_examples, num_labels)
        _, predictions = scores.max(dim=1)  # (num_examples,)
        return scores, predictions

    def weight(self):
        return self.scorer.weight

    def grad(self):
        return self.scorer.weight.grad


class ThreeLayerNet(nn.Module):  # A toy 3-layer net to demonstrate FSDP

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(2, 2), nn.ReLU())  # 6 params
        self.layer2 = nn.Sequential(nn.Linear(2, 2), nn.ReLU())  # 6 params
        self.layer3 = nn.Linear(2, 3)  # 9 params

    def forward(self, examples):  # (num_examples, 2)
        output1 = self.layer1(examples)
        output2 = self.layer2(output1)
        scores = self.layer3(output2)  # (num_examples, 3)
        _, predictions = scores.max(dim=1)
        return scores, predictions

    def init(self, delta=0.001):  # Make weights easily readable
        val = 0

        def init_linear(linear, val):
            with torch.no_grad():
                for i in range(linear.weight.shape[0]):
                    for j in range(linear.weight.shape[1]):
                        val += delta
                        linear.weight[i, j] = torch.tensor([val])
                for i in range(len(linear.bias)):
                    val += delta
                    linear.bias[i] = torch.tensor([val])
            return val

        val = init_linear(self.layer1[0], val)
        val = init_linear(self.layer2[0], val)
        val = init_linear(self.layer3, val)

    def weight_str(self):
        s = f'layer1[0] weight/bias: '
        s += param_str(self.layer1[0].weight)
        s += ', ' + param_str(self.layer1[0].bias)

        s += f'\nlayer2[0] weight/bias: '
        s += param_str(self.layer2[0].weight)
        s += ', ' + param_str(self.layer2[0].bias)

        s += f'\nlayer3 weight/bias: '
        s += param_str(self.layer3.weight)
        s += ', ' + param_str(self.layer3.bias)
        return s


def param_str(param):
    np.set_printoptions(formatter={'float': lambda x: '{0:0.4f}'.format(x)})
    with torch.no_grad():
        return str(param.cpu().numpy()).replace('\n', '')
