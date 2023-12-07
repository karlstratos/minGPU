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
        return self.examples[index], self.labels[index]


class MyModel(nn.Module):

    def __init__(self, dim, num_labels):
        super().__init__()
        self.scorer = nn.Linear(dim, num_labels)

    def forward(self, examples):  # (num_examples, dim)
        scores = self.scorer(examples)  # (num_examples, num_labels)
        _, predictions = scores.max(dim=1)  # (num_examples,)
        return scores, predictions

    def weight(self):
        return self.scorer.weight

    def grad(self):
        return self.scorer.weight.grad
