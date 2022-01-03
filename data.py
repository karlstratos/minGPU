import torch


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, num_examples, dim, num_labels):
        self.examples = torch.randn(num_examples, dim)
        self.labels = torch.randint(num_labels, (num_examples,))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index], self.labels[index]
