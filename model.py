import torch


class MyModel(torch.nn.Module):

    def __init__(self, dim, num_labels):
        super().__init__()
        self.scorer = torch.nn.Linear(dim, num_labels)

    def forward(self, examples):  # (N, d)
        scores = self.scorer(examples)  # (N, L)
        _, predictions = scores.max(dim=1)  # (N,)
        return scores, predictions

    def weight(self):
        return self.scorer.weight

    def grad(self):
        return self.scorer.weight.grad
