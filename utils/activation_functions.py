import torch.nn as nn


class swish(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()
