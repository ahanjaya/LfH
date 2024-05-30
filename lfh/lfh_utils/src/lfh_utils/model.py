import torch
import torch.nn as nn
import torch.nn.functional as F

class LfHModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LfHModel, self).__init__()

        self.linears = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.linears(x)