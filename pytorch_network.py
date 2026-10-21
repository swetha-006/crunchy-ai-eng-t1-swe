"""The matching PyTorch model. PyTorch autograd is used only in the PyTorch version."""
import torch
from torch import nn


class TorchNetwork(nn.Module):
    def __init__(self, input_size=10, hidden1=16, hidden2=8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden1, hidden2, dtype=torch.float64)
        self.fc3 = nn.Linear(hidden2, 1, dtype=torch.float64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
