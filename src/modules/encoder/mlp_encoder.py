import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        return x
