import torch
import torch.nn as nn
from typing import Self

class IgnitionNet(nn.Module):
    def __init__(self, input_dims: float, output_dims: float):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dims)
        )

    def forward(self, x):
        return self.model(x)
    
    @classmethod
    def load(cls, input_dims: float, output_dims: float, path: str) -> Self:
        model = cls(input_dims, output_dims)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
