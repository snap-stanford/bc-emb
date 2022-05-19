import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from typing import Optional

class MLP(nn.Module):
    def __init__(self, 
                    input_emb_dim: int,
                    hidden_dim: int,
                    dropout: float = 0,
                    num_mlp_layers: int = 2,
                    out_dim: int = 1,
                    # used for clamping output values (useful for regression task)
                    min_value: Optional[float] = None, 
                    max_value: Optional[float] = None,
                ):
        super(MLP, self).__init__()
        
        module_list = [
            nn.Linear(input_emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        for i in range(num_mlp_layers - 1):
            module_list += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ]

        module_list += [nn.Linear(hidden_dim, out_dim)]

        self.min_value = min_value
        self.max_value = max_value

        self.mlp = nn.Sequential(*module_list)


    def forward(self, x):
        out = self.mlp(x)
        if self.training:
            return out
        else:
            if (self.min_value is not None) and (self.max_value is not None):
                return torch.clamp(out, min = self.min_value, max = self.max_value)
            else:
                return out