from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, n_layers: int, input_dim: int, hidden_dim: int, output_dim:int, dropout:float, activation):
        super(MLP, self).__init__()

        if n_layers == 1:
            modules = [
                nn.Dropout(p = dropout),
                nn.Linear(input_dim, output_dim)
            ]
        else:
            modules = [
                nn.Dropout(p = dropout),
                nn.Linear(input_dim, hidden_dim)
            ]

            for i in range(n_layers - 2):
                modules.extend([
                    activation,
                    nn.Dropout(p = dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                ])
            
            modules.extend([
                activation,
                nn.Dropout(p = dropout),
                nn.Linear(hidden_dim, output_dim)
            ])

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return torch.sigmoid(self.net(x))