import torch
from torch import nn


class DQN(nn.Module):

    def __init__(self, n_input: int, n_output: int) -> None:

        super().__init__()
        #TODO: afegir hl = núm de capes
        self.f = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_output)
        )

    def forward(self, x):
        return self.f(x)
