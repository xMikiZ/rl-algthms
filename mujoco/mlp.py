import torch
import torch.nn as nn
from torch.distributions import Normal


class A2C(nn.Module):

    def __init__(self, n_input: int, n_output: int):

        super().__init__() 

        self.n_input = n_input
        self.n_output = n_output 

        self.backbone = nn.Sequential(
            nn.Linear(n_input, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh()
        )

        self.mu_head = nn.Sequential(
            nn.Linear(1024, n_output),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(n_output)) # donat que estem en un cas on tot té igual rang i efecte, volem poca var, ho acceptem

        self.v_head = nn.Linear(1024, 1)

    def forward(self, observation):

        shared = self.backbone(observation)

        mu = self.mu_head(shared)
        log_std_clamped = torch.clamp(self.log_std, min=-4.6, max=-1.4)
        std = torch.exp(log_std_clamped)

        v_value = self.v_head(shared)
        
        return Normal(mu, std), v_value







