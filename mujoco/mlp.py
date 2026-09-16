import torch
import torch.nn as nn
from torch.distributions import Normal


class A2C(nn.Module):

    def __init__(self, n_input: int, n_output: int):

        super().__init__() 

        self.backbone = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh()
        )

        self.mu_head = nn.Sequential(
            nn.Linear(512, n_output),
            nn.Tanh()
        )

        self.log_std_head = nn.Linear(512, n_output)

        # self.log_std = nn.Parameter(torch.zeros(n_output)) # donat que estem en un cas on tot té igual rang i efecte, volem poca var, ho acceptem

        self.v_head = nn.Linear(512, 1)

    def forward(self, observation):

        shared = self.backbone(observation)
        
        mu = self.mu_head(shared)
        log_std = self.log_std_head(shared)
        log_std_clamped = torch.clamp(log_std, min=-7, max=0)
        std = torch.exp(log_std_clamped)

        v_s = self.v_head(shared)
        
        return Normal(mu, std), v_s







