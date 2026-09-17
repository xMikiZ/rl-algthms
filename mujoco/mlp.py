import torch
import torch.nn as nn
from torch.distributions import Normal


class Policy(nn.Module):

    def __init__(self, n_input: int, n_output: int):

        super().__init__() 

        self.mu_linear = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, n_output)
        )

        self.log_std = nn.Parameter(torch.zeros(n_output)) # donat que estem en un cas on tot té igual rang i efecte, volem poca var, ho acceptem

    def forward(self, observation):
        
        mu = self.mu_linear(observation)
        log_std_clamped = torch.clamp(self.log_std, min=-7, max=0)
        std = torch.exp(log_std_clamped)
     
        return Normal(mu, std)


class ValueFunction(nn.Module):

    def __init__(self, n_input: int):

        super().__init__() 

        self.v_linear = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, observation):

        v_s = self.v_linear(observation)
        
        return v_s















