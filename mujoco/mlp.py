import torch
import torch.nn as nn
from torch.distributions import Normal



class A2C:

    def __init__(self, n_input, n_output):
        
        self.actor = Actor(n_input, n_output)
        self.critic = Critic(n_input)


class Actor(nn.Module):
    
    def __init__(self, n_input: int, n_output: int):

        super().__init__() 

        self.mlp = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, n_output)
        )

        self.log_std = nn.Parameter(torch.zeros(n_output)) # donat que estem en un cas on tot té igual rang i efecte, volem poca var, ho acceptem

    def forward(self, observation):

        mu = self.mlp(observation)
        mu_clamped = torch.clamp(mu, min=-1, max = 1)
        log_std_clamped = torch.clamp(self.log_std, min=-7, max=1)
        std = torch.exp(log_std_clamped)

        return Normal(mu, std)


class Critic(nn.Module):
    
    def __init__(self, n_input: int):

        super().__init__() 

        self.mlp = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, observation):
        v_s = self.mlp(observation)
        return v_s