import torch
import torch.nn as nn
from torch.distributions import Normal



class PPO:

    def __init__(self, n_input, n_output):
        
        self.actor = Actor(n_input, n_output)
        self.new_actor = Actor(n_input, n_output)
        self.critic = Critic(n_input)


class Actor(nn.Module):
    
    def __init__(self, n_input: int, n_output: int):

        super().__init__() 

        self.mlp = nn.Sequential(
            nn.Linear(n_input, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, n_output),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(n_output)) # donat que estem en un cas on tot té igual rang i efecte, volem poca var, ho acceptem

    def forward(self, observation):

        mu = 3*self.mlp(observation)
        std = torch.exp(self.log_std)

        return Normal(mu, std)


class Critic(nn.Module):
    
    def __init__(self, n_input: int):

        super().__init__() 

        self.mlp = nn.Sequential(
            nn.Linear(n_input, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, observation):
        v_s = self.mlp(observation)
        return v_s
