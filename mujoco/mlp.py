import torch
import torch.nn as nn
from torch.distributions import Normal



class Policy(nn.Module):

    def __init__(self, n_input: int, n_output: int):

        super().__init__() 

        self.n_input = n_input
        self.n_output = n_output 

        self.mu_mlp = nn.Sequential(
            nn.Linear(n_input, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2028),
            nn.ReLU(),
            nn.Linear(2028, n_output),
            nn.Tanh()
        ) # Tanh pq action space entre -1 i 1

        # recomanació gemini
        self.log_std = nn.Parameter(torch.zeros(n_output)) # donat que estem en un cas on tot té igual rang i efecte, volem poca var, ho acceptem

    def forward(self, state: torch.Tensor) -> Normal:
        mu = self.mu_mlp(state)
        
        log_std_clamped = torch.clamp(self.log_std, min=-20, max=2)
        std = torch.exp(log_std_clamped)
        
        return Normal(mu, std)


class ValueFunction(nn.Module):

    def __init__(self, n_input, n_output):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_input, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
             nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, observation):

        fx = self.mlp(observation)
        return fx








