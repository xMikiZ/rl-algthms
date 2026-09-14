import torch
import torch.nn as nn



class Policy(nn.Module):

    def __init__(self, n_input: int, n_output: int):

        super().__init__() 

        self.n_input = n_input
        self.n_output = n_output 

        self.mlp = nn.Sequential(
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
            nn.Linear(2028, 2*n_output),
        )

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, observation):

        fx = self.mlp(observation)
        fx = fx.view((self.n_output, 2))
        fx[:, 0] = self.softmax(fx[:, 0])
        fx[:, 1] = self.tanh(fx[:, 1])

        return fx


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








