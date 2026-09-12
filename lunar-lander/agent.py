import torch
from torch import nn

import numpy as np

from dqn import DQN

obs_size = 8
col_size = 19
    
class LunarLanderAgent:

    clock = 0

    def __init__(
            self,
            env, 
            n_obs, 
            n_actions, 
            buffer_size, 
            mini_batch_size, 
            smoothing_factor,
            eps_initial,
            eps_decay,
            eps_final,
            discount = 0.95,
            learning_rate = 0.001
            ) -> None:

        self.env = env

        self.critic = DQN(n_obs, n_actions)
        self.target = DQN(n_obs, n_actions)

        # Recomanacions Gemini
        self.target.load_state_dict(self.critic.state_dict())
        self.target.eval()

        self.buffer_size = buffer_size
        self.buffer = torch.zeros((buffer_size, col_size))

        self.mini_batch_size = mini_batch_size

        self.smoothing_factor = smoothing_factor

        self.epsilon = eps_initial
        self.eps_decay = eps_decay
        self.eps_final = eps_final

        self.discount = discount

        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr = learning_rate)
        self.loss = nn.MSELoss(reduction='mean')

    def get_action(self, observation) -> None:

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
            
        else:
            with torch.no_grad():
                return int(torch.argmax(self.critic(observation)))

    def update_buffer(self, experience) -> None:

        self.buffer[self.clock % self.buffer_size] = experience
        self.clock += 1
        
    def update_policy(self) -> None:

        if self.clock < self.buffer_size:
            return
        
        indexes = np.random.choice(self.buffer_size, self.mini_batch_size, replace=False)
        mini_batch = self.buffer[indexes]

        # El Gemini ha ordenat una mica també aquí :)

        states      = mini_batch[:, 0:8]
        actions     = mini_batch[:, 8].long()
        rewards     = mini_batch[:, 9]
        next_states = mini_batch[:, 10:18]
        terminateds = mini_batch[:, 18]

        with torch.no_grad():
            max_next_q = self.target(next_states).max(dim=1).values
            targets = rewards + (1 - terminateds) * self.discount * max_next_q

        # Q-values for actions taken
        q_values = self.critic(states)
        predictions = q_values[torch.arange(self.mini_batch_size), actions]

        loss = self.loss(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # el Gemini ha ajudat aquí també :)
        for target_param, critic_param in zip(self.target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.smoothing_factor * critic_param.data + (1.0 - self.smoothing_factor) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_final, self.epsilon*self.eps_decay)

