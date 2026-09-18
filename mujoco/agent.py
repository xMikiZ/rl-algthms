import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from mlp import A2C




class AntAgent():

    def __init__(
            self,
            num_observations,
            num_actions,
            env: gym.Env,
            lr_actor,
            lr_critic,
            discount
            ):

        self.a2c = A2C(num_observations, num_actions)

        self.env = env

        self.discount = discount

        self.actor_optimizer = torch.optim.Adam(self.a2c.actor.parameters(), lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.a2c.critic.parameters(), lr_critic)

    def get_action(self, observation):

        dist = self.a2c.actor(observation)
        action = dist.sample()

        return torch.clamp(action, -1, 1)

    def get_losses(self, observation, action, reward, next_observation, done):
        """Store experience as the loss"""

        with torch.no_grad():
            v_next = self.a2c.critic(next_observation).squeeze(-1)
            target = reward + (1 - done) * self.discount * v_next

        v_s = self.a2c.critic(observation).squeeze(-1)
        critic_loss = 0.5 * (target.detach() - v_s).pow(2).mean() # aquest .detach() és redundant?

        advantage = (target - v_s).detach()
        log_prob = self.a2c.actor(observation).log_prob(action).sum(dim=-1)
        actor_loss = -(advantage*log_prob).mean()

        return actor_loss, critic_loss

    def update_weights(self, actor_loss, critic_loss):

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        