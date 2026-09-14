import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from mlp import Policy, ValueFunction




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

        self.actor = Policy(num_observations, num_actions).eval()
        self.critic = ValueFunction(num_observations, num_actions).eval()

        self.num_observations = num_observations
        self.num_actions = num_actions

        self.env = env
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.discount = discount

        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr_actor)
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr_critic)

    def get_action(self, observation):

        dist = self.actor(observation)
        return dist.sample()

    def update_discount(self):
        self.cumulative_discount *= self.discount

    def restart_discount(self):
        self.cumulative_discount = 1

    def update_weights(self, observation, action, reward, next_observation):

        delta = reward + self.discount * self.critic(next_observation)
        delta = delta.detach()

        # actor update
        dist = self.actor(observation)
        log_prob = dist.log_prob(action).sum(dim=-1)*self.cumulative_discount

        actor_loss = -log_prob

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # critic update
        critic_loss = -self.critic(observation)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()



        