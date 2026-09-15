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

        self.num_observations = num_observations
        self.num_actions = num_actions

        self.env = env
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.discount = discount

        self.actor_optimizer = torch.optim.SGD(self.a2c.parameters(), lr_actor)
        self.critic_optimizer = torch.optim.SGD(self.a2c.parameters(), lr_critic)

    def get_action(self, observation):

        dist, _ = self.a2c(observation)
        return dist.sample()

    def update_discount(self):
        self.cumulative_discount *= self.discount

    def restart_discount(self):
        self.cumulative_discount = 1

    def update_weights(self, observation, action, reward, next_observation):

        dist, v_s = self.a2c(observation)

        delta = reward + self.discount * self.a2c(next_observation)[1] - v_s
        delta = delta.detach()


        # actor update
        log_prob = dist.log_prob(action).sum(dim=-1)
        actor_loss = -self.cumulative_discount*delta*log_prob

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)

        # critic update
        critic_loss = -delta*v_s

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # recomanació del gemini: sinó s'acumulen masses gradients i torna nan
        torch.nn.utils.clip_grad_norm_(self.a2c.parameters(), max_norm=0.5)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

                    



        