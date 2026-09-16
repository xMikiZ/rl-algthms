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
            lr,
            discount,
            batch_size
            ):

        self.a2c = A2C(num_observations, num_actions)

        self.num_observations = num_observations
        self.num_actions = num_actions

        self.env = env
        self.lr = lr

        self.discount = discount

        self.optimizer = torch.optim.Adam(self.a2c.parameters(), lr)

        self.batch_size = batch_size
        self.actor_batch = [0]*batch_size
        self.critic_batch = [0]*batch_size
        self.batch_clock = 0

    def get_action(self, observation):

        dist, _ = self.a2c(observation)
        action = dist.sample()

        return action

    def update_discount(self):
        self.cumulative_discount *= self.discount

    def restart_discount(self):
        self.cumulative_discount = 1

    def update_batch_clock(self):
        self.batch_clock = (self.batch_clock + 1) % self.batch_size 

    def update_weights(self, observation, action, reward, next_observation, done):

        dist, v_s = self.a2c(observation)

        v_next = self.a2c(next_observation)[1].detach()
        target = reward + (1 - done)*self.discount * v_next

        delta = (target - v_s).detach()

        # actor loss
        log_prob = dist.log_prob(action).sum(dim=-1)
        actor_loss = -self.cumulative_discount*delta*log_prob
        self.actor_batch[self.batch_clock] = actor_loss

        # critic loss
        critic_loss = 0.5*(target - v_s)**2
        self.critic_batch[self.batch_clock] = critic_loss

        if self.batch_clock == self.batch_size - 1:

            batch_actor_loss = torch.stack(self.actor_batch).mean()
            batch_critic_loss = torch.stack(self.critic_batch).mean()

            self.optimizer.zero_grad()
            loss = batch_actor_loss + 0.5*batch_critic_loss 
            loss.backward()

            # recomanació del gemini: sinó s'acumulen masses gradients i torna nan
            torch.nn.utils.clip_grad_norm_(self.a2c.parameters(), max_norm=0.5)

            self.optimizer.step()

        self.update_batch_clock()
        
                    



        