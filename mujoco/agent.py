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

    def get_action(self, observation):
        """Returns action: arg and value"""
        all_actions = self.actor(observation)

        #TODO: replace greedy best_action to sampled_action 
        best_action = torch.argmax(all_actions[:, 0])
        res = np.zeros(self.num_actions)
        res[best_action] = all_actions[best_action, 1] 

        return res

    def update_discount(self):
        self.cumulative_discount *= self.discount

    def restart_discount(self):
        self.cumulative_discount = 1

    def update_weights(self, observation, reward, action, next_observation):

        with torch.no_grad():
            delta = reward[np.argmax(action)] + self.discount * self.critic(next_observation)

        # critic weights
        grad_critic = torch.autograd.grad(self.critic(observation), self.critic.parameters())

        # w <- w + alpha * delta * grad
        with torch.no_grad():
            for param, grad in zip(self.critic.parameters(), grad_critic):
                param += self.lr_critic * delta * grad

                # critic weights
        eps = 1e-5
        log_policy = torch.log(self.actor(observation)[np.argmax(action), 0] + eps)
        grad_actor = torch.autograd.grad(log_policy, self.actor.parameters())

        # 0 <- 0 + alpha * I * delta * grad
        with torch.no_grad():
            for param, grad in zip(self.actor.parameters(), grad_actor):
                param += self.lr_actor * self.cumulative_discount * delta * grad