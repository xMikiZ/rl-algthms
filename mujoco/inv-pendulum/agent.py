import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from mlp import PPO




class InvPendulumAgent():

    def __init__(
            self,
            num_observations,
            num_actions,
            env: gym.Env,
            lr_actor,
            lr_critic,
            eps,
            discount,
            beta,
            beta_decay
            ):

        self.ppo = PPO(num_observations, num_actions)

        self.env = env

        self.discount = discount
        self.beta = beta
        self.beta_decay = beta_decay

        self.eps = eps

        self.actor_optimizer = torch.optim.Adam(self.ppo.new_actor.parameters(), lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.ppo.critic.parameters(), lr_critic)

    def get_action(self, observation):

        dist = self.ppo.actor(observation)
        action = dist.sample()

        return action

    def get_losses(self, observation, action, reward, next_observation, done):
        """Store experience as the loss"""

        with torch.no_grad():
            v_next = self.ppo.critic(next_observation).squeeze(-1)
            target = reward + (1 - done) * self.discount * v_next

        v_s = self.ppo.critic(observation).squeeze(-1)
        critic_loss = 0.5 * (target.detach() - v_s).pow(2).mean() # aquest .detach() és redundant?

        advantage = (target - v_s).detach()
        with torch.no_grad():
            dist = self.ppo.actor(observation)
            log_prob = dist.log_prob(action).sum(dim=-1)

        new_dist = self.ppo.new_actor(observation)
        new_log_prob = new_dist.log_prob(action).sum(dim=-1)

        log_ratio = new_log_prob - log_prob.detach() # aquest .detach() és redundant?
        ratio = torch.exp(log_ratio)

        # entropy = new_dist.entropy().mean()
        actor_loss = -torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage).mean() # - self.beta * entropy
        return actor_loss, critic_loss

    def update_weights(self, actor_loss, critic_loss):

        # self.ppo.actor.load_state_dict(self.ppo.new_actor.state_dict())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_beta(self):
        self.beta = self.beta*self.beta_decay 

        
