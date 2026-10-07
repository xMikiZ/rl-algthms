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
            discount,
            lmd,
            beta,
            beta_decay
            ):

        self.a2c = A2C(num_observations, num_actions)

        self.env = env

        self.discount = discount
        self.beta = beta
        self.beta_decay = beta_decay

        self.lmd = lmd

        self.actor_optimizer = torch.optim.Adam(self.a2c.actor.parameters(), lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.a2c.critic.parameters(), lr_critic)

    def get_action(self, observation):

        dist = self.a2c.actor(observation)
        action = dist.sample()

        return action

    def get_gaes(self, observation, action, reward, next_observation, done, truncated):

        T = reward.shape[0]
        
        gae = torch.zeros(reward.shape[1])
        gaes = torch.zeros(reward.shape)

        v_s_all = self.a2c.critic(observation).squeeze(-1)
        v_next_all = self.a2c.critic(next_observation).squeeze(-1)

        for t in reversed(range(T)):

            # reset gae if episode finished
            gae = gae * (1 - done[t]) * (1 - truncated[t]) 

            r = reward[t]
            v_next = v_next_all[t]
            v_s = v_s_all[t]

            delta = r + self.discount * v_next * (1 - done[t]) - v_s
            gae = self.discount * self.lmd * gae + delta 
            gaes[t] = gae 

        returns = gaes + v_s_all
        return gaes, returns


    def get_losses(self, observation, action, reward, next_observation, done, gaes, returns):
        """Store experience as the loss"""

        # with torch.no_grad():
        #     v_next = self.a2c.critic(next_observation).squeeze(-1)
        #     target = reward + (1 - done) * self.discount * v_next
        

        v_s = self.a2c.critic(observation).squeeze(-1)
        critic_loss = 0.5 * (returns.detach() - v_s).pow(2).mean() # aquest .detach() és redundant?

        # advantage = (target - v_s).detach()
        dist = self.a2c.actor(observation)
        log_prob = dist.log_prob(action).sum(dim=-1)

        entropy = dist.entropy().mean()
        actor_loss = -(gaes*log_prob).mean() - self.beta * entropy

        return actor_loss, critic_loss

    def update_weights(self, actor_loss, critic_loss):

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.a2c.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.a2c.critic.parameters(), 0.5)
        self.critic_optimizer.step()

    def update_beta(self):
        self.beta = self.beta*self.beta_decay 

        
