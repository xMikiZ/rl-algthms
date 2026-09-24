import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from mlp import PPO




class AntAgent():

    def __init__(
            self,
            num_observations,
            num_actions,
            env: gym.Env,
            lr_actor,
            lr_critic,
            eps,
            lmd,
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

        self.lmd = lmd


        self.actor_optimizer = torch.optim.Adam(self.ppo.actor.parameters(), lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.ppo.critic.parameters(), lr_critic)

    def get_action(self, observation):

        dist = self.ppo.actor(observation)
        action = dist.sample()

        return action

    def get_gaes(self, observations, actions, rewards, next_observations, done, truncated):

        T = rewards.shape[0]
        
        gae = torch.zeros(rewards.shape[1])
        gaes = torch.zeros(rewards.shape)

        v_s_all = self.ppo.critic(observations).squeeze(-1)
        v_next_all = self.ppo.critic(next_observations).squeeze(-1)

        for t in reversed(range(T)):

            # reset gae if episode finished
            gae = gae * (1 - done[t]) * (1 - truncated[t]) 

            r = rewards[t]
            v_next = v_next_all[t]
            v_s = v_s_all[t]

            delta = r + self.discount * v_next * (1 - done[t]) - v_s
            gae = self.discount * self.lmd * gae + delta 
            gaes[t] = gae 

        returns = gaes + v_s_all
        return gaes, returns

    def get_log_prob(self, observation, action):

        dist = self.ppo.actor(observation)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return log_prob
    
    def get_losses(self, observation, action, reward, next_observation, done, gaes, log_prob, returns):
        """Store experience as the loss"""

        # with torch.no_grad():
        #     v_next = self.ppo.critic(next_observation).squeeze(-1)
        #     target = reward + (1 - done) * self.discount * v_next
 

        v_s = self.ppo.critic(observation).squeeze(-1)
        critic_loss = 0.5 * (returns.detach() - v_s).pow(2).mean() # aquest .detach() és redundant?

        new_dist = self.ppo.actor(observation)
        new_log_prob = new_dist.log_prob(action).sum(dim=-1)

        log_ratio = new_log_prob - log_prob
        ratio = torch.exp(log_ratio)

        entropy = new_dist.entropy().mean()
        actor_loss = -torch.min(ratio * gaes, torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * gaes).mean() - self.beta * entropy
        return actor_loss, critic_loss

    def update_weights(self, actor_loss, critic_loss):
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.ppo.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.ppo.critic.parameters(), 0.5)
        self.critic_optimizer.step()

    def update_beta(self):
        self.beta = self.beta*self.beta_decay 

        
