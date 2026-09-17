import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from mlp import Policy, ValueFunction




class bbb():

    def __init__(
            self,
            num_observations,
            num_actions,
            env: gym.Env,
            lr,
            eps,
            discount,
            batch_size
            ):

        self.policy = Policy(num_observations, num_actions)
        self.new_policy = Policy(num_observations, num_actions)

        self.value_function = ValueFunction(num_observations)

        self.num_observations = num_observations
        self.num_actions = num_actions

        self.env = env
        self.lr = lr
        self.eps = eps
        self.discount = discount

        self.policy_optimizer = torch.optim.Adam(self.new_policy.parameters(), lr)
        self.value_function_optimizer = torch.optim.Adam(self.value_function.parameters(), lr)



        self.batch_size = batch_size
        # TODO: Unify in a Buffer class that stores this subbuffers 
        # self.log_probs_batch = [0]*batch_size
        # self.advantages_batch = [0]*batch_size
        # self.observatios_batch = [0]*batch_size
        # self.actions_batch = [0]*batch_size

        # self.critic_loss_batch = [0]*batch_size

        self.log_probs_batch = [0]*batch_size
        self.advantages_batch = [0]*batch_size
        self.observatios_batch = [0]*batch_size
        self.actions_batch = [0]*batch_size

        self.critic_loss_batch = [0]*batch_size
        self.batch_clock = 0

    def get_action(self, observation):

        dist = self.policy(observation)
        action = dist.sample()

        return action

    def update_discount(self):
        self.cumulative_discount *= self.discount

    def restart_discount(self):
        self.cumulative_discount = 1

    def update_batch_clock(self):
        self.batch_clock = (self.batch_clock + 1) % self.batch_size 

    def update_weights(self, observation, action, reward, next_observation, done):

        dist = self.policy(observation)
        v_s = self.value_function(observation)

        v_next = self.value_function(next_observation).detach()
        target = reward + (1 - done)*self.discount * v_next

        advantage = (target - v_s).detach()

        # actor loss
        log_prob = dist.log_prob(action).sum(dim=-1)

        self.log_probs_batch[self.batch_clock] = log_prob
        self.advantages_batch[self.batch_clock] = advantage
        self.observatios_batch[self.batch_clock] = observation
        self.actions_batch[self.batch_clock] = action

        # critic loss
        critic_loss = 0.5*(target - v_s)**2
        self.critic_loss_batch[self.batch_clock] = critic_loss

        if self.batch_clock == self.batch_size - 1:

            self.policy.load_state_dict(self.new_policy.state_dict())

            log_probs = torch.stack(self.log_probs_batch).detach()
            advantages = torch.stack(self.advantages_batch).detach()
            observations = torch.stack(self.observatios_batch).detach()
            actions = torch.stack(self.actions_batch).detach()


            distributions = self.new_policy(observations)
            new_log_probs = distributions.log_prob(actions).sum(dim=-1)

            log_ratios = new_log_probs - log_probs
            ratios = torch.exp(log_ratios)
            
            surr1 = ratios*advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps)*advantages

            #TODO: K epochs as a hyperparam to pass to agent
            #      also for minibatch
            K = 16
            mini_batch = 32
            for _ in range(K):

                indicies = np.random.choice(self.batch_size, mini_batch, replace=False)
                batch_actor_loss = -torch.min(surr1[indicies], surr2[indicies]).mean()

                self.policy_optimizer.zero_grad()
                batch_actor_loss.backward()
                # recomanació del gemini: sinó s'acumulen masses gradients i torna nan
                torch.nn.utils.clip_grad_norm_(self.new_policy.parameters(), max_norm=0.5)
                self.policy_optimizer.step()

                batch_critic_loss = torch.stack(self.critic_loss_batch[indicies]).mean()
                self.value_function_optimizer.zero_grad()
                batch_critic_loss.backward()
                self.value_function_optimizer.step()
            
        

        self.update_batch_clock()
        
                    