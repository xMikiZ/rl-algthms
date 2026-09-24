import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, NormalizeObservation
from gymnasium.wrappers.vector import ClipAction

import torch
from agent import AntAgent
from tqdm import tqdm
import numpy as np


def make_env(env_id, idx, capture_video=False):
  def thunk():

    if idx == 0:
        env = gym.make(env_id, render_mode="rgb_array", include_cfrc_ext_in_observation=False)
    else:
        env = gym.make(env_id, include_cfrc_ext_in_observation=False)

    env = NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=0.997)
    env = RecordEpisodeStatistics(env)

    # Apply RecordVideo ONLY to the first sub-environment (idx == 0)
    if capture_video and idx == 0:
      env = gym.wrappers.RecordVideo(
          env,
          video_folder=f"videos",
          episode_trigger=lambda ep_id: ep_id % 50
          == 0,  # Record every 100th episode on worker 0
      )
    return env

  return thunk


# Instantiate vectorized environments
num_envs = 16 
env_fns = [
    make_env("Ant-v5", idx=i, capture_video=True) for i in range(num_envs)
]

# https://farama.org/Vector-Autoreset-Mode
envs = gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
envs = ClipAction(envs)
device = torch.device("cpu")


num_observations = envs.single_observation_space.shape[0]
num_actions = envs.single_action_space.shape[0]
agent = AntAgent(
    num_observations = num_observations,
    num_actions = num_actions,
    env = envs,
    lr_actor = 0.0001,
    lr_critic = 0.0005,
    eps = 0.1,
    lmd = 0.96,
    discount = 0.997,
    beta = 0,
    beta_decay = 1
    )

# agent.ppo.load_state_dict(torch.load("begin_ppo_weights_600.pt"))

n_updates = 1000
trajectory_size = 512
mini_batch_size = 64    
K = 4


mean_reward = np.zeros(n_updates)

for sample_phase in tqdm(range(n_updates)):

    ep_observations = torch.zeros(trajectory_size, num_envs, num_observations, device=device)
    ep_actions = torch.zeros(trajectory_size, num_envs, num_actions, device=device)
    ep_rewards = torch.zeros(trajectory_size, num_envs, device=device)
    ep_next_observations = torch.zeros(trajectory_size, num_envs, num_observations, device=device)
    terminateds = torch.zeros(trajectory_size, num_envs, device=device)
    truncateds = torch.zeros(trajectory_size, num_envs, device=device)

    # at the start of training reset all envs to get an initial state
    if sample_phase == 0:
        observations, info = envs.reset(seed=42)
        observations = torch.Tensor(observations).to(device)

    # play n steps in our parallel environments to collect data
    for step in range(trajectory_size):

        actions = agent.get_action(observations)

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        next_observations, rewards, terminated, truncated, infos = envs.step(
            actions.cpu().numpy()
        )
        next_observations = torch.Tensor(next_observations).to(device)
        rewards = torch.Tensor(rewards).to(device)
        terminated = torch.Tensor(terminated).to(device)
        truncated = torch.Tensor(truncated).to(device)

        ep_observations[step] = observations
        ep_actions[step] = actions
        ep_rewards[step] = rewards

        for i in range(num_envs):
            if terminated[i] or truncated[i]: 
                ep_next_observations[step][i] = torch.Tensor(infos["final_obs"][i]).to(device)
            else:
                ep_next_observations[step][i] = next_observations[i]

        observations = next_observations
                
        terminateds[step] = terminated
        truncateds[step] = truncated

        mean_reward[sample_phase] += rewards.mean()

    with torch.no_grad():
        gaes, returns = agent.get_gaes(ep_observations, ep_actions, ep_rewards, ep_next_observations, terminateds, truncateds)
        log_probs = agent.get_log_prob(ep_observations, ep_actions)

    unsq_observations = ep_observations.view(trajectory_size * num_envs, -1)
    unsq_actions = ep_actions.view(trajectory_size * num_envs, -1)
    unsq_rewards = ep_rewards.view(trajectory_size * num_envs, -1).squeeze()
    unsq_next_observations = ep_next_observations.view(trajectory_size * num_envs, -1)
    unsq_terminateds = terminateds.view(trajectory_size * num_envs, -1).squeeze()

    unsq_gaes = gaes.view(trajectory_size*num_envs, -1).squeeze()
    unsq_returns = returns.view(trajectory_size*num_envs, -1).squeeze()
    unsq_log_probs = log_probs.view(trajectory_size*num_envs, -1).squeeze()

    for _ in range(K):

        indices = np.arange(trajectory_size * num_envs)
        np.random.shuffle(indices)
        
        for i in range(0, trajectory_size*num_envs, mini_batch_size):

            mb_idx = indices[i : i + mini_batch_size]

            # normalize gaes across minibatch
            mb_gaes = unsq_gaes[mb_idx]

            actor_loss, critic_loss = agent.get_losses(
                unsq_observations[mb_idx],
                unsq_actions[mb_idx],
                unsq_rewards.squeeze()[mb_idx],
                unsq_next_observations[mb_idx],
                unsq_terminateds.squeeze()[mb_idx],
                mb_gaes,
                unsq_log_probs[mb_idx],
                unsq_returns[mb_idx]
            )

            # update the actor and critic networks
            agent.update_weights(actor_loss, critic_loss)
            agent.update_beta()

    if sample_phase % 5 == 0:
        print(mean_reward[sample_phase] / trajectory_size)

    if sample_phase % 200 == 0:
       torch.save(agent.ppo.state_dict(), f"ppo_weights_{sample_phase}.pt")


envs.close()
