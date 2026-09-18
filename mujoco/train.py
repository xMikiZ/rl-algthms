import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, NormalizeObservation

import torch
from agent import AntAgent
from tqdm import tqdm
import numpy as np

num_episodes = 1000

num_envs = 4
# envs = gym.make_vec("Ant-v5", num_envs=4, render_mode = "rgb_array")

def make_env(env_id, idx, capture_video=False, run_name="a2c_exp"):
  def thunk():
    # 1. Must specify rgb_array render mode
    env = gym.make(env_id, render_mode="rgb_array")

    # 2. Apply RecordVideo ONLY to the first sub-environment (idx == 0)
    if capture_video and idx == 0:
      env = gym.wrappers.RecordVideo(
          env,
          video_folder=f"videos/{run_name}",
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

envs = gym.vector.SyncVectorEnv(env_fns)
device = torch.device("cpu")


num_observations = envs.single_observation_space.shape[0]
num_actions = envs.single_action_space.shape[0]
agent = AntAgent(
    num_observations = num_observations,
    num_actions = num_actions,
    env = envs,
    lr_actor = 0.0001,
    lr_critic = 0.0001,
    discount = 0.99
    )


n_updates = 1000
n_steps_per_update = 128

mean_reward = np.zeros(100)

for sample_phase in tqdm(range(n_updates)):

    ep_observations = torch.zeros(n_steps_per_update, num_envs, num_observations, device=device)
    ep_actions = torch.zeros(n_steps_per_update, num_envs, num_actions, device=device)
    ep_rewards = torch.zeros(n_steps_per_update, num_envs, device=device)
    ep_next_observations = torch.zeros(n_steps_per_update, num_envs, num_observations, device=device)
    terminateds = torch.zeros(n_steps_per_update, num_envs, device=device)

    # at the start of training reset all envs to get an initial state
    if sample_phase == 0:
        observations, info = envs.reset(seed=42)
        observations = torch.Tensor(observations).to(device)

    # play n steps in our parallel environments to collect data
    for step in range(n_steps_per_update):

        actions = agent.get_action(observations)

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        next_observations, rewards, terminated, truncated, infos = envs.step(
            actions.cpu().numpy()
        )

        next_observations = torch.Tensor(next_observations).to(device)
        rewards = torch.Tensor(rewards).to(device)
        terminated = torch.Tensor(terminated).to(device)

        ep_observations[step] = observations
        ep_actions[step] = actions
        ep_rewards[step] = rewards
        ep_next_observations[step] = next_observations

        observations = next_observations
        terminateds[step] = terminated

        mean_reward = np.append(mean_reward, rewards.mean())

    # calculate the losses for actor and critic
    actor_loss, critic_loss = agent.get_losses(
        ep_observations,
        ep_actions,
        ep_rewards,
        ep_next_observations,
        terminateds
    )

    # update the actor and critic networks
    agent.update_weights(actor_loss, critic_loss)

    if sample_phase % 50 == 0:
        print(mean_reward[:-100].mean())


envs.close()
