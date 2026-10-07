import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, NormalizeObservation
from gymnasium.wrappers.vector import ClipAction

import torch
from agent import AntAgent
from tqdm import tqdm
import numpy as np

num_episodes = 1000

# envs = gym.make_vec("Ant-v5", num_envs=4, render_mode = "rgb_array")

def make_env(env_id, idx, capture_video=False, run_name="a2c_exp"):
  def thunk():
    # 1. Must specify rgb_array render mode
    if idx == 0:
        env = gym.make(env_id, render_mode="rgb_array", include_cfrc_ext_in_observation = False)
    else:
        env = gym.make(env_id, include_cfrc_ext_in_observation = False)

    env = NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=0.997)   
 
    # 2. Apply RecordVideo ONLY to the first sub-environment (idx == 0)
    if capture_video and idx == 0:
      env = gym.wrappers.RecordVideo(
          env,
          video_folder=f"videos/bullshit",
          episode_trigger=lambda ep_id: ep_id % 100
          == 0, 
      )
    return env

  return thunk


# Instantiate vectorized environments
num_envs = 8
env_fns = [
    make_env("Ant-v5", idx=i, capture_video=True) for i in range(num_envs)
]

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
    discount = 0.997,
    beta = 0.001,
    beta_decay = 0.991
    )


n_updates = 1000
trajectory_size = 1024

mean_reward = np.zeros(100)

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

        terminateds[step] = terminated
        truncateds[step] = truncated
        
        observations = next_observations

        mean_reward = np.append(mean_reward, rewards.mean())

    # print(
    #     ep_observations.shape,
    #     ep_actions.shape,
    #     ep_rewards.shape,
    #     ep_next_observations.shape,
    #     terminateds.shape,
    #     truncateds.shape,
    # )

    K = 1
    mini_batch_size = 128
    for _ in range(K):

        indices = np.arange(trajectory_size * num_envs)
        np.random.shuffle(indices)
        
        for i in range(0, trajectory_size*num_envs, mini_batch_size):

            mb_idx = indices[i : i + mini_batch_size]

            # ep, get_losses funciona per batches, aquí estàs passant només vectors...!
            # print(ep_observations.view(trajectory_size * num_envs, -1), ep_actions.view(trajectory_size * num_envs, -1))
            # calculate the losses for actor and critic
            actor_loss, critic_loss = agent.get_losses(
                ep_observations.view(trajectory_size * num_envs, -1)[mb_idx],
                ep_actions.view(trajectory_size * num_envs, -1)[mb_idx],
                ep_rewards.view(trajectory_size * num_envs)[mb_idx],
                ep_next_observations.view(trajectory_size * num_envs, -1)[mb_idx],
                terminateds.view(trajectory_size * num_envs)[mb_idx]
            )

            # update the actor and critic networks
            agent.update_weights(actor_loss, critic_loss)
            agent.update_beta()


    if sample_phase % 50 == 0:
        print(mean_reward[:-100].mean())


envs.close()
