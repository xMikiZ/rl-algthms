import gymnasium as gym
from agent import LunarLanderAgent
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import torch
import numpy as np

from tqdm import tqdm
import logging


n_episodes = 500

training_period = 250

env = gym.make("LunarLander-v3", render_mode = "rgb_array")
# env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# # https://gymnasium.farama.org/introduction/record_agent/
# env = RecordVideo(
#     env,
#     video_folder="videos",    # Folder to save videos
#     name_prefix="eval",               # Prefix for video filenames
#     episode_trigger=lambda x: x % training_period == 0   # Record every 250 episodes
# )

env = RecordEpisodeStatistics(env)

agent = LunarLanderAgent(
    env = env,
    n_obs = 8, 
    n_actions = 4, 
    buffer_size = 8192,
    mini_batch_size = 128,
    smoothing_factor= 0.005,
    eps_initial = 1,
    eps_decay = 0.99,
    eps_final = 0.025,
    discount = 0.99,
    learning_rate = 0.0001
)


clock = 0
for episode in tqdm(range(n_episodes)):

    old_observation, info = env.reset()
    terminated = False
    truncated = False

    episode_reward = 0
    step_count = 0

    while not terminated and not truncated:

        action = agent.get_action(torch.from_numpy(old_observation))

        observation, reward, terminated, truncated, info = env.step(action)

        agent.update_buffer(torch.as_tensor([*old_observation, action, reward, *observation, terminated]))
        
        agent.update_policy()

        old_observation = observation

        episode_reward += reward
        step_count += 1

    agent.decay_epsilon()

    if "episode" in info:
        episode_data = info["episode"]
        logging.info(f"Episode {episode}: "
                    f"reward={episode_data['r']:.1f}, "
                    f"length={episode_data['l']}, "
                    f"time={episode_data['t']:.2f}s")

        # Additional analysis for milestone episodes
        if episode % 100 == 0:
            # Look at recent performance (last 100 episodes)
            recent_rewards = list(env.return_queue)[-100:]
            if recent_rewards:
                avg_recent = sum(recent_rewards) / len(recent_rewards)
                print(f"  -> Average reward over last 100 episodes: {avg_recent:.1f}")

env.close()

print(f'\nEvaluation Summary:')
print(f'Episode durations: {list(env.time_queue)}')
print(f'Episode rewards: {list(env.return_queue)}')
print(f'Episode lengths: {list(env.length_queue)}')

# Calculate some useful metrics
avg_reward = np.sum(env.return_queue)
avg_length = np.sum(env.length_queue)
std_reward = np.std(env.return_queue)

print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
print(f'Average episode length: {avg_length:.1f} steps')
print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')
