import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

import torch
from agent import AntAgent
from tqdm import tqdm
import logging

num_episodes = 1000

env = gym.make("Ant-v5", render_mode = "rgb_array")
env = RecordEpisodeStatistics(env)



agent = AntAgent(
    num_observations = env.observation_space.shape[0],
    num_actions = env.action_space.shape[0],
    env = env,
    lr = 0.0001,
    discount = 0.9996,
    batch_size = 128
)


for episode in tqdm(range(num_episodes)):

    observation, info = env.reset()
    observation = torch.from_numpy(observation).to(torch.float32) # torch uses float32 as default, but this is float64

    terminated, truncated = False, False

    agent.restart_discount()

    while not terminated and not truncated:

        action = agent.get_action(observation)
        clamped_action = torch.clamp(action, -1, 1)

        new_observation, reward, terminated, truncated, info = env.step(clamped_action.numpy())
        new_observation = torch.from_numpy(new_observation).to(torch.float32)

        agent.update_weights(observation, action, reward, new_observation, terminated)

        observation = new_observation

        agent.update_discount()

    if "episode" in info:
        episode_data = info["episode"]
        logging.info(f"Episode {episode}: "
                    f"reward={episode_data['r']:.1f}, "
                    f"length={episode_data['l']}, "
                    f"time={episode_data['t']:.2f}s")

        # Additional analysis for milestone episodes
        if episode % 25 == 0:
            # Look at recent performance (last 25 episodes)
            recent_rewards = list(env.return_queue)[-25:]
            if recent_rewards:
                avg_recent = sum(recent_rewards) / len(recent_rewards)
                print(f"  -> Average reward over last 25 episodes: {avg_recent:.1f}")

env.close()

        

