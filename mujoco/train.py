import gymnasium as gym
import torch
from agent import AntAgent
from tqdm import tqdm


num_episodes = 10000

env = gym.make("Ant-v5", render_mode = "human")


agent = AntAgent(
    num_observations = env.observation_space.shape[0],
    num_actions = env.action_space.shape[0],
    env = env,
    lr = 0.0001,
    discount = 0.99,
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

        agent.update_weights(observation, action, reward, new_observation)

        agent.update_discount()

        observation = new_observation

env.close()

        

