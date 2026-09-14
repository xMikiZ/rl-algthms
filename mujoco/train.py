import gymnasium as gym
import torch
from agent import AntAgent


num_episodes = 10000

env = gym.make("Ant-v5", render_mode = "human")


agent = AntAgent(
    num_observations = env.observation_space.shape[0],
    num_actions = env.action_space.shape[0],
    env = env,
    lr_actor = 0.0001,
    lr_critic = 0.0001,
    discount = 0.99,
)


for episode in range(num_episodes):

    observation, info = env.reset()
    observation = torch.from_numpy(observation).to(torch.float32) # torch uses float32 as default, but this is float64

    terminated, truncated = False, False

    agent.restart_discount()

    while not terminated or truncated:

        action = agent.get_action(observation)

        new_observation, reward, terminated, truncated, info = env.step(action)
        new_observation = torch.from_numpy(new_observation).to(torch.float32)

        agent.update_weights(observation, action, reward, new_observation)

        agent.update_discount()

        observation = new_observation

env.close()

        

