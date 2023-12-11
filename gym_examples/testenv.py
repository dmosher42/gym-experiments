import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from time import perf_counter

import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from gymnasium.envs.registration import register
from gym_examples.envs.Reinforce import REINFORCE
from gymnasium import envs
import time
import random


env = gym.make("gym_examples/ThrustBox-v0", render_mode="human")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(5e3)  # Total number of episodes
# Observation-space of ThrustBox-v0 (3)
obs_space_dims = env.observation_space.shape[0]
# Action-space of ThrustBox-v0 (1)
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []
seeds = [1,2,3,5,8,13]

t1_start = time.perf_counter()
for seed in seeds:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)
    rewards_over_seeds.append(reward_over_episodes)
t1_stop = time.perf_counter()
print(f"Elapsed time:{(t1_stop-t1_start)/60} minutes {(t1_stop-t1_start)%60} seconds")


# %%
# Plot learning curve
# ~~~~~~~~~~~~~~~~~~~
#

rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()
"""
env = gym.make('gym_examples/ThrustBox-v0',render_mode="human")
observation, info = env.reset()

#print(envs.registry.items())    # print the available environments

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

frames = []


for i_episode in range(200):
    observation = env.reset()
    for t in range(300):
        #print(f"Observation#{t}: {observation}")
        action = env.action_space.sample()    # take a random action\
        #print(f"ActionArray#{t}: {action}")
        env.step(action)
        #if done:
        #    print("Episode finished after {} timesteps".format(t+1))
        #    break

env.close()
"""