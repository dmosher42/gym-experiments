from __future__ import annotations

import random
import time
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from gymnasium.envs.registration import register
from gymnasium.experimental.wrappers.rendering import RecordVideoV0

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

import os

import gymnasium as gym


register(
    id="gym_examples/SimpleThrustBox-Z-v0",
    entry_point="gym_examples.envs:SimpleThrustboxEnvZ"
)

plt.rcParams["figure.figsize"] = (10, 5)

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Old REINFORCE Here

def triggerrecord (i: int):
    return i%1000==0

logging.basicConfig(level=logging.INFO)

savedata = False
savelogs = False
savemodel = False

# from logging cookbook
# create logger with 'spam_application'
logger = logging.getLogger("z_axis")
# component_logger = logger.getChild("component-a")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
startedtime = time.strftime("%Y-%m-%d.%H_%M_%S")

set_episode_steps = 1500

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)

if savedata|savelogs|savemodel:
    savepath = f"data/{startedtime}/"
    os.mkdir(savepath)

if savelogs:
    # create directory for all saved data

    logfilename = f'z_axis.{startedtime}.log'

    print(f"Saving Logs to {savepath+logfilename}")

    fh = logging.FileHandler(savepath+logfilename)
    fh.setLevel(logging.DEBUG)
    logger.info(f"Saving log to {logfilename}")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
else:
    print("NOT SAVING LOGS")

renderer_hum=False
# Set name prefix for video recordings
recording_name_prefix = "rl-video"

# Create and wrap the environment
if renderer_hum:
    rendermode="human"
else:
    rendermode="rgb_array"
env = gym.make("gym_examples/SimpleThrustBox-Z-v0",max_episode_steps=set_episode_steps,render_mode= rendermode,
               usebuilder = False)

wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

if savedata:
    wrapped_env = RecordVideoV0(wrapped_env,savepath,episode_trigger=triggerrecord)
total_num_episodes = int(1e4) #int(5e3) # Total number of episodes
# Observation-space
obs_space_dims = env.observation_space.shape[0]
# Action-space
action_space_dims  = env.action_space.shape[0]
rewards_over_seeds = []
time_over_seeds    = []
highscore = -np.inf

seeds=[1, 2, 3, 5, 8, 13, 21, 34]

agent = PPO("MlpPolicy", wrapped_env, verbose=1)
if savelogs:
    # set up training logger
    traininglogfilename = f'z_axis.{startedtime}_training.log'
    train_logger = configure(savepath + traininglogfilename, ["stdout", "csv", "tensorboard"])
    agent.set_logger(train_logger)

logger.info(f"Starting training on {len(seeds)} seeds.")
timer_all_start = time.time()
for seed in seeds:  # Fibonacci seeds
    timer_seed_start = time.time()
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    # agent = REINFORCE(obs_space_dims, action_space_dims)

    # Stable Baselines 3
    agent.learn(total_timesteps=set_episode_steps)

    reward_over_episodes = []
    time_over_episodes   = []
    # Do logging and change video naming prefix for seed
    logger.info(f"Seed: {seed}")
    # wrapped_env.name_prefix = f"{recording_name_prefix}-seed-{seed}"

    for episode in range(total_num_episodes):
        # obs_old = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        timer_eps_start = time.time()
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)
        # print(obs)

        done = False
        while not done:
            #action = agent.sample_action(obs)

            action,_states = agent.predict(obs,deterministic = True)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            # print(obs_old - obs)
            # obs_old = obs
            #agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated
        run_reward = wrapped_env.return_queue[-1]
        reward_over_episodes.append(run_reward)
        time_over_episodes.append(time.time()-timer_eps_start)
        #agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            avg_time = (np.mean(time_over_episodes))
            est_time = ((total_num_episodes-episode)*avg_time)
            logger.info(f"Episode:{episode}  Average Reward: {avg_reward}  Average Time: {round(avg_time,4)}s  Est. Seed Time Left:{round(est_time / 60)} minutes {round(est_time % 60,4)}s")

    rewards_over_seeds.append(reward_over_episodes)
    time_over_seeds.append(time.time()-timer_seed_start)
    logger.info(f"Seed:{seed} finished  Time:{np.mean(time_over_seeds)}  Est. remaining: {round(len(seeds) * np.mean(time_over_seeds), 4)}")

    if savemodel:
        agent.save(f"ppo_thrustbox_{seed}")

logger.info(f"Run time: {(np.sum(time_over_seeds))}")
rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title=f"PPO for SimpleThrustbox-v0 at {startedtime}"
)
plt.show()