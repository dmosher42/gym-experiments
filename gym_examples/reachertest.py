import gymnasium as gym
from gymnasium.wrappers import RescaleAction
env = gym.make('Reacher-v4', render_mode = "human")
observation, info = env.reset()

from gymnasium import envs
#print(envs.registry.items())    # print the available environments

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)




for i_episode in range(200):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()    # take a random action
        env.step(action)
        # if done:
        #    print("Episode finished after {} timesteps".format(t+1))
        #    break
env.close()