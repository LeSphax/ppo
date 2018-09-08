import gym
import gym_ui
import time
import random
import numpy as np

env = gym.make("RandomButton-v0")
env.reset()

for _ in range(300):
    # env.render()

    action = np.random.rand(2)
    print(action)
    env.step(action)
    time.sleep(0.3)
