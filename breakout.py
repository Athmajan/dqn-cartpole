import numpy as np
import gym
from gym.utils import passive_env_checker
np.bool8 = np.bool

env = gym.make("Breakout-v4", render_mode="human")
import ipdb; ipdb.set_trace()
env.reset()
env.render()

while True:
    act = int(input("Type Action"))
    env.step(act)
    env.render()
