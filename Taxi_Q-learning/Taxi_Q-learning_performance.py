import gym
import numpy as np
import pandas as pd

q_table = np.array(pd.read_csv('./q_table'))

if __name__ == '__main__':
    env = gym.make('Taxi-v3',render_mode = 'human')
    obs, info = env.reset()
    ter = False
    while not ter:
        ac = np.argmax(q_table[obs])
        obs, re, ter, tun, info = env.step(ac)
