import gym
import math
import numpy as np
import pandas as pd

q_table = np.zeros( (500, 6) )
round = 100000
discount_rate = 0.9
gamma = 0.1
epsilon = 0.6
state_value_history = np.zeros( (5000, 500) )

def get_rand_ac(obs):
    a = np.random.rand()
    if a < epsilon:
        return env.action_space.sample()
    return np.argmax(q_table[obs])

if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='ansi')
    round = 100000
    for rd in range(0,round):
        obs, info = env.reset()
        ter = False
        while not ter:
            ac = get_rand_ac(obs)
            obs_new, re, ter, tun, info = env.step(ac)
            q_table[obs][ac] = (1 - gamma) * q_table[obs][ac] + gamma * (re + discount_rate * np.max(q_table[obs_new]))
            obs = obs_new

        if rd % 20 == 0:
            state_value_history[rd//20] = np.max(q_table, axis=1)

    Q = pd.DataFrame(q_table)
    Q.to_csv('q_table',index=None)
    state_value_history_final = pd.DataFrame(state_value_history)
    state_value_history_final.to_csv('State_value', index=None)

