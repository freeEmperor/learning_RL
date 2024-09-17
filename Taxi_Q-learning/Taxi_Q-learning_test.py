import gym
import numpy as np
import pandas as pd

q_table = np.array(pd.read_csv('./q_table'))
average_step = 0
wrong_action = 0
round = 10000

if __name__ == '__main__':
    env = gym.make('Taxi-v3',render_mode = 'ansi')
    for i in range(0,round):
        obs, info = env.reset()
        ter = False
        while not ter:
            ac = np.argmax(q_table[obs])
            obs, re, ter, tun, info = env.step(ac)
            average_step += 1
            if re == -10:
                wrong_action += 1

    print(f'We test this policy for {round} episodes.')
    print(f'It toke wrong action for {wrong_action} times.')
    print(f'The average steps of the episodes is {average_step/round}.')