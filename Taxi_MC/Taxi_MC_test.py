import gym
import numpy as np
import pandas as pd


def encode(taxi_row, taxi_col, passenger_index, destination):
    return int(((taxi_row * 5 + taxi_col) * 5 + passenger_index) * 4 + destination)


def decode(observation):
    destination = observation % 4
    observation //= 4
    passenger_index = observation % 5
    observation //= 5
    taxi_col = observation % 5
    observation //= 5
    taxi_row = observation % 5
    return int(taxi_row), int(taxi_col), int(passenger_index), int(destination)


if __name__ == '__main__':
    env = gym.make('Taxi-v3',render_mode = 'human')
    po = np.array(pd.read_csv('./Policy'))
    policy = [0]*500
    for obs in range(0,500):
        mx_ac = -1
        mx = -1
        for ac in range(1,7):
            if mx < po[obs][ac]:
                mx = po[obs][ac]
                mx_ac = ac - 1
        policy[obs] = mx_ac

    #print(policy)
    observation, info = env.reset()
    env.render()
    if decode(observation)[2] != decode(observation)[3]:
        o = 0
        for i in range(0, 1000):
            print(int(policy[observation]))
            st = env.step(int(policy[observation]))
            if st[2] == True:
                break
            observation = st[0]
