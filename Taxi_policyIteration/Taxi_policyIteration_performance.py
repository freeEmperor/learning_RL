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
    policy = np.array(pd.read_csv('./Policy'))
    policy = policy.T[0]
    obs, info = env.reset()
    env.render()
    ter = False
    while not ter:
        ac = int(policy[obs])
        obs, re, ter, tun, info = env.step(ac)
