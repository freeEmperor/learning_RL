import gym
import math
import numpy as np
import pandas as pd

policy = np.ones(3000).reshape(500,6) / 6
q_table = np.zeros(3000).reshape(500,6)
q_table_count = np.zeros(3000).reshape(500,6)
round = 100
discount_rate = 0.9

def encode(taxi_row, taxi_col, passenger_index, destination):
    return int (((taxi_row * 5 + taxi_col) * 5 + passenger_index) * 4 + destination)

def decode(observation):
    destination = observation % 4
    observation //= 4
    passenger_index = observation % 5
    observation //= 5
    taxi_col = observation % 5
    observation //= 5
    taxi_row = observation % 5
    return int(taxi_row), int(taxi_col), int(passenger_index), int(destination)

def get_rand_ac(obs):
    a = np.random.rand()
    ac = 0
    while a >= 0:
        a -= policy[obs][ac]
        ac += 1
    return int(ac - 1)

def deal_episode(epi):
    global q_table
    global q_table_count
    siz = len(epi)
    g = 0
    for i in range(siz-1,-1,-1):
        g = g + discount_rate * epi[i][2]
        q_table[epi[i][0]][epi[i][1]] += g
        q_table_count[epi[i][0]][epi[i][1]] += 1

def update_policy(rd):
    epsilon = 1-rd/(1.5*round)
    re = 0
    for obs in range(0, 500):
        mx = -100000
        mx_ac = 0
        for ac in range(0, 6):
            if q_table_count[obs][ac] != 0:
                q_table[obs][ac] /= q_table_count[obs][ac]
                if mx <= q_table[obs][ac]:
                    mx = q_table[obs][ac]
                    mx_ac = ac
            else:
                #print(decode(obs))
                re += 1
        '''
        if mx_ac == -1:
            for ac in range(0, 6):
                policy[obs][ac] = 1 / 6
            continue
        '''

        for ac in range(0, 6):
            policy[obs][ac] = epsilon / 6
        policy[obs][mx_ac] = 1 - 5 * epsilon / 6
    print(re)
    print('======================================================================')

if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='ansi')
    round = 10
    for rd in range(0,round):
        print(rd)
        step_num = 0
        q_table = np.zeros(3000).reshape(500, 6)
        q_table_count = np.zeros(3000).reshape(500, 6)
        for i in range(0,500):
            episode = []
            step_num = 0
            obs, info = env.reset()
            while step_num <= 20000:
                #if decode(obs)[2] == decode(obs)[3]:
                    #continue
                ac = get_rand_ac(obs)
                st = env.step(ac)
                episode.append( (obs, ac, st[1]) )
                obs = st[0]
                step_num += 1
                if st[2] == True:
                    print(step_num)
                    break
                if step_num == 20000:
                    print('fuckfuckfuck')
                    break
            deal_episode(episode)
        update_policy(rd)
        #print(rd)
        print(policy)
    Policy = pd.DataFrame(policy)
    Policy.to_csv('Policy')

