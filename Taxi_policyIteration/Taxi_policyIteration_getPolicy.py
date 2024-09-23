import gym
import numpy as np
import pandas as pd

round = 100
policy = np.ones(500) * 5
matrix_P = np.zeros((500,500))
array_R = np.zeros(500)
discount_rate = 0.9
state_value = np.zeros(500)
state_value_history = np.zeros((round,500))
dict = {}


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

def get_model():
    global dict
    dest12 = {0:(0,0), 1:(0,4), 2:(4,0), 3:(4,3)}
    dest21 = np.ones(25).reshape(5,5) * -1
    lis0 = [(4,0),(4,1),(4,2),(4,3),(4,4)]
    lis1 = [(0,0),(0,1),(0,2),(0,3),(0,4)]
    lis2 = [(3,0),(4,0),(3,2),(4,2),(0,1),(1,1),(0,4),(1,4),(2,4),(3,4),(4,4)]
    lis3 = [(3,1),(4,1),(3,3),(4,3),(0,2),(1,2),(0,0),(1,0),(2,0),(3,0),(4,0)]
    for i in range(0,4):
        dest21[dest12[i][0]][dest12[i][1]] = i
    for obs in range(0, 500):
        for ac in range(0, 6):
            tx, ty, pa, de = decode(obs)
            re = -1
            ter = False
            obs_nx = obs
            if ac == 5:
                if pa != 4:
                    re = -10
                elif (tx,ty) == dest12[de]:
                    pa = True
                    re = 20
                    obs_nx = encode(tx,ty,de,de)
                elif dest21[tx][ty] != -1:
                    re = -10
                    obs_nx = encode(tx, ty, dest21[tx][ty], de)
                else:
                    re = -10
            elif ac == 4:
                if pa == 4:
                    re = -10
                elif dest12[pa] == (tx,ty):
                    obs_nx = encode(tx, ty, 4, de)
                else:
                    re = -10
            elif ac == 3:
                if (tx,ty) not in lis3:
                    obs_nx = encode(tx, ty - 1, pa, de)
            elif ac == 2:
                if (tx,ty) not in lis2:
                    obs_nx = encode(tx, ty + 1, pa, de)
            elif ac == 1:
                if (tx,ty) not in lis1:
                    obs_nx = encode(tx - 1, ty, pa, de)
            elif ac == 0:
                if (tx,ty) not in lis0:
                    obs_nx = encode(tx + 1, ty, pa, de)
            dict[(obs, ac)] = (obs_nx, re, ter)
            #print(decode(obs),ac,decode(obs_nx))

def get_P():
    global matrix_P
    matrix_P = np.zeros((500, 500))
    for obs in range(0,500):
        ac = policy[obs]
        #print(dict[(obs,ac)][0])
        #print(type(dict[(obs,ac)][0]))
        matrix_P[dict[(obs,ac)][0]][obs] = 1

def get_R():
    global array_R
    for obs in range(0,500):
        ac = policy[obs]
        array_R[obs] = dict[(obs,ac)][1]

def iteration_state_value():
    global state_value
    value = state_value
    value_new = array_R + discount_rate * np.matmul(value , matrix_P)
    while np.linalg.norm(value - value_new, ord=2) > 0.001 :
        value = value_new
        value_new = array_R + discount_rate * np.matmul(value, matrix_P)
    state_value = value_new

def update_policy():
    global policy
    for obs in range(0,500):
        mx = -100000
        mx_ac = 0
        for ac in [5,4,3,2,1,0]:
            val = dict[(obs,ac)][1] + discount_rate * state_value[dict[(obs,ac)][0]]
            if mx < val:
                mx = val
                mx_ac = ac
        policy[obs] = mx_ac

if __name__ == '__main__':

    get_model()

    for rd in range(0,round) :
        get_P()
        get_R()
        iteration_state_value()
        state_value_history[rd] = state_value
        update_policy()
    #print(policy)
    policy_final = pd.Series(policy)
    policy_final.to_csv('Policy',index=None)
    #print(state_value_history)
    state_value_history_final = pd.DataFrame(state_value_history)
    state_value_history_final.to_csv('State_value',index=None)

