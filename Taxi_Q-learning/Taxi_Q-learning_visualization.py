import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import imageio.v2 as imageio


val = np.array(pd.read_csv('./State_value'))
round = len(val)
pas = 1
des = 0
loc = {0:(0,0), 1:(0,4), 2:(4,0), 3:(4,3), 4:'taxi'}

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

def select(pas, des):
    lis = []
    for i in range(0,500):
        tp = decode(i)
        if tp[2] == pas and tp[3] == des:
            lis.append(True)
        else:
            lis.append(False)
    return np.array(lis)

if __name__ == '__main__':

    x = np.arange(0, 5, 1)
    y = np.arange(0, 5, 1)
    x, y = np.meshgrid(x, y)
    for i in range(0, round//25):
        z = np.zeros((5, 5))
        for xi in range(0, 5):
            for yi in range(0, 5):
                z[xi][yi] = val[i][encode(xi,yi,pas,des)]

        ax = plt.figure(figsize=(10,8)).add_subplot(projection='3d')
        ax.view_init(elev=19, azim=136)
        ax.plot_surface(x, y, z, rcount=10, ccount=10, cmap=plt.get_cmap('rainbow'), antialiased=True)
        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_yticks(np.arange(0, 5, 1))
        ax.set_xlabel('taxi_row')
        ax.set_ylabel('taxi_column')
        ax.set_zlabel('state_value')
        ax.set_title( f'State value with: passenger in {loc[pas]}, destination is {loc[des]}, after {20*i} episode')
        plt.savefig(f'./figure/figure{i+1}.jpg',dpi = 100)
        plt.close()

    with imageio.get_writer(uri=f'state_value_iteration({pas},{des}).gif', mode='I', fps=5) as writer:
        for i in range(0, round//25):
            writer.append_data(imageio.imread(f'./figure/figure{i+1}.jpg'))



