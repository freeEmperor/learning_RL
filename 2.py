import gym

policy = 0


def encode(taxi_row, taxi_col, passenger_index, destination):
    return ((taxi_row * 5 + taxi_col) * 5 + passenger_index) * 4 + destination

def decode(observation):
    destination = observation % 4
    observation //= 4
    passenger_index = observation % 5
    observation //= 5
    taxi_col = observation % 5
    observation //= 5
    taxi_row = observation % 5
    return taxi_row, taxi_col, passenger_index, destination

def printit(env, n):
    print(env.step(n))

if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='human')

    ob, info = env.reset(seed=0)
    lis = [ 1,2,2,2,0,0,4,1,1,3,3,3,1,5,0,0]
    for i in range(0, len(lis)):
        printit(env, lis[i])

    '''
    env.render()
    '''
    env.render()
