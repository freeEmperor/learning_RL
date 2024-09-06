import gym
policy = 0



env =gym.make('Taxi-v3',render_mode = 'ansi')
op , info = env.reset(seed=0)
print()