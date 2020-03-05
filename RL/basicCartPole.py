import random
import time
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from statistics import mean as mean
scores = []

ENV_NAME = "CartPole-v1"
#ENV_NAME = "LunarLander-v2"

LEARNING_RATE = 0.001

NUM_PREV_STATES = 5

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

AVERAGE_THRESHOLD = 150.0

'''
Num	Observation            Min       Max
0	Cart Position         -2.4       2.4
1	Cart Velocity         -Inf       Inf
2	Pole Angle         ~ -41.8°   ~ 41.8°
3	Pole Velocity At Tip  -Inf       Inf
'''

def cartpole():
    #setup the simulation
    env = gym.make(ENV_NAME)
    env._max_episode_steps = 2000 #not that we will need this much time...
    #get size of observation (states) and action spaces
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    action_list = [0,1] # 0: push to left, 1: push to right
    #set up the training loop
    run = 0
    num_trainings = int(input("Enter number of training cycles: "))
    for run in range(num_trainings):
        #reset the simulation
        state = env.reset()
        #state = [position of cart, velocity of cart, angle of pole, Pole Velocity At Tip]

        state = np.reshape(state, [1, observation_space])
        #setup the simulation loop
        step = 0
        while True: # keep going until failure
            step += 1
            #env.render() # uncomment this line to see visualizations of training simulations (not recommended)
            #get the desired action based on our model and current state
            #action = random.choice(action_list)
            if (state[0, 3] >= 0):
                action = 1
            else:
                action = 0
            #action = 0
            #do the action and get the results
            state_next, reward, sim_done, _ = env.step(action)
            #if sim terminates, make reward negative (bad thing happened)
            reward = reward if not sim_done else -reward
            #have the model remember what happend with the action
            state_next = np.reshape(state_next, [1, observation_space])
            state = state_next
            #time.sleep(.1)
            if sim_done:
                print("Run: {:5} ,  score: {:5}" .format(run,step))
                scores.append(step)
                break
    env.close()

if __name__ == "__main__":
    try:
        cartpole()
        print("Max Score:", max(scores))
        print("Min Score:", min(scores))
        print("Average Score:", mean(scores))
    except KeyboardInterrupt:
        print("Max Score:", max(scores))
        print("Min Score:", min(scores))
        print("Average Score:", mean(scores))