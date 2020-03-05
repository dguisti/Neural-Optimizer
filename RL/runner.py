'''
The following code is an annotated and modified version of the example
code found at https://github.com/gsurma/cartpole/blob/master/cartpole.py

requires the following modules/packages:
  numpy
  gym
  keras
  matplotlib
  tensorflow

November 2019
'''
import random
import time
import gym
import numpy as np
from collections import deque
#import dequey
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import load_model

ENV_NAME = "CartPole-v1"
AVERAGE_THRESHOLD = 200.00

def cartpole():
    fName = input("Please input filename: ")
    model = load_model(fName)
    cycleNum = int(input("Please input number of training cycles: "))
    #setup the simulation
    env = gym.make(ENV_NAME)
    env._max_episode_steps = 2000 #not that we will need this much time...
    #how we will keep track of how well we are doing (not really needed, but good to see)
    #get size of observation (states) and action spaces
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    #use our model to show how well it does (or doesn't)
    env.render()
    #time.sleep(3)
    scores = []
    for i in range(cycleNum):
        state = env.reset()
        step = 0
        sim_done = False
        while not sim_done:
            env.render()
            time.sleep(0.001)
            step += 1
            state = np.reshape(state, [1, observation_space])
            q_values = model.predict(state)
            randVal = random.random()
            if q_values[0][0] > randVal: action = 0
            else: action = 1
            state, reward, sim_done, _ = env.step(action)
            if sim_done:
                print("Done after ", step, " steps.")
                scores.append(step)
        env.close()

    print("Max:", max(scores), "Min:", min(scores), "Mean:", sum(scores)/len(scores))

if __name__ == "__main__":
    cartpole()
