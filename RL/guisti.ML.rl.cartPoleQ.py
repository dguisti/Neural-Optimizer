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
from datetime import datetime

ENV_NAME = "CartPole-v1"

GAMMA = 0.99 #discount factor for future value 0.85
LEARNING_RATE = 0.00005 #how much the new information can change the existing information
ALPHA = LEARNING_RATE

MEMORY_SIZE = 100_000
BATCH_SIZE = 100

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.001

# Reward Params
'''
Num	Observation            Min       Max
0	Cart Position         -2.4       2.4
1	Cart Velocity         -Inf       Inf
2	Pole Angle         ~ -41.8°   ~ 41.8°
3	Pole Velocity At Tip  -Inf       Inf
'''
REWARD_FACTOR = .02
POSITION_WEIGHT = 0.25 * REWARD_FACTOR
VELOCITY_WEIGHT = 0.5 * REWARD_FACTOR
ANGLE_WEIGHT = 5 * REWARD_FACTOR
PVELOCITY_WEIGHT = 1 * REWARD_FACTOR

AVERAGE_THRESHOLD = 200.0 #the number of steps we are trying to achieve

class StepStats:
    '''
    A class that keeps track of statistics on how the simulation goes.
    '''
    def __init__(self):
        self.steps = []

    def addSteps(self,score):
        self.steps.append(score)
    
    def reportStats(self):
        minScore = min(self.steps)
        maxScore = max(self.steps)
        avgScore = sum(self.steps)/len(self.steps)
        return [ minScore, avgScore, maxScore ]

class Q_Learn_Agent:
    '''
    Our Agent that uses Q-Learning to create a "smart" agent.

    '''
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.observation_space = observation_space

        #create the space to store "training" steps
        self.memory = deque(maxlen=MEMORY_SIZE)

        #create the Neural Newtwork model - adjust as desired
        self.model = Sequential()
        self.model.add(Dense(4, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(16, activation="tanh"))
        self.model.add(Dense(24, activation="linear"))
        self.model.add(Dense(self.action_space, activation="softmax"))
        self.model.compile(loss="mse", optimizer=Adam(lr=ALPHA))
        """self.model = Sequential()
        self.model.add(Dense(4, input_shape=(self.observation_space,), activation="linear"))
        self.model.add(Dense(24, activation="tanh"))
        self.model.add(Dense(24, activation="sigmoid"))
        self.model.add(Dense(50, activation="relu"))
        self.model.add(Dense(50, activation="selu"))
        self.model.add(Dense(self.action_space, activation="relu"))
        self.model.compile(loss="mse", optimizer=SGD(lr=LEARNING_RATE))"""
    
    def remember(self, state, action, reward, next_state, done):
        #save information for learning
        self.memory.append((state, action, reward, next_state, done))

    def save(self, stepsie):
        #save the model
        #fn = input("Enter model file name (include the .h5 file type). Enter nothing to skip saving.:")
        if False != "sdfsjkdnsjkdnl":
            self.model.save("model" + str(stepsie) + ".h5")
        #for future loading, use:        self.model = load_model('CartPoleModel.h5')

    def act(self, state):
        #predict an action during training

        #explore
        #if np.random.rand() < self.exploration_rate:
        #    return random.randrange(self.action_space)
        #exploit
        q_values = self.model.predict(state)
        print(q_values)
        randVal = random.random()
        if q_values[0][0] > randVal: return 0
        else: return 1

    def modelAct(self, state):
        #predict an action (not to be used during training)
        q_values = self.model.predict(state)
        randVal = random.random()
        if q_values[0][0] > randVal: return 0
        else: return 1

    def learn(self):
        #the Q-learning algorithm used during training

        #don't do anything until you have enough data
        if len(self.memory) < BATCH_SIZE: 
            batch = self.memory
        else:
            batch = random.sample(self.memory, BATCH_SIZE)
        #pick random data from all saved data to use to improve the model
        
        states_batch = []
        q_values_batch = []
    
        for state, action, reward, state_next, terminal in batch:
            q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0])) if not terminal else reward
            q_values = self.model.predict(state)
            q_values_old = q_values[0][action]
            q_values[0][action] = (1-ALPHA)*q_values_old + ALPHA*q_update 
            states_batch.append(state[0])
            q_values_batch.append(q_values[0])
                
        states_batch = np.array(states_batch).reshape(-1,self.observation_space)
        q_values_batch = np.array(q_values_batch).reshape(-1,self.action_space)
    
    
        self.model.fit(states_batch, q_values_batch, epochs=200, verbose=0)
    
        # reduce explorations
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def cartpole():
    #setup the simulation
    env = gym.make(ENV_NAME)
    env._max_episode_steps = 2000 #not that we will need this much time...
    #how we will keep track of how well we are doing (not really needed, but good to see)
    step_logger = StepStats()
    #get size of observation (states) and action spaces
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    #initialize the ANN-based Q-learning solver object
    qLearnAgent = Q_Learn_Agent(observation_space, action_space)
    #set up the training loop
    run = 0
    #num_trainings = int(input("Enter number of training cycles:"))
    num_trainings = 20
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
            action = qLearnAgent.act(state)
            #do the action and get the results
            #state_next, reward, sim_done, _ = env.step(action)
            state_next, stepsies, sim_done, _ = env.step(action)
            reward = min((ANGLE_WEIGHT/abs(state_next[2])) + min(PVELOCITY_WEIGHT/abs(state_next[3]), 10) + min(VELOCITY_WEIGHT/abs(state_next[1]), 10) + POSITION_WEIGHT/abs(state_next[0]), 5)
            print("Reward:", reward)
            #if sim terminates, make reward negative (bad thing happened)
            reward = reward if not sim_done else -reward
            #have the model remember what happend with the action
            state_next = np.reshape(state_next, [1, observation_space])
            qLearnAgent.remember(state, action, reward, state_next, sim_done)
            state = state_next
            if sim_done:
                #print a bunch of statistical information about this training run
                print("Run: {:5} ,  exploration: {:8.3f} , score: {:5}" .format(run,qLearnAgent.exploration_rate,step))
                step_logger.addSteps(step)
                scrs = step_logger.reportStats()
                print("           Min: {:5}  Mean: {:8.3f}  Max: {:5}".format(scrs[0],scrs[1],scrs[2]))
                break
            #update the learning of the model
            qLearnAgent.learn()

        #check to see if we have trained well enough.
        scrs = step_logger.reportStats()
        if scrs[1] > AVERAGE_THRESHOLD:
            break
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Time of termination =", current_time)
    #save (optionally) our trained model
    #_ = input("Training completed. Press enter to continue.")

    #use our model to show how well it does (or doesn't)
    state = env.reset()
    #env.render()
    #time.sleep(3)
    sim_done = False
    step = 0
    while not sim_done:
        #env.render()
        #time.sleep(0.01)
        step += 1
        state = np.reshape(state, [1, observation_space])
        action = qLearnAgent.modelAct(state)
        state, reward, sim_done, _ = env.step(action)
        if sim_done:
            print("Done after ", step, " steps.")
    env.close()
    qLearnAgent.save(scrs[1])

if __name__ == "__main__":
    for cycley in range(20):
        cartpole()
