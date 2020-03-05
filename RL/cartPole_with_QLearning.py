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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

ENV_NAME = "CartPole-v1"

GAMMA = 0.85 #discount factor for future value
LEARNING_RATE = 0.10 #how much the new information can change the existing information
ALPHA = LEARNING_RATE

MEMORY_SIZE = 100_000
BATCH_SIZE = 100

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

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
        self.model.add(Dense(4, input_shape=(self.observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="tanh"))
        self.model.add(Dense(24, activation="tanh"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        #save information for learning
        self.memory.append((state, action, reward, next_state, done))

    def save(self):
        #save the model
        fn = input("Enter model file name (include the .h5 file type). Enter nothing to skip saving.:")
        if fn != "":
            self.model.save(fn)
        #for future loading, use:        self.model = load_model('CartPoleModel.h5')

    def act(self, state):
        #predict an action during training

        #explore
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        #exploit
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def modelAct(self, state):
        #predict an action (not to be used during training)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def learn(self):
        #the Q-learning algorithm used during training

        #don't do anything until you have enough data
        if len(self.memory) < BATCH_SIZE: 
            return   
        #pick random data from all saved data to use to improve the model
        batch = random.sample(self.memory, BATCH_SIZE)
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
    num_trainings = int(input("Enter number of training cycles:"))
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
            state_next, _, sim_done, _ = env.step(action)
            reward = (1/abs(state[0][2]))*500
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

    #save (optionally) our trained model
    _ = input("Training completed. Press enter to continue.")
    qLearnAgent.save()

    #use our model to show how well it does (or doesn't)
    state = env.reset()
    env.render()
    time.sleep(3)
    sim_done = False
    step = 0
    while not sim_done:
        env.render()
        time.sleep(0.01)
        step += 1
        state = np.reshape(state, [1, observation_space])
        action = qLearnAgent.modelAct(state)
        state, reward, sim_done, _ = env.step(action)
        if sim_done:
            print("Done after ", step, " steps.")
    env.close()

if __name__ == "__main__":
    cartpole()
