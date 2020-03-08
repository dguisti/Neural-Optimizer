import tensorflow as tf
from collections import deque
from data import actListCalls, optListCalls, lossListCalls
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import random

import numpy as np

GAMMA = 0.85 #discount factor for future value
LEARNING_RATE = 0.10 #how much the new information can change the existing information
ALPHA = LEARNING_RATE

MEMORY_SIZE = 100_000
BATCH_SIZE = 100

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
EPOCHS = 10

LEARNING_SIZE = 100000000000000
INDIVIDUAL_LEARNING = False



class Teacher_Agent:
    '''
    Our Agent that uses Q-Learning to create a "smart" agent.

    '''
    def __init__(self):
        """
        Training Parameters (Things that need to be trained)
        Activations - Numpy array of activations for each layer
            Type: Value of 0-len(actListCalls.keys())
            Length: Number of layers
        Optimizer - Integer value
            Type: Int value, 0-len(optListCalls.keys())
        Losses - Integer value
            Type: Int value, 0-len(lossListCalls.keys())

        """
        GAMMA = 0.85 #discount factor for future value
        LEARNING_RATE = 0.10 #how much the new information can change the existing information
        ALPHA = LEARNING_RATE

        MEMORY_SIZE = 100_000
        BATCH_SIZE = 100

        EXPLORATION_MAX = 1.0
        EXPLORATION_MIN = 0.01
        EXPLORATION_DECAY = 0.995

        POSSIBLE_LAYERS = 10

        self.exploration_rate = EXPLORATION_MAX
        self.action_space = 9
        MSE = list(lossListCalls.keys()).index("mse")
        RELU = list(actListCalls.keys()).index("relu")
        ADAM = list(optListCalls.keys()).index("Adam")
        input_size = 5
        initial_layers = 2
        initial_epochs = 5
        initial_blank_layers = [[RELU, 0] for i in range(POSSIBLE_LAYERS)]
        blank_layers = [item for sublist in initial_blank_layers for item in sublist]
        initial_state = [GAMMA, ALPHA, EXPLORATION_MAX, EXPLORATION_MIN, EXPLORATION_DECAY, 5, MSE, ADAM, RELU, input_size, RELU, 24, RELU, 12] + blank_layers
        initial_state_tensor = tf.convert_to_tensor([initial_state])
        self.state = initial_state_tensor
        self.q_values = initial_state
        self.state_next = self.state
        self.observation_space = len(initial_state)
        self.reward = 0

        #create the space to store "training" steps
        self.memory = deque(maxlen=MEMORY_SIZE)

        #create the Neural Newtwork model - adjust as desired
        self.model = Sequential()
        self.model.add(Dense(input_size, input_shape=(self.observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="tanh"))
        self.model.add(Dense(24, activation="tanh"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self):
        #print(self.model.predict(state))
        #save information for learning
        myState = self.state.numpy()
        myNext_state = self.state_next.numpy()
        
        #print(flat_state)
        #self.memory.append((np.asarray(flat_state).T, action, reward, np.asarray(flat_next_state).T))
        self.memory.append((myState, self.reward, myNext_state))

    def learn(self):
        self.state = self.state_next
        if len(self.memory) < LEARNING_SIZE: 
            batch = self.memory   
        #pick random data from all saved data to use to improve the model
        else:
            batch = random.sample(self.memory, LEARNING_SIZE)
        states_batch = []
        q_values_batch = []
    
        #use each set of data to improve the Q values
        for state, reward, state_next in batch:
            q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)))
            self.q_values = self.model.predict(state)
            q_values_old = self.q_values#[0][self.actionList.index(action[0])]
            self.q_values = (1-ALPHA)*q_values_old + ALPHA*q_update #[0][self.actionList.index(action[0])]
            if INDIVIDUAL_LEARNING:
                self.model.fit(state, self.q_values, epochs=EPOCHS, verbose=0)
            else:
                states_batch.append(state[0])
                q_values_batch.append(self.q_values[0])
        #self.q_values = self.q_values[0]
        self.q_values = self.q_values[:4] + [int(y) for y in self.q_values[5:]]
        #self.model.fit(self.state, tf.convert_to_tensor([self.q_values]), epochs=100, verbose=0)
        #self.q_values = [int(val) for val in self.model.predict(self.state)[0]]
        self.state_next = tf.convert_to_tensor([self.q_values])
        if self.q_values[5] == 0:
            print("Network Returned 0 Epochs")
            return
        print(self.q_values)
        print("Gamma:", self.q_values[0])
        print("Alpha:", self.q_values[1])
        print("Exp. Max:", self.q_values[2])
        print("Exp. Min:", self.q_values[3])
        print("Exp. Decay:", self.q_values[4])
        print("Epochs:", self.q_values[5])
        print("Loss:", list(lossListCalls.keys())[self.q_values[6]])
        print("Optimizer:", list(optListCalls.keys())[self.q_values[7]])
        q_val_layers = self.q_values[9:]
        q_layers = []
        i = 0
        for i in range(0, len(q_val_layers), 2):
            q_layers.append((q_val_layers[i], q_val_layers[i+1]))

        for layer, density in q_layers:
            if density == 0: continue
            print(density, "nodes", list(actListCalls.keys())[layer])

teacher = Teacher_Agent()
for i in range(1, 11):
    print("Training stage", i)
    teacher.learn()
    teacher.remember()
    print("------------------------------")