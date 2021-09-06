# Xun's test code
# Hint: Use the env with Tensorflow >2.0
# I use this code to test the new env with 2 cart-pole systems. 
#               by Professor Xun Huang @ Peking University, 2021. 


import gym

#from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common import make_vec_env
#from stable_baselines import A2C
import pdb
from copy import deepcopy

# Define learning policy
#import rl 
from agent import SARSAAgent_Xun
#from rl.agents import SARSAAgent, DQNAgent, DDPGAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from keras import Input
import random
import numpy as np
from tensorflow import keras
#import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam




"""
# Parallel environments
env = make_vec_env('XunCartPole-v0', n_envs=1)

pdb.set_trace()
model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
"""






env = gym.make('CartPole-v1') #CartPole-v1') #XunCartPole-v0')    

# print states
states = env.observation_space.shape[0]
print('States', states)

# print actions
actions = env.action_space.n
print('Actions', actions)

# Random control performance? 
episodes = 0
for episode in range(1,episodes+1):
    # At each begining reset the game 
    state = env.reset()
    # set done to False
    done = False
    # set score to 0
    score = 0
    # while the game is not finished
    while not done:  # When done= True, the game is lost  
        # visualize each step
        env.render()
        # choose a random action
        action = random.choice([0,1])
        # execute the action
        n_state, reward, done, info = env.step(action)
        # keep track of rewards
        score+=reward
    print('episode {} score {}'.format(episode, score))


# Define a smart agent (a very small and simple network)
def agent_NN(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


# Define the model for one cart-pole system
obs_shape = 4       # observations: x, x_dot, theta, theta_dot
ctr_shape = 2       # control: +force, -force
model1 = agent_NN(obs_shape, ctr_shape)
model1.summary()







#################################################################################
# Define agent type 
# 10000 can be changed in the core.py, log_interval
agent = 1   # only 1 is working for now.  0: dqn (not working anymore because of incompatible version), 1: sarsa, 2: ddpg
if agent==0:
    # all parameters are from https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_cartpole.py    
    memory = SequentialMemory(limit=50000, window_length=1)  
    policy = BoltzmannQPolicy()
    pdb.set_trace()
    agent = DQNAgent(model = model1, memory = memory, policy = policy, nb_actions = env.action_space.n, nb_steps_warmup=10)

elif agent==1:
    policy = EpsGreedyQPolicy()     # from the original tutorial code
    agent1 = SARSAAgent_Xun(model = model1, policy = policy, nb_actions = env.action_space.n)
    agent2 = SARSAAgent_Xun(model = model1, policy = policy, nb_actions = env.action_space.n)
elif agent==2:
    pass
    
else:
    print('incorrect agent?!?')
    
# Agent and compile and training
agent1.compile('adam', metrics = ['mse'])
agent2.compile('adam', metrics = ['mse'])

# only for tests
agent1.load_weights('sarsa_weights.h5f')
agent2.load_weights('sarsa_weights.h5f')
env=gym.make('XunCartPole-v0')
env.reset()
pdb.set_trace()
#env.step(action)
#_ = agent.test(env, nb_episodes = 2, visualize= True)

episodes = 20
for episode in range(1,episodes+1):
    #pdb.set_trace()    
    # At each begining reset the game 
    obs = deepcopy(env.reset()) 
    # set done to False
    done = False
    # set score to 0
    score = 0
    # while the game is not finished
    #pdb.set_trace()
    while not done:  # When done= True, the game is lost  
        obs1= obs[0:4]  # The first 4 for sys 1
        obs2= obs[4:]   # The second 4 for sys 2

        # choose actions from the agent
        action1 = agent1.forward(obs1)
        action2 = agent2.forward(obs2)
        actions = [action1, action2] 
        #actions = action1
        # execute the action
        #
        obs, reward, done, info = env.step(actions)
        # keep track of rewards
        env.render()
        score+=reward
        
        #print('actions {}'.format(actions))
    #pdb.set_trace()    
    print('episode {} score {}'.format(episode, score))

env.close()





pdb.set_trace()


agent.fit(env, nb_steps = 30001, visualize = False, verbose = 1)
#model1.summary()

# Then, test the trained model
scores = agent.test(env, nb_episodes = 10, visualize=True)
env.env.close()

# Next, save the model 
agent.save_weights('sarsa_weights.h5f', overwrite=True)


pdb.set_trace()
# load the weights
# agent.load_weights('sarsa_weights.h5f')
env = gym.make('CartPole-v1')           #env=gym.make('XunCartPole-v0')
_ = agent.test(env, nb_episodes = 10, visualize= True)
#pdb.set_trace()
env.close()