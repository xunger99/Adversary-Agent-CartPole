# Xun's test code, show how to fit two agents simultaneously for a simple env with two isolated cart-pole systems. 
#
# Set-ups:  Model: simple, 4 inputs, 2cp env with ks=20, reward: win+1, lose-100. Fit two agents together. 
#
# Hint: Use the env with Tensorflow >2.0
#
#               by Professor Xun Huang @ Peking University, 2021. 
#

import gym

#from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common import make_vec_env
#from stable_baselines import A2C
import pdb
from copy import deepcopy

# Define learning policy
#import rl 
from Xun_agent import SARSAAgent_Xun
from rl.agents import SARSAAgent, DQNAgent, DDPGAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from keras import Input
import random
import numpy as np
#from tensorflow import keras
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
#from tensorflow.keras.optimizers import Adam


# modify the name to XunCartPole-v0, XunCartPole-v1, XunCartPole-v2, to XunCartPole-v4
# XunCartPole-v0, no spring
# XunCartPole-v1, reward loss: -100, win: 1 
# XunCartPole-v2, 
# XunCartPole-v3, 
# XunCartPole-v4, 
env = gym.make('XunCartPole-v1') 

# print states
states = env.observation_space.shape[0]
print('States', states)

# print actions
actions = env.action_space.n
print('Actions', actions)


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
full_obs_shape = 8 
ctr_shape = 2       # control: +force, -force
#model1 = agent_NN(obs_shape, ctr_shape)
model1 = agent_NN(obs_shape, ctr_shape)
model2 = agent_NN(obs_shape, ctr_shape)
model1.summary()



#################################################################################
# Define agent type 
# 10000 can be changed in the core.py, log_interval
policy = EpsGreedyQPolicy()     # from the original tutorial code
agent1 = SARSAAgent_Xun(model = model1, policy = policy, nb_actions = env.action_space.n)
agent2 = SARSAAgent_Xun(model = model2, policy = policy, nb_actions = env.action_space.n)

    
# Agent and compile and training
agent1.compile('adam', metrics = ['mse'])
agent2.compile('adam', metrics = ['mse'])

env.reset()
#pdb.set_trace()



###############################
##  Fit, prepared for two agents! 
###############################
import collections
import numpy as np
from keras.callbacks import History
from keras.models import Model
from keras.layers import Input, Lambda
import keras.backend as K

from rl.core import Agent
from rl.agents import SARSAAgent
from rl.agents.dqn import mean_q
from rl.util import huber_loss
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import get_object_config
from copy import deepcopy
import pdb
from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)
# Extensively simplified fit for 2 agents
def Xun_fit2(agent1, agent2, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            nb_max_start_steps=0, log_interval=10000,
            nb_max_episode_steps=None):
    """Trains the agent on the given environment.
    """
    print("The fit function from Xun 20201")
    agent1.training = True
    agent2.training = True
    callbacks1 = [] if not callbacks else callbacks[:]
    callbacks1 += [TrainIntervalLogger(interval=log_interval)]
    callbacks2 = [] if not callbacks else callbacks[:]
    callbacks2 += [TrainIntervalLogger(interval=log_interval)]    
    

    history1 = History()
    callbacks1 += [history1]
    callbacks1 = CallbackList(callbacks1)
    history2 = History()
    callbacks2 += [history2]
    callbacks2 = CallbackList(callbacks2)    
    
    
    #pdb.set_trace()             
    callbacks1.set_model(agent1)  
    callbacks1._set_env(env) 
    callbacks2.set_model(agent2)  
    callbacks2._set_env(env)         
        
    params = {'nb_steps': nb_steps,}
    if hasattr(callbacks1, 'set_params'):
        callbacks1.set_params(params)
    else:
        callbacks1._set_params(params)
    if hasattr(callbacks2, 'set_params'):
        callbacks2.set_params(params)
    else:
        callbacks2._set_params(params)        
        
    #self._on_train_begin()
    callbacks1.on_train_begin()      # cannot remove. xun
    callbacks2.on_train_begin()     
    
    
    episode = np.int16(0)
    agent1.step = np.int16(0)
    agent2.step = np.int16(0)
    observation = None
    episode_reward1 = None
    episode_reward2 = None    
    episode_step = None
    did_abort = False
        
    try:
        while agent1.step < nb_steps:
            if observation is None:  # start of a new episode            
                callbacks1.on_episode_begin(episode)
                callbacks2.on_episode_begin(episode)
                
                episode_step = np.int16(0)
                episode_reward1 = np.float32(0)
                episode_reward2 = np.float32(0)
                
                # Obtain the initial observation by resetting the environment.
                agent1.reset_states()
                agent2.reset_states()
                observation = deepcopy(env.reset())
                #assert observation is not None

            # Run a single step.
            callbacks1.on_step_begin(episode_step)   # cannot remove. xun
            callbacks2.on_step_begin(episode_step)
            
            obs1= observation[0:4]
            obs2= observation[4:]
            
            action1 = agent1.forward(obs1)      # for each agent... xun    
            action2 = agent2.forward(obs2)      # for each agent... xun   
            actions = [action1, action2]  
             
            reward1 = np.float32(0)
            reward2 = np.float32(0)
            done = False
            
           
                
            for _ in range(action_repetition):
                observation, r, done, info = env.step(actions)   # combine and env.step
                observation = deepcopy(observation)             # reward shall be separate
                reward1 += r[0]     # for agent 1 and agent 2. xun   
                reward2 += r[1]                                      
                if done:
                    break
            if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                # Force a terminal state.
                done = True

            metrics1 = agent1.backward(reward1, terminal=done)       # The returned metrics are null
            metrics2 = agent2.backward(reward2, terminal=done)
            
            episode_reward1 += reward1              # xun
            episode_reward2 += reward2              # xun            
            
            
            step_logs1 = {
                'action': action1,
                'observation': obs1,
                'reward': reward1,
                'metrics': metrics1,
                'episode': episode,
                'info': {}, # accumulated_info,
            }
            step_logs2 = {
                'action': action2,
                'observation': obs2,
                'reward': reward2,
                'metrics': metrics2,
                'episode': episode,
                'info': {}, # accumulated_info,
            }
            callbacks1.on_step_end(episode_step, step_logs1)
            callbacks2.on_step_end(episode_step, step_logs2)
            
            
            episode_step += 1
            agent1.step += 1
            agent2.step += 1

            if done:
                # This episode is finished, report and reset.
                episode_logs1 = {
                    'episode_reward': episode_reward1,
                    'nb_episode_steps': episode_step,
                    'nb_steps': agent1.step,
                }
                episode_logs2 = {
                    'episode_reward': episode_reward2,
                    'nb_episode_steps': episode_step,
                    'nb_steps': agent2.step,
                }                
                
                callbacks1.on_episode_end(episode, episode_logs1)
                callbacks2.on_episode_end(episode, episode_logs2)
                episode += 1
                observation = None
                episode_step = None
                episode_reward = None
                    
    except KeyboardInterrupt:
        did_abort = True
    
    callbacks1.on_train_end(logs={'did_abort': did_abort})
    callbacks2.on_train_end(logs={'did_abort': did_abort})
    
    agent1._on_train_end()
    agent2._on_train_end()
    return history1, history2
    

agent1.load_weights('1CP_SARSA_weights.h5f') 
agent2.load_weights('1CP_SARSA_weights.h5f') 
# Use my own fit func to fit two agents together. Xun, Sep 2021
Xun_fit2(agent1, agent2, env, nb_steps = 80000, verbose = 1)
env.env.close()
#pdb.set_trace()

agent1.save_weights('2CP_SARSA_agent_4states_1.h5f', overwrite=True)
agent2.save_weights('2CP_SARSA_agent_4states_2.h5f', overwrite=True)
env.close()
    


