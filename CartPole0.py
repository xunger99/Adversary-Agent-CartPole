# Xun's test code
# Test how to work, and how to produce my own fit function, my own agent, etc., 
# and produces a SARSA agent for single cart-pole system 
#           Tutorial code by Professor Xun Huang @ Peking University, 2021. 
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
from rl.agents import SARSAAgent #, DQNAgent, DDPGAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from keras import Input
import random
import numpy as np
#from tensorflow import keras
#import keras
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
#from tensorflow.keras.optimizers import Adam




env = gym.make('CartPole-v0') #CartPole-v1') #XunCartPole-v0')    
# Random control performance? 



# print states
states = env.observation_space.shape[0]
print('States', states)

# print actions
actions = env.action_space.n
print('Actions', actions)

# Random control performance? 
episodes = 3
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



policy = EpsGreedyQPolicy()     # from the original tutorial code
agent1 = SARSAAgent(model = model1, policy = policy, nb_actions = env.action_space.n)

# Q: new agent with poor fit performance? why???    
# Agent and compile and training
agent1.compile('adam', metrics = ['mse'])
# Use the agent's fit.
#agent1.fit(env, nb_steps = 30000, visualize = False, verbose = 1)


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
# Extensively simplified fit. Working! Xun, Sep 7, 2021. 
def Xun_fit(agent, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
    """Trains the agent on the given environment.
    """
    print("The fit function from Xun 20201")
    agent.training = True
    callbacks = [] if not callbacks else callbacks[:]
    callbacks += [TrainIntervalLogger(interval=log_interval)]

    history = History()
    callbacks += [history]
    callbacks = CallbackList(callbacks)
    
    #pdb.set_trace()    
    callbacks.set_model(agent)
    callbacks._set_env(env)
        
    params = {
        'nb_steps': nb_steps,
    }
    if hasattr(callbacks, 'set_params'):
        callbacks.set_params(params)
    else:
        callbacks._set_params(params)
        
    #self._on_train_begin()
    callbacks.on_train_begin()      # cannot remove. xun

    episode = np.int16(0)
    agent.step = np.int16(0)
    observation = None
    episode_reward = None
    episode_step = None
    did_abort = False
        
    try:
        while agent.step < nb_steps:
            if observation is None:  # start of a new episode            
                callbacks.on_episode_begin(episode)
                episode_step = np.int16(0)
                episode_reward = np.float32(0)

                # Obtain the initial observation by resetting the environment.
                agent.reset_states()
                observation = deepcopy(env.reset())
                #assert observation is not None

            # Run a single step.
            callbacks.on_step_begin(episode_step)   # cannot remove. xun
            action = agent.forward(observation)      # for each agent... xun    

            reward = np.float32(0)
            #accumulated_info = {}
            done = False
                
            for _ in range(action_repetition):
                observation, r, done, info = env.step(action)   # combine and env.step
                observation = deepcopy(observation)             # reward shall be separate
                reward += r                                     
                if done:
                    break
            if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                # Force a terminal state.
                done = True
            metrics = agent.backward(reward, terminal=done)
            episode_reward += reward

            step_logs = {
                'action': action,
                'observation': observation,
                'reward': reward,
                'metrics': metrics,
                'episode': episode,
                'info': {}, # accumulated_info,
            }
            callbacks.on_step_end(episode_step, step_logs)
            episode_step += 1
            agent.step += 1

            if done:
                # This episode is finished, report and reset.
                episode_logs = {
                    'episode_reward': episode_reward,
                    'nb_episode_steps': episode_step,
                    'nb_steps': agent.step,
                }
                callbacks.on_episode_end(episode, episode_logs)
                episode += 1
                observation = None
                episode_step = None
                episode_reward = None
                    
    except KeyboardInterrupt:
        did_abort = True
    callbacks.on_train_end(logs={'did_abort': did_abort})
    agent._on_train_end()
    return history

#pdb.set_trace()
# Test my own fit func. xun, 2021
Xun_fit(agent1, env, nb_steps = 50000, visualize = False, verbose = 1)
env.env.close()

# Next, save the model, load again and test in a new env
agent1.save_weights('1cp_sarsa_weights.h5f', overwrite=True)

#pdb.set_trace()
# load the weights
agent1.load_weights('1cp_sarsa_weights.h5f')
env = gym.make('CartPole-v1')           #env=gym.make('XunCartPole-v0')
_ = agent1.test(env, nb_episodes = 10, visualize= True)
#pdb.set_trace()
env.close()









