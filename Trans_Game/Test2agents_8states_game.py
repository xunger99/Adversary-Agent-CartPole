#
#   Test two agents on the 2CP competitive env. 
#               by Professor Xun Huang @ Peking University, 2021. 
#
import collections
import matplotlib.pyplot as plt
from scipy import interpolate
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

from keras.layers import Dropout
from keras import Input
import random
import numpy as np
#from tensorflow import keras

import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
#from tensorflow.keras.optimizers import Adam

# modify the name to XunCartPole-v0, XunCartPole-v1, XunCartPole-v2, to XunCartPole-v4
# XunCartPole-v1, reward 100
# XunCartPole-v3, reward 1
# XunCartPole-v4, reward 100 vs 1
env = gym.make('XunCartPole-v2')

# Define a smart agent (a very small and simple network)
def agent_NN(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def agent_NN2(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(72, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(72, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(72, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
    
# Define the model for one cart-pole system
obs_shape = 4           # observations: x, x_dot, theta, theta_dot
obs_full_shape = 8      # observations: x, x_dot, theta, theta_dot; x, x_dot, theta, theta_dot
ctr_shape = 2           # control: +force, -force

#model1 =  agent_NN(obs_shape, ctr_shape)
#model2 =  agent_NN(obs_shape, ctr_shape)
model1 =   agent_NN(obs_shape, ctr_shape)
model2 =  agent_NN2(obs_full_shape, ctr_shape)
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


agent1.load_weights('CL3Ag3_double_SARSA_agent1.h5f')
agent2.load_weights('CL3Ag3_double_SARSA_agent2.h5f')



# Test 100 runs. 
episodes = 30
agent1.reset_states()
agent2.reset_states()
win1 = 0
win2 = 0  
score_all = 0
for episode in range(1,episodes+1):
    #pdb.set_trace()    
    # At each begining reset the game 
    obs = deepcopy(env.reset()) 
    # set done to False
    done = False
    # set score to 0
    score = 0
    score1 = 0  # for agent1's reward 
    score2 = 0  # for agent2's reward 
    
    # while the game is not finished
    #pdb.set_trace()
    
    act_his = collections.deque(maxlen=500)
    obs_his = collections.deque(maxlen=500)
    spring_k = 15       # Make sure this value is consistent with the value inside the environment!
    spring_len = 1
        
  
    while not done:     # When done= True, the game is lost  
        obs1= obs[0:4]  # The first 4 for sys 1
        obs2= obs   #[4:]   # The second 4 for sys 2

        # choose actions from the agent
        action1 = agent1.forward(obs1)
        action2 = agent2.forward(obs2)
        two_actions = [action1, action2] 
        #actions = action1
        # execute the action
        obs, reward, done, info = env.step(two_actions)
        # keep track of rewards
        env.render()
        #if done:
        #    pdb.set_trace()
        
        try:
            score1+=reward[0]
            score2+=reward[1]
        except:
            pdb.set_trace()
        
        # Record the history of observation and action.      
        obs_his.append(obs)       # Obs: x1, v1, theta1, dot(theta1), x2, v2, theta2, dot(theta2)   
        act_his.append(two_actions)   # Two actions for two agents.  

    #pdb.set_trace()    
    if reward[0] < reward[1]:
        print("Sys 2 wins, reward{}".format(reward))
        win2 += 1
        score = score2
    elif reward[1] < reward[0]:
        print("Sys 1 wins, reward{}".format(reward))
        win1 += 1
        score = score1
    else: 
        print("Both fell or time is up")  
        score = score1      
    print('episode {} score {}'.format(episode, score))
    score_all += score
    
    
    """
    # Enable the following section to plot the analysis drawings. 
    #############################################################
    # Xun's code. Here I draw the actions vs observations to show the mech underneath 
    # the agent. 
    #############################################################
    #pdb.set_trace()
    #Fs =  spring_k *(obs(4)-obs(0))
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(211)
    ax2 = ax1.twinx()
    dt=0.02 # check the env, where self.tau=0.02, make sure the values are consistent to the env. xun
    display_len = len(act_his) 
    x1=[i*dt for i in range(display_len)]
    
    # Calculate the spring force
    Fs = [obs_his[i][4]*spring_k + spring_len*spring_k - obs_his[i][0]*spring_k for i in range(display_len)]
    
    # Show for agent1. xun 2021
    y1=[act_his[i][0]*20-10 for i in range(display_len)]       # *20-10: then act:1 -> 10N,xun 2021 act:0->10N
    the1=[obs_his[i][2]*180/np.pi for i in range(display_len)]
    the1dot=[obs_his[i][3]*180/np.pi for i in range(display_len)]
    # zeroth interpolation to represent bang-bang control actuation.
    func=interpolate.interp1d(x1, y1, kind='zero')
    xnew = np.arange(0, max(range(display_len))*dt, 0.1*dt)     # interpolate into the same region with 10 more points. xun 2021
    ynew = func(xnew)
    ax1.plot(xnew,ynew,'b-',x1,Fs,'r--')      # Plot the bang-bang force
    #ax2.plot(x1,the1,'o',x1,the2,'-') 
    #ax1.set_xlabel("Time (s)", fontsize=15)
    ax1.set_ylabel(r"Control force (N)", fontsize=15,color='b')
    ax2.plot(x1,the1,'ko')
    ax2.set_ylabel(r"$\theta_1$ (deg)", fontsize=15,color='k')
    ax2.set_ylim([-12,12])


    # Show for agent1. xun 2021
    ax3 = plt.subplot(212)
    ax4 = ax3.twinx()
    #Fs2 = -Fs      # The spring force is negative for sys2
    Fs2 = [obs_his[i][0]*spring_k - obs_his[i][4]*spring_k - spring_len*spring_k for i in range(display_len)]
    the2=[obs_his[i][6]*180/np.pi for i in range(display_len)]
    the2dot=[obs_his[i][7]*180/np.pi for i in range(display_len)]
    y2=[act_his[i][1]*20-10 for i in range(display_len)]  # Action force for system 2
    func2=interpolate.interp1d(x1, y2, kind='zero')
    ynew2 = func2(xnew)
    ax3.plot(xnew,ynew2,'b-',x1,Fs2,'r--')
    ax3.set_ylabel(r"Control force (N)", fontsize=15,color='b')
    ax3.set_xlabel("Time (s)", fontsize=15)
    ax4.plot(x1,the2,'ko') 
    ax4.set_ylabel(r"$\theta_2$ (deg)", fontsize=15,color='k')
    ax4.set_ylim([-12,12])
    #ax1.set_xlim([0,1.0])
    #ax3.set_xlim([0,1.0])
    ax1.set_title('System 1', fontsize=15)
    ax3.set_title('System 2', fontsize=15)
    
        
    # 3D scatter drawing for system 1
#    font = {'family': 'serif','color':  'darkred','weight': 'normal','size': 16,}
    Ft1 = [act_his[i][0]*20-10 + obs_his[i][4]*spring_k + spring_len*spring_k - obs_his[i][0]*spring_k for i in range(display_len)]
    fig2 = plt.figure(figsize=(7,7))
    ax3d = plt.axes(projection='3d') 
    display_len = len(act_his)
    # Actuation force
    Fa1 = [act_his[i][0]*20-10 for i in range(display_len)]
    # Spring force
    Fs1 = [obs_his[i][4]*spring_k + spring_len*spring_k - obs_his[i][0]*spring_k for i in range(display_len)]
    x1=[obs_his[i][0] for i in range(display_len)]
    x1dot=[obs_his[i][1] for i in range(display_len)]  
    ax3d.scatter3D(the1,the1dot,Fs1,c=the1,cmap='Reds')      # Spring force
    ax3d.scatter3D(the1,the1dot,Fa1,c=the1,cmap='Blues')     # Actuation force
    ax3d.set_xlabel(r"$\theta_1$", fontsize=15)
    ax3d.set_ylabel(r"$\dot{\theta}_1$", fontsize=15)
    ax3d.set_zlabel(r"Control force", fontsize=15)
    #ax3d.text(2, 0.65, 1, r"Defensive", fontsize=15)
    #ax3d.text(-5, 0.65, 3, r"Lose", fontsize=15)
    #ax3d.set_xlim([2,-12])
#    ax3d.set_ylim([-30,20])
    ax3d.view_init(18, 65)
    ax3d.set_title('System 1')
    
    
    # 3D scatter drawing for system 2
    Ft2 = [act_his[i][1]*20-10 + obs_his[i][0]*spring_k - obs_his[i][4]*spring_k - spring_len*spring_k for i in range(display_len)] 
    fig3 = plt.figure(figsize=(7,7))
    ax3d2 = plt.axes(projection='3d') 
    display_len = len(act_his)
    # Actuation force for sys 2
    Fa2 = [act_his[i][1]*20-10 for i in range(display_len)]
    # Spring force
    Fs2 = [obs_his[i][0]*spring_k - obs_his[i][4]*spring_k - spring_len*spring_k for i in range(display_len)]    
    ax3d2.scatter3D(the2,the2dot,Fs2,c=the2,cmap='Reds')     # Spring force
    ax3d2.scatter3D(the2,the2dot,Fa2,c=the1,cmap='Blues')    # Actuation force    
    ax3d2.set_xlabel(r"$\theta_2$", fontsize=15)
    ax3d2.set_ylabel(r"$\dot{\theta}_2$", fontsize=15)
    ax3d2.set_zlabel(r"Control force", fontsize=15)
    #ax3d2.set_xlim([-2,8])
#    ax3d2.set_ylim([-20,25])
    ax3d2.view_init(18, 65)
    ax3d2.set_title('System 2')    
        
    # Show the figure
    plt.show()
    """
    

print('Sys1 wins: {} Sys2 wins: {} and the mean score:{}'.format(win1, win2, score_all/episodes))    
env.close()  



