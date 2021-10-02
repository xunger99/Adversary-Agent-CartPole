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


class SARSAAgent_Xun(SARSAAgent):

    """The test function from core.py is overridden here to provide action:
        input: observation;
        output: actions
    """
    
    def forward(self, observation):
        # Select an action.
        # pdb.set_trace()
        #print('aaa')
        q_values = self.compute_q_values([observation])
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)
        # Book-keeping. Important! xun
        self.observations.append(observation)      # Disable the two lines will fail the fit. 
        self.actions.append(action)
        return action
    
    
   


        
    # Extensively simplified fit
    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')

        self.training = True
        callbacks = [] if not callbacks else callbacks[:]
        callbacks += [TrainIntervalLogger(interval=log_interval)]

        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        
        #pdb.set_trace()
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
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
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    #assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)   # cannot remove. xun
                action = self.forward(observation)      # for each agent... xun    

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
                metrics = self.backward(reward, terminal=done)
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
                self.step += 1

                if done:
                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)
                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
                    
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history
    
        
        
    """   
    # An extensive simplificatiof of test function from core.Agent    
    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):          
        self.training = False
        self.step = 0

        #callbacks = [] if not callbacks else callbacks[:]
        for episode in range(nb_episodes):
            episode_reward = 0.
            episode_step = 0

            observation = deepcopy(env.reset())

            done = False
            while not done:
                #callbacks.on_step_begin(episode_step)
                action = self.forward(observation)                          ### !!! 
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):

                    observation, r, d, info = env.step(action)      
                    observation = deepcopy(observation)    
                    env.render()
                    
                    reward += r

                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                #self.backward(reward, terminal=done)
                episode_reward += reward
        
        return history
        """


