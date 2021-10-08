# Adversary-Agent-CartPole

## Installation: pip install ...
1. mac osX: anaconda, python3.6, tensorflow2.6, keras, gym, keras-rl, pyglet. 
2. Win10: anaconda, python3.6, tensorflow-estimator==2.1.0, tensorflow=2.1.0, keras, gym, keras-rl, pyglet. In addition, all imports shall be directly from keras ranther than tensorflow. Hence, please check the downloaded code and modify the import section in case there is any warnings.  
3. Download the env files and copy to Anaconda3/envs/You_Env_Name/site-packages/gym/envs/classical_control.
4. Modify __init__.py therein by following the __init__.py here. 
5. Modify __init__.py under Anaconda3/envs/You_Env_Name/site-packages/gym/envs/ by following the __init__(2).py here.
6. Test the set-up and environment by running: python CartPole0.py. If everything is OK, a single cart-pole case will be trained. 

# Possible issues:
1. A warning/error for time_limit.py could appear, which could be fixed by annotating line 21 of that file, i.e. info["TimeLimit.truncated"] = not done
2. ../site-packages/rl/callbacks.py, line 241, modify to the following: 
    def on_step_begin(self, step, logs):
        """ Print metrics if interval is over """
        if self.step % self.interval == 0:
            if len(self.episode_rewards) > 0:
                print('{} episodes - episode_reward: {:.3f} [{:.3f}, {:.3f}]'.format(len(self.episode_rewards), np.mean(self.episode_rewards), np.min(self.episode_rewards), np.max(self.episode_rewards)))
                print('')
            self.reset()
            print('Interval {} ({} steps performed)'.format(self.step // self.interval + 1, self.step))
3. The above code can be found in this folder, download and copy to your own folder to replace the callbacks.py. 






## Files: 
1. CartPole0.py, the RL code for the classical cart-pole system. 
2. CartPole1.py, fit two competing agents simultaneously in the opponent cart-pole system.
3. xun_cartpole_0.py, the competitive environment with 2 identical cart-pole systems attached to a spring. 
