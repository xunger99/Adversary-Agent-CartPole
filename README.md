# Adversary-Agent-CartPole

## Installation: pip install ...
1. mac osX: anaconda, python3.6, tensorflow2.6, keras, gym, keras-rl, pyglet. 
2. Win10: anaconda, python3.6, tensorflow-estimator==2.1.0, tensorflow=2.1.0, keras, gym, keras-rl, pyglet. In addition, all imports shall be directly from keras ranther than tensorflow. Hence, please check the downloaded code and modify the import section in case there is any warnings.  
3. Download the env files and copy to Anaconda3/envs/You_Env_Name/site-packages/gym/envs/classical_control.
4. Modify __init__.py therein by following the __init__.py here. 
5. Modify __init__.py under Anaconda3/envs/You_Env_Name/site-packages/gym/envs/ by following the __init__(2).py here.
6. Test the set-up and environment by running: python CartPole0.py. If everything is OK, a single cart-pole case will be trained. 

## Files: 
1. CartPole0.py, the RL code for the classical cart-pole system. 
2. xun_cartpole_0.py, the competitive environment with 2 identical cart-pole systems attached to a spring. 
