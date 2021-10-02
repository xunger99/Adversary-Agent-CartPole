"""
Two identical cart-pole systems atteched to a spring. 

Modified from the classical CartPoleEnv. Xun, Aug 2021. 

The modifications:
1. Include self.x_specific, which is the target x position of the card-pole sys.
2. Include the 2nd cart-pole system and increase the state to: 
     States: x, x_dot, theta, theta_dot, x2, x2_dot, theta2, theta2_dot
3. Include the spring. 
"""

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import pdb


class XunCartPole2Env(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
        States 4-7 are the same as 0-3. Xun

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        
        self.total_steps = 500  # set to 500, please check __init__.py file 
        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates. (50Hz? Xun) 
        self.kinematics_integrator = "euler"
        

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # Modified by Xun
        self.x_specific = 0         # The pole is supposed to maintain at here
        self.x_threshold = 3      
        self.x_threshold2 = 3   #0.5
        
        # The initial position of the two system
        #self.x1init = 50
        #self.x2init = -50

        print("Xun's test env 2021(V1)")
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        
        
        # modified by xun 
        self.screen_width = 900
        self.world_width  = self.x_threshold * 2 
        self.scale = self.screen_width / self.world_width
        
        
        
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,                
            ],
            dtype=np.float32,
        )
        
        
        self.action_space = spaces.Discrete(2)      # 
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        #pdb.set_trace()
        
        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        ###########################################################
        # Modified by Xun
        ###########################################################
        self.spring_len = 1         # The length of the spring, where the spring force = 0 
        self.spring_k = 20           # kx=sprince force

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):

        force1 = self.force_mag if action[0] == 1 else -self.force_mag   # modified by xun
        force2 = self.force_mag if action[1] == 1 else -self.force_mag   # modified by xun

        state = self.state
        x, x_dot, theta, theta_dot, x2, x2_dot, theta2, theta2_dot = state        
        ###########################################################
        # Dynamics for the spring
        ###########################################################  
        spring_force = self.spring_k * (x - x2 - self.spring_len)     
        force1 = force1 - spring_force      # Include the spring force. 
        force2 = force2 + spring_force      
        if x - x2 <0:
            #pdb.set_trace()
            print("Something must be incorrect here, x: {}, x2:{}, xun".format(x, x2))
            
        
        ###########################################################
        # Dynamics for sys 1
        ###########################################################
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force1 + self.polemass_length * theta_dot *theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot



        #################################################################
        # The 2nd cart-pole system 
        # modified by xun, Sep 2021, to include a new cart-pole system
        #################################################################
        costheta2 = math.cos(theta2)
        sintheta2 = math.sin(theta2)
        temp2 = (
            force2 + self.polemass_length * theta2_dot *theta2_dot * sintheta2
        ) / self.total_mass
        thetaacc2 = (self.gravity * sintheta2 - costheta2 * temp2) / (
            self.length * (4.0/3.0 - self.masspole * costheta2 * costheta2 / self.total_mass)
        )
        # The original code is incorrect! The bug was found by Professor Xun Huang on Sep 3, 2021 
        # No no no...
        xacc2 = temp2 - self.polemass_length * thetaacc2 * costheta2 / self.total_mass

        if self.kinematics_integrator == "euler":
            x2 = x2 + self.tau * x2_dot
            x2_dot = x2_dot + self.tau * xacc2
            theta2 = theta2 + self.tau * theta2_dot
            theta2_dot = theta2_dot + self.tau * thetaacc2
        else:  # semi-implicit euler
            x2_dot = x2_dot + self.tau * xacc2
            x2 = x2 + self.tau * x2_dot
            theta2_dot = theta2_dot + self.tau * thetaacc2
            theta2 = theta2 + self.tau * theta2_dot
        #################################################################
        #################################################################
        #################################################################
        
                        
        self.state = (x, x_dot, theta, theta_dot, x2, x2_dot, theta2, theta2_dot)
        #pdb.set_trace()
        done1 = bool(
            x < -self.x_threshold2 + self.x_specific 
            or x > self.x_threshold2 + self.x_specific 
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )  
        done2 = bool(
            x2 < -self.x_threshold2 + self.x_specific 
            or x2 > self.x_threshold2 + self.x_specific 
            or theta2 < -self.theta_threshold_radians
            or theta2 > self.theta_threshold_radians
        )          
        done = (done1 or done2)
        winner = "2" if done1 == True else "1"      # Who is the winner. Xun
 
        if not done:
            reward = [1.0,1.0]
        elif self.steps_beyond_done is None:
            #pdb.set_trace()
            # Pole just fell!
            self.steps_beyond_done = 0
            #reward = 1.0
            # To encourage the one still not fall down. But will not encourage cooperation. Xun
            if done1 is True and done2 is True:         
                reward = [1.0,1.0]      # Both fell, impose small reward first, then impose big penalty to who is actually fell.   
                if theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians:
                    reward[0] = -100.0          # CP1 fell
                if theta2 < -self.theta_threshold_radians or theta2 > self.theta_threshold_radians:
                    reward[1] = -100.0          # CP2 fell
            elif done1 is True:
                reward = [-100.0,500.0]
            else:
                reward = [500.0,-100.0]
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            #reward = 0.0
            if done1 is True and done2 is True:    
                reward = [1.0,1.0]  # Both fell, impose small reward first, then impose big penalty to who is actually fell.  
                if theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians:
                    reward[0] = -100.0          # CP1 fell
                if theta2 < -self.theta_threshold_radians or theta2 > self.theta_threshold_radians:
                    reward[1] = -100.0          
            elif done1 is True:
                reward = [-100.0,500.0]
            else:
                reward = [500.0,-100.0]                     
                            
        return np.array(self.state), reward, done, winner



    def reset(self):
        #pdb.set_trace()
        #self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(8,))
        
        #################################################################
        # Modified by xun to include the initial lengh of the spring below.
        # For simplicity, here I neglect the width of the two carts, and 
        # the initial positions are fixed to ensure intially zero spring force. 
        ##################################################################
        self.state[0] =  + self.spring_len * 0.5 
        self.state[4] =  - self.spring_len * 0.5 
        
        #pdb.set_trace()
        
        self.steps_beyond_done = None
        return np.array(self.state)



    def render(self, mode="human"):
        screen_width = self.screen_width    #800
        screen_height = 400 #600 #400

        #pdb.set_trace()
        world_width = self.world_width  #self.x_threshold * 2
        scale = self.scale
        #pdb.set_trace()
        
        #spring_scale = self.spring_len * scale
                 
        carty = 100  # TOP OF CART
        cart2y= 100  # Top of cart 2, included by Xun
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        #pdb.set_trace()
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])    
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            
            
            
            #################################################################
            # The 2nd cart-pole system 
            # modified by xun, Sep 2021, to include a new cart-pole system
            #################################################################
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)]) 
            self.carttrans2 = rendering.Transform()
            cart2.add_attr(self.carttrans2)
            self.viewer.add_geom(cart2)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2.set_color(0.9, 0.3, 0.3)
            self.poletrans2 = rendering.Transform(translation=(0, axleoffset))
            pole2.add_attr(self.poletrans2)
            pole2.add_attr(self.carttrans2)
            self.viewer.add_geom(pole2)            
            #################################################################
            #################################################################
            
            """How to do this? 
            #################################################################
            # The spring 
            # modified by xun, Sep 2021
            #################################################################    
            spring = rendering.Line((0, carty+20), (100, carty+20))
            spring.set_color(0.1, 0.2, 0.3)
            spring.add_attr(self.) 
            self.viewer.add_geom(self.spring)
            #################################################################
            #################################################################            
            """        
            
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)

            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)



            self._pole_geom = pole
            self._pole_geom2 = pole2

            
            
        if self.state is None:
            return None

            
        #################################################################
        # Modified by xun
        #################################################################
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )        
        pole2 = self._pole_geom2
        pole2.v = [(l, b), (l, t), (r, t), (r, b)]
        #################################################################
        
                
        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]
        

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        
        #################################################################
        # Modified by xun
        #################################################################
        #pdb.set_trace()
        cart2x = x[4] * scale + screen_width / 2.0 
        self.carttrans2.set_translation(cart2x, cart2y)
        self.poletrans2.set_rotation(-x[6])
        

        return self.viewer.render(return_rgb_array=mode == "rgb_array")



    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
