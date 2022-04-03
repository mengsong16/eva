import torch
import numpy as np
import os
import gym
from gym import spaces
import random
from gym.envs.registration import EnvSpec
from gym.envs.registration import register

# Original gym env
class BitFlippingGymEnv(gym.Env):

    # https://github.com/openai/gym/blob/master/gym/core.py
    # Another way to set metadata is to set self.metadata in any of the class function
    metadata = {'render.modes': ['human']}
    spec = EnvSpec(id='bitflip-v0')

    # environment_dimension: the length of bit sequence and the max length of episode
    def __init__(self, environment_dimension=20, random_goal=False, random_start=False):
        self.action_space = spaces.Discrete(environment_dimension)
        # if deterministic:
        #     self.observation_space = spaces.Box(0, 1, shape=(environment_dimension,), dtype=np.float32)
        # else:    
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(0, 1, shape=(environment_dimension,), dtype=np.float32),
            achieved_goal=spaces.Box(0, 1, shape=(environment_dimension,), dtype=np.float32),
            observation=spaces.Box(0, 1, shape=(environment_dimension,), dtype=np.float32),
        ))

        self.environment_dimension = environment_dimension
        self.reward_for_achieving_goal = 0.0
        self.step_reward_for_not_achieving_goal = -1.0

        #self.deterministic = deterministic
        self.random_start = random_start
        self.random_goal = random_goal

        self.spec.max_episode_steps = environment_dimension

    def reset(self):
        # randomize intial state and goal state
        # if not self.deterministic:
        #     self.desired_goal = self.randomly_pick_state_or_goal()
        #     self.state = self.randomly_pick_state_or_goal()
        # # if deterministic, initial state is all 1, goal state is all 0   
        # else:
        #     self.desired_goal = [0 for _ in range(self.environment_dimension)]
        #     self.state = [1 for _ in range(self.environment_dimension)]
        if self.random_start:
            self.state = self.randomly_pick_state_or_goal()
        else:
            self.state = [1 for _ in range(self.environment_dimension)]

        if self.random_goal:
            self.desired_goal = self.randomly_pick_state_or_goal() 
        else:
            self.desired_goal = [0 for _ in range(self.environment_dimension)]           
        
        self.achieved_goal = self.state
        self.step_count = 0

        # if self.deterministic:
        #     return np.array(self.state).copy()
        # else:    
        return {"observation": np.array(self.state).copy(), "desired_goal": np.array(self.desired_goal).copy(),\
        "achieved_goal": np.array(self.achieved_goal).copy()}

    def randomly_pick_state_or_goal(self):
        return [random.randint(0, 1) for _ in range(self.environment_dimension)]

    def step(self, action):
        """Conducts the discrete action chosen and updated next_state, reward and done"""
        if type(action) is np.ndarray:
            action = action[0]

        if action < 0 or action >= self.environment_dimension:
            # check argument is in range
            print("Invalid action! Must be integer ranging from 0 to %d"%(self.environment_dimension-1))
            return    
        
        # flip the bit with index action
        self.state[action] = (self.state[action] + 1) % 2
        self.achieved_goal = self.state

        # step count
        self.step_count += 1

        if self._is_success(np.array(self.achieved_goal), np.array(self.desired_goal)):
            self.reward = self.reward_for_achieving_goal
            done = True
            success = True
        else:
            self.reward = self.step_reward_for_not_achieving_goal
            if self.step_count >= self.environment_dimension:
                done = True
            else:
                done = False

            success = False    

        #self.reward = random.randint(1, 5)       
        # if self.deterministic:
        #     return np.array(self.state).copy(), float(self.reward), done, {'is_success': success}
        # else:    
        return {"observation": np.array(self.state).copy(),\
        "desired_goal": np.array(self.desired_goal).copy(), "achieved_goal": np.array(self.achieved_goal).copy()}, float(self.reward), done,\
        {'is_success': success}

    # Must be of this exact interface to fit with the open AI gym specifications
    # achieved_goal and desired_goal are numpy arrays 
    def _is_success(self, achieved_goal, desired_goal):
        if (achieved_goal == desired_goal).all():
            return True
        else:
            return False

    # Must be of this exact interface to fit with the open AI gym specifications
    # achieved_goal and desired_goal are numpy arrays      
    def compute_reward(self, achieved_goal, desired_goal, info):
        if (achieved_goal == desired_goal).all():
            reward = self.reward_for_achieving_goal
        else:
            reward = self.step_reward_for_not_achieving_goal
        return reward 

    def seed(self, seed=None):
        random.seed(seed)
    
    def render(self, mode='human'):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        Returns:
            str: current state and goal state of environment.

        """
        return f'State: {np.array(self.state)}, Goal: {np.array(self.desired_goal)}'              

# register envs to gym 
register(
    id='bitflip-v0',
    entry_point='eva.envs.bitflip:BitFlippingGymEnv',
)    


