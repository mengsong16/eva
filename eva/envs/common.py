import numpy as np
import random
import gym
from gym.spaces import Dict, Box, Discrete
import torch

from eva.envs.gcsl_envs.room_env import PointmassGoalEnv, PointmassRoomsEnv
from eva.envs.gcsl_envs.sawyer_push import SawyerPushGoalEnv
from eva.envs.gcsl_envs.sawyer_door import SawyerDoorGoalEnv
from eva.envs.gcsl_envs.lunarlander import LunarEnv
from eva.envs.gcsl_envs.claw_env import ClawEnv
from eva.envs.gcsl_envs import goal_env
from eva.envs.mazelab.env import EmptyMazeEnv, UMazeEnv, FourRoomEnv
from eva.envs.bitflip import BitFlippingGymEnv

class GymToGoalReaching(gym.ObservationWrapper):
    """Wrap a Gym env into a GoalReaching env.
       Need to provide goal_space, achieved_goal, desired_goal
    """

    def __init__(self, env):
        super(GymToGoalReaching, self).__init__(env)
        
        self.achieved_goal = None
        self.desired_goal = None

        assert self.env.observation_space["achieved_goal"] == self.env.observation_space["desired_goal"]
        self.goal_space = self.env.observation_space["achieved_goal"]

        self.observation_space = self.env.observation_space["observation"]

    def reset(self):
        observation = super(GymToGoalReaching, self).reset()
        
        return observation

    def step(self, action):
        """Take a step in the environment."""
        observation, reward, done, info = super(GymToGoalReaching, self).step(action)

        return observation, reward, done, info

    def observation(self, state):
        """Fetch the environment observation."""
        self.achieved_goal = state['achieved_goal']
        self.desired_goal = state['desired_goal']

        return state['observation']
    

class GCSLToGoalReaching(gym.ObservationWrapper):
    """Wrap a GCSL env into a GoalReaching env.
       Need to provide goal_space, achieved_goal, desired_goal
    """

    def __init__(self, env):
        super(GCSLToGoalReaching, self).__init__(env)
        
        self.achieved_goal = None
        self.desired_goal = None

    def reset(self):
        """Reset the environment and the desired goal"""
        desired_goal_state = self.env.sample_goal()
        self.desired_goal = self.env.extract_goal(desired_goal_state)

        observation = super(GCSLToGoalReaching, self).reset()
        
        return observation

    def step(self, action):
        """Take a step in the environment."""
        observation, reward, done, info = super(GCSLToGoalReaching, self).step(action)

        return observation, reward, done, info

    def observation(self, state):
        """Fetch the real environment observation and achieved goal"""
        self.achieved_goal = self.env.extract_goal(state)

        return self.env.observation(state)

def is_instance_gym_goal_env(env): 
    
    if not isinstance(env.observation_space, gym.spaces.Dict):
        return False

    if "achieved_goal" in env.observation_space.keys() and "desired_goal" in env.observation_space.keys():
        return True
    else:
        return False

def is_instance_gcsl_env(env):
    # Note that gym.make returns a gym wrapper instead of the real class
    # unwrapped removes the register wrapper
    if not isinstance(env.unwrapped, goal_env.GoalEnv):
        return False
    else:
        return True   

def is_instance_goalreaching_env(env):
    if isinstance(env, GCSLToGoalReaching) or isinstance(env, GymToGoalReaching):
        return True
    else:
        return False    

def create_env(env_id):
    env = gym.make(env_id)
    if is_instance_gym_goal_env(env):
        env = GymToGoalReaching(env)
    elif is_instance_gcsl_env(env):
        env = GCSLToGoalReaching(env)
    #else:
    #    print("The environment in neither a GCSL env nor a Gym goal env")
    return env	


