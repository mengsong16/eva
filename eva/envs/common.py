import numpy as np
import random
import gym
import torch

from eva.envs.gcsl_envs.room_env import PointmassGoalEnv
from eva.envs.gcsl_envs.sawyer_push import SawyerPushGoalEnv
from eva.envs.gcsl_envs.sawyer_door import SawyerDoorGoalEnv
from eva.envs.gcsl_envs.lunarlander import LunarEnv
from eva.envs.gcsl_envs.claw_env import ClawEnv
from eva.envs.gcsl_envs import goal_env
from eva.envs.mazelab.env import EmptyMazeEnv, UMazeEnv, FourRoomEnv
from eva.envs.bitflip import BitFlippingGymEnv

class GymToGoalReaching(gym.ObservationWrapper):
    """Wrap a Gym env into a GoalReaching env.
    """

    def __init__(self, env):
        super(GymToGoalReaching, self).__init__(env)
        
        self.achieved_goal = None
        self.desired_goal = None

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
    
    def goal_space(self):
        assert self.env.observation_space["achieved_goal"] == self.env.observation_space["desired_goal"]
        return self.env.observation_space["achieved_goal"]

class GCSLToGoalReaching(gym.ObservationWrapper):
    """Wrap a GCSL env into a GoalReaching env.
    """

    def __init__(self, env):
        if not isinstance(env, goal_env.GoalEnv):
            print("Error: the environment needs to be a GCSL env")
            exit()
        
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

def create_env(env_type, env_id, fixed_start=True, fixed_goal=True):
    if env_type == "gym":
        env = gym.make(env_id)
    elif env_type == "gcsl":   
        if "Pointmass" in env_id: 
            if "Rooms" in env_id:
                env = PointmassGoalEnv(room_type='rooms', fixed_start=fixed_start, fixed_goal=fixed_goal) 
            elif "Wall" in env_id:
                env = PointmassGoalEnv(room_type='wall', fixed_start=fixed_start, fixed_goal=fixed_goal)
            else:
                env = PointmassGoalEnv(room_type='empty', fixed_start=fixed_start, fixed_goal=fixed_goal)        
        elif env_id == "SawyerPush": 
            env = SawyerPushGoalEnv(fixed_start=fixed_start, fixed_goal=fixed_goal) 
        elif env_id == "SawyerDoor": 
            env = SawyerDoorGoalEnv(fixed_start=fixed_start, fixed_goal=fixed_goal)
        elif env_id == "Lunar":
            env = LunarEnv(fixed_start=fixed_start, fixed_goal=fixed_goal)  
        elif env_id == "Claw":
            env = ClawEnv(fixed_start=fixed_start, fixed_goal=fixed_goal)              
        else:
            print("Error: undefined env: %s"%(env_id))
            exit()
    else:
        print("Error: undefined env: %s"%(env_id))
        exit()

    return env	


