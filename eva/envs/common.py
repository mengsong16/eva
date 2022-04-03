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

# GCSL built in sample_goal method
def sample_goal(env):
    if isinstance(env, goal_env.GoalEnv):
        desired_goal_state = env.sample_goal()
        desired_goal = env.extract_goal(desired_goal_state)
        return desired_goal
    else:
        print("Error: the environment needs to be goal conditioned")
        exit()
