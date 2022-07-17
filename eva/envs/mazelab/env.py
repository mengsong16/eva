from eva.envs.mazelab.base_env import BaseMazeEnv
from eva.envs.mazelab.motion import VonNeumannMotion
from eva.envs.mazelab.maze import Maze

import gym
from gym import wrappers
from gym.spaces import Box, Dict
from gym.spaces import Discrete
from gym import spaces
import matplotlib.pyplot as plt

from eva.envs.mazelab.solvers import dijkstra_solver

from pathlib import Path
import os
import numpy as np

# BaseMazeEnv is already gym.Env and already implement seed
class MazeEnv(BaseMazeEnv):
    def __init__(self, env_id=None, start=None, goal=None, random_goal=True, random_start=True, max_episode_steps=200):
        super().__init__()
        
        # load maze
        m = self.load_maze(env_id)
        self.maze = Maze(m)

        #print(self.maze.objects)
        self.random_goal = random_goal
        self.random_start = random_start

        self.impassable_array = self.maze.to_impassable()
        self.motions = VonNeumannMotion()
    
        #if self.random_goal:
        # state: [height, width]
        self.observation_space = Dict(
            desired_goal=Box(low=np.array([0,0]), high=np.array([self.maze.size[0]-1,self.maze.size[1]-1]), shape=(2,), dtype=np.int8),
            achieved_goal=Box(low=np.array([0,0]), high=np.array([self.maze.size[0]-1,self.maze.size[1]-1]), shape=(2,), dtype=np.int8),
            observation=Box(low=np.array([0,0]), high=np.array([self.maze.size[0]-1,self.maze.size[1]-1]), shape=(2,), dtype=np.int8),
        )
        # else:    
        #     self.observation_space = Box(low=np.array([0,0]), high=np.array([self.maze.size[0],self.maze.size[1]]), shape=(2,), dtype=np.uint8)
        
        self.action_space = Discrete(len(self.motions))

        self.reward_for_achieving_goal = 0.0 #0.0
        self.step_reward_for_not_achieving_goal = -1.0

        self.max_episode_steps = max_episode_steps
        self.spec.max_episode_steps = max_episode_steps

        
        # set start and goal
        # check whether given start is valid
        if (not self.random_start) and (start is not None):
            self.start_pos = np.array(start)
            if self.is_impassable(self.start_pos):
                raise ValueError("Given starting location [%d, %d] is not in free space"%(self.start_pos[0], self.start_pos[1]))
            
            if not self.is_within_bound(self.start_pos):
                raise ValueError("Given starting location [%d, %d] is out of bounds"%(self.start_pos[0], self.start_pos[1]))  
        else:
            self.generate_start()
        
        # check whether given goal is valid
        if (not self.random_goal) and (goal is not None):
            goal_pos = np.array(goal)
            if self.is_impassable(goal_pos):
                raise ValueError("Given goal location [%d, %d] is not in free space"%(goal_pos[0], goal_pos[1]))

            if not self.is_within_bound(goal_pos):
                raise ValueError("Given goal location [%d, %d] is out of bounds"%(goal_pos[0], goal_pos[1]))
            
            self.maze.objects.goal.positions = [goal_pos]

            if self.is_start_goal_overlap():
                raise ValueError("Given start and goal should not overlap") 
        else:    
            self.generate_goal()  

        # reset robot
        self.reset_robot() 

        # get achieved goal
        self._achieved_goal = self.get_observation()    

        #print("------------------------")
        #print("Maze initialized")
        #print("------------------------")
        #self.print_start_goal()       
        
    # should return state, reward, done, info    
    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]

        stay = self.bounce_back(new_position)
        # move forward and update agent's position
        if not stay:
            self.maze.objects.agent.positions = [np.array(new_position)]
        
        self._achieved_goal = self.get_observation()

        # step count
        self.step_count += 1
        
        # compute rewards and check game over
        if self._is_success(achieved_goal=self.maze.objects.agent.positions[0], desired_goal=self.maze.objects.goal.positions[0]):
            reward = self.reward_for_achieving_goal
            done = True
            success = True
        else:
            reward = self.step_reward_for_not_achieving_goal
            if self.step_count >= self.max_episode_steps:
                done = True
            else:    
                done = False

            success = False    

        #if self.random_goal:
        return {"observation": self.get_observation().copy(),\
        "desired_goal": self.get_goal().copy(), "achieved_goal": self.get_achieved_goal().copy()}, float(reward), done, \
        {'is_success': success}    
        #{"action_mask": self.get_action_mask(), 'is_success': success}
        # else:    
        #     # return agent's current position as observation
        #     return self.get_observation().copy(), float(reward), done, {"action_mask": self.get_action_mask(), 'is_success': success}
        
    def reset(self):
        if self.random_start:   
            self.generate_start()

        if self.random_goal:
            self.generate_goal()

        self.reset_robot()    
        #self.print_start_goal()

        self._achieved_goal = self.get_observation()

        self.step_count = 0
        
        #if self.random_goal:
        return {"observation": self.get_observation().copy(),\
        "desired_goal": self.get_goal().copy(), "achieved_goal": self.get_achieved_goal().copy()}
        # else:    
        #     return self.get_observation().copy()   

    def reset_robot(self):
        self.maze.objects.agent.positions = [self.start_pos]

    def print_start_goal(self):
        print("Start: %s"%self.get_start())
        print("Goal: %s"%self.get_goal()) 
        print("Robot: %s"%self.get_observation()) 
        print("------------------------")

    # get action mask based on current state, float np array
    # 1 - illegal action, 0 - otherwise 
    # return np array   
    def get_action_mask(self): 
        # current_position is np array 
        current_position = self.maze.objects.agent.positions[0]

        action_n = len(self.motions)
        action_mask = np.zeros(action_n, dtype=float) 

        for i, motion in enumerate(self.motions):
            new_position = [current_position[0] + motion[0], current_position[1] + motion[1]] 
            # wall   
            if self.is_impassable(new_position):
                action_mask[i] = 1.0

        return action_mask

    # state: [h,w]([row, col]), return np array   
    def get_observation(self): 
        return np.array(self.maze.objects.agent.positions[0])   

    def is_within_bound(self, position):
        # true if cell is still within bounds after the move
        return position[0] >= 0 and position[1] >= 0 and position[0] < self.maze.size[0] and position[1] < self.maze.size[1] 

    def is_impassable(self, position): 
        return self.impassable_array[position[0]][position[1]]     

    def bounce_back(self, position):
        # true if should bounde back after the move
        return (not self.is_within_bound(position)) or self.is_impassable(position)
    
    # achieved_goal and desired_goal are numpy arrays 
    def _is_success(self, achieved_goal, desired_goal):
        if (achieved_goal == desired_goal).all():
            return True
        else:
            return False

    # achieved_goal and desired_goal are numpy arrays 
    def compute_reward(self, achieved_goal, desired_goal, info): 
        if (achieved_goal == desired_goal).all():
            return self.reward_for_achieving_goal
        else:
            return self.step_reward_for_not_achieving_goal    

    def is_start_goal_overlap(self):
        return self.start_pos[0] == self.maze.objects.goal.positions[0][0] and self.start_pos[1] == self.maze.objects.goal.positions[0][1]
    
    def get_image(self):
        return self.maze.to_rgb()

    def get_background_image(self):
        return self.maze.to_background_rgb()   

    def get_obstacle(self):
        return self.maze.to_impassable()
            
    def generate_start(self):
        while True:
            self.start_pos = np.array((np.random.randint(0, self.maze.size[0]), np.random.randint(0, self.maze.size[1]))) 

            if not self.is_impassable(self.start_pos):
                return
                
    def generate_goal(self):
        while True:
            goal_pos = np.array((np.random.randint(0, self.maze.size[0]), np.random.randint(0, self.maze.size[1])))
            self.maze.objects.goal.positions = [goal_pos]
            
            if (not self.is_start_goal_overlap()) and (not self.is_impassable(goal_pos)):
                return

    def load_maze(self, env_id):
        maze_file = str(env_id)+".npy"
        parent_dir = "/home/meng/eva/eva/envs/mazelab"
        file_path = os.path.join(parent_dir, "sample_maze", maze_file)

        if not os.path.exists(file_path):
            raise ValueError("Cannot find %s." % file_path)

        else:
            return np.load(file_path, allow_pickle=False, fix_imports=True)
       
    def shortest_path(self):
        return self.shortest_path_s_g(start=self.start_pos, goal=self.maze.objects.goal.positions[0])

    # start and goal are np arrays [row, col]    
    def shortest_path_s_g(self, start, goal):
        actions = dijkstra_solver(self.impassable_array, self.motions, start, goal)
        if actions is not None:
            return actions, len(actions)
        else:
            return [], 0
         
    # return np array    
    def get_start(self):
        return np.array(self.start_pos)

    # return np array 
    def get_goal(self):
        return np.array(self.maze.objects.goal.positions[0])

    # return np array
    def get_achieved_goal(self):
        return self._achieved_goal            

#------------------- Children mazes -------------------#
class EmptyMazeEnv(MazeEnv):
    def __init__(self, start=(1,10), goal=(7,10), random_goal=True, random_start=True, max_episode_steps=200):
        super(EmptyMazeEnv, self).__init__(env_id="empty-maze-v0", start=start, goal=goal, random_goal=random_goal, random_start=random_start, max_episode_steps=max_episode_steps)     

class UMazeEnv(MazeEnv):
    def __init__(self, start=(1,10), goal=(7,10), random_goal=True, random_start=True, max_episode_steps=200):
        super(UMazeEnv, self).__init__(env_id="umaze-v0", start=start, goal=goal, random_goal=random_goal, random_start=random_start, max_episode_steps=max_episode_steps)   

class SimRoomEnv(MazeEnv):
    def __init__(self, start=(1,10), goal=(7,10), random_goal=True, random_start=True, max_episode_steps=500):
        #super(SimRoomEnv, self).__init__(env_id="replica-room0", start=start, goal=goal, random_goal=random_goal, random_start=random_start, max_episode_steps=max_episode_steps)  
        super(SimRoomEnv, self).__init__(env_id="replica-apt1", start=start, goal=goal, random_goal=random_goal, random_start=random_start, max_episode_steps=max_episode_steps)    

class FourRoomEnv(MazeEnv):
    def __init__(self, start=(1,1), goal=(17,17), random_goal=True, random_start=True, max_episode_steps=200):
        super(FourRoomEnv, self).__init__(env_id="four-room-v0", start=start, goal=goal, random_goal=random_goal, random_start=random_start, max_episode_steps=max_episode_steps)

    def in_which_room(self, start, goal): 
        if self.in_room_1(start) and self.in_room_1(goal):
            return "same"
        elif self.in_room_2(start) and self.in_room_2(goal):
            return "same"    
        elif self.in_room_3(start) and self.in_room_3(goal):
            return "same"
        elif self.in_room_4(start) and self.in_room_4(goal):
            return "same"   
        #---------------------------------------------------    
        elif self.in_room_1(start) and self.in_room_2(goal):
            return "neighbor"
        elif self.in_room_1(start) and self.in_room_3(goal):
            return "neighbor"    
        elif self.in_room_2(start) and self.in_room_1(goal):
            return "neighbor"
        elif self.in_room_2(start) and self.in_room_4(goal):
            return "neighbor" 
        elif self.in_room_3(start) and self.in_room_1(goal):
            return "neighbor"
        elif self.in_room_3(start) and self.in_room_4(goal):
            return "neighbor"    
        elif self.in_room_4(start) and self.in_room_2(goal):
            return "neighbor"
        elif self.in_room_4(start) and self.in_room_3(goal):
            return "neighbor"
        #---------------------------------------------------
        elif self.in_room_1(start) and self.in_room_4(goal):
            return "diagonal"
        elif self.in_room_2(start) and self.in_room_3(goal):
            return "diagonal"    
        elif self.in_room_3(start) and self.in_room_2(goal):
            return "diagonal"
        elif self.in_room_4(start) and self.in_room_1(goal):
            return "diagonal"
        else:
            print("Error: wrong room check!")
            print(start)
            print(goal)
            return -1                

    def in_room_1(self, loc):
        if loc[0]>=0 and loc[0]<=8 and loc[1]>=0 and loc[1]<=8:
            return True
        # N door    
        elif loc[0]==6 and loc[1]==9:
            return True
        # W door    
        elif loc[0]==9 and loc[1]==5:
            return True        
        else:
            return False   

    def in_room_2(self, loc):
        if loc[0]>=0 and loc[0]<=8 and loc[1]>=10 and loc[1]<=18:
            return True
        # N door    
        elif loc[0]==6 and loc[1]==9:
            return True
        # E door    
        elif loc[0]==9 and loc[1]==14:
            return True    
        else:
            return False    

    def in_room_3(self, loc):
        if loc[0]>=10 and loc[0]<=18 and loc[1]>=0 and loc[1]<=8:
            return True
        # S door    
        elif loc[0]==15 and loc[1]==9:
            return True
        # W door    
        elif loc[0]==9 and loc[1]==5:
            return True    
        else:
            return False   

    def in_room_4(self, loc):
        if loc[0]>=10 and loc[0]<=18 and loc[1]>=10 and loc[1]<=18:
            return True
        # E door    
        elif loc[0]==9 and loc[1]==14:
            return True
        # S door    
        elif loc[0]==15 and loc[1]==9:
            return True    
        else:
            return False                 


if __name__ == "__main__":
    env = SimRoomEnv(start=(8,10), goal=(-1,-1), random_start=True, random_goal=True) 
    #env = EmptyMazeEnv(start=(1,10), goal=(7,10), random_start=False, random_goal=False) 
    #env = UMazeEnv(start=(1,1), goal=(6,4), random_start=True, random_goal=True)
    #env = FourRoomEnv(start=(1,1), goal=(6,4), random_start=False, random_goal=False)
    #env_id = "four-room-v0"
    #env = gym.make(env_id, start=(1,10), goal=(7,10), random_goal=True)
    actions, path_length = env.shortest_path()
    
    num_episodes = 20
    num_steps_per_epoch = 200

    
    for episode in range(100):
        print("***********************************")
        print('Episode: {}'.format(episode))
        step = 0
        env.reset()
        for _ in range(20):  # 10 seconds
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print('-----------------------------')
            print('action: %s'%(action))
            print('state: %s, %s, %s'%(obs["observation"].shape, obs["desired_goal"].shape, obs["achieved_goal"].shape))
            #print('state: %s'%(obs))
            print('reward: %f'%(reward))
            print('-------------------------------')
            
            env.render() 
            step += 1
            if done:
                break
                    
        print('Episode finished after {} timesteps.'.format(step))

    print('-----------------------------')
    print("Action space: %s"%(env.action_space)) 
    print("Observation space: %s"%(env.observation_space)) 
    print("Shortest path length: %d"%(path_length)) 
    print('-----------------------------') 
    
    env.close()
