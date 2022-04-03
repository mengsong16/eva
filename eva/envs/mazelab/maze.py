import numpy as np
from eva.envs.mazelab.base_maze import BaseMaze
from eva.envs.mazelab.object import Object
from eva.envs.mazelab.color_style import DeepMindColor as color


class Maze(BaseMaze):
    def __init__(self, m):
        super().__init__(m)
        
    
    def make_objects(self, m):
        free = Object('free', 0, color.free, False, False, False, np.stack(np.where(m == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, False, False, np.stack(np.where(m == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, True, False, []) # starting location is included
        goal = Object('goal', 3, color.goal, False, False, False, []) # could be multiple goals
        
        return free, obstacle, agent, goal   


