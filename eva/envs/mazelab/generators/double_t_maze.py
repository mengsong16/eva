import numpy as np
from eva.envs.mazelab.generators.utils import save_maze

def double_t_maze():
	x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
								[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
								[1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
								[1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
								[1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
								[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], 
								[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
								[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
								[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
								[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)


	env_id = 'double-tmaze-v0'

	save_maze(env_id, x)
	
	return x

if __name__ == "__main__":
	maze = double_t_maze()
	print(maze)
