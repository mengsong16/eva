import numpy as np
from eva.envs.mazelab.generators.utils import save_maze

def empty_maze(width, height):
	x = np.zeros([height, width], dtype=np.uint8)
	# boundary are walls
	x[0, :] = 1
	x[-1, :] = 1
	x[:, 0] = 1
	x[:, -1] = 1

	env_id = 'empty-maze-v0'

	save_maze(env_id, x)

	return x

if __name__ == "__main__":
	maze = empty_maze(12, 9)
	print(maze)    