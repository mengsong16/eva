import numpy as np
from skimage.draw import rectangle
from eva.envs.mazelab.generators.utils import save_maze

# Free space is the grid minus boundary
# Passage are free space minus obstacle rectangle
def u_maze(width, height, obstacle_width, obstacle_height):
	x = np.zeros([height, width], dtype=np.uint8)
	# boundary are walls
	x[0, :] = 1
	x[-1, :] = 1
	x[:, 0] = 1
	x[:, -1] = 1
	
	# [start_r, start_c]
	start = [height//2 - obstacle_height//2, 1]
	rr, cc = rectangle(start, extent=[obstacle_height, obstacle_width], shape=x.shape)
	x[rr, cc] = 1
	
	# opening is rightwards
	x = np.fliplr(x)

	env_id = 'umaze-v0'

	save_maze(env_id, x)

	return x

if __name__ == "__main__":
	maze = u_maze(12, 9, 8, 2)
	print(maze)