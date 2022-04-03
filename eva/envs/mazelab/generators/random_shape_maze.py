import numpy as np
from skimage.draw import random_shapes
from eva.envs.mazelab.generators.utils import save_maze


def random_shape_maze(width, height, max_shapes, max_size, allow_overlap, shape=None):
	x, _ = random_shapes([height, width], max_shapes, max_size=max_size, multichannel=False, shape=shape, allow_overlap=allow_overlap)
	
	x[x == 255] = 0
	x[np.nonzero(x)] = 1
	
	# wall
	x[0, :] = 1
	x[-1, :] = 1
	x[:, 0] = 1
	x[:, -1] = 1

	env_id = 'random-shape-maze-v0'

	save_maze(env_id, x)

	
	return x

if __name__ == "__main__":
	x = random_shape_maze(width=50, height=50, max_shapes=50, max_size=8, allow_overlap=False, shape=None)
	print(x)