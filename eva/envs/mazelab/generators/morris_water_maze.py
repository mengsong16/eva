import numpy as np
from skimage.draw import circle
from eva.envs.mazelab.generators.utils import save_maze


def morris_water_maze(radius, platform_center, platform_radius):
	x = np.ones([2*radius, 2*radius], dtype=np.uint8)
	
	rr, cc = circle(radius, radius, radius - 1)
	x[rr, cc] = 0
	
	platform = np.zeros_like(x)
	rr, cc = circle(*platform_center, platform_radius)
	platform[rr, cc] = 3  # goal
	x += platform

	env_id = 'morris-water-maze-v0'
	save_maze(env_id, x)
	
	return x

if __name__ == "__main__":
	x = morris_water_maze(radius=20, platform_center=[15, 30], platform_radius=4)
	print(x)