import numpy as np
from eva.envs.mazelab.generators.utils import save_maze

def four_room(width, height, vertical_wall, horizon_wall, N_door, S_door, W_door, E_door):

	x = np.zeros([height, width], dtype=np.uint8)
	# boundary are walls
	x[0, :] = 1
	x[-1, :] = 1
	x[:, 0] = 1
	x[:, -1] = 1

	# check validity
	if 0 < vertical_wall < width-1 and 0 < horizon_wall < height-1 and 0 < N_door < horizon_wall and horizon_wall < S_door < height-1 and 0 < W_door < vertical_wall and vertical_wall < E_door < width-1:

		# interior walls
		x[:, vertical_wall] = 1
		x[horizon_wall, :] = 1

		# doors
		x[N_door, vertical_wall] = 0
		x[S_door, vertical_wall] = 0
		x[horizon_wall, E_door] = 0
		x[horizon_wall, W_door] = 0

	else:
		raise ValueError("Interior wall or door location is wrong.")

	env_id = 'four-room-v0'

	save_maze(env_id, x)	

	return x

if __name__ == "__main__":
	maze = four_room(19, 19, 9, 9, 6, 15, 5, 14)
	print(maze)    