from gym.envs.registration import register

import os
import numpy as np

def save_maze(env_id, maze):
	maze_file = str(env_id)+".npy"
	file_path = os.path.join(sample_maze_path, maze_file)
	if not os.path.exists(os.path.dirname(file_path)):
		raise ValueError("Cannot find the directory for %s." % file_path)

	else:
		np.save(file_path, maze, allow_pickle=False, fix_imports=True)