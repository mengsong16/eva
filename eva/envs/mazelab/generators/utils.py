import os
import numpy as np
from pathlib import Path

def save_maze(env_id, maze):
	#current_dir = Path(os.path.abspath(os.getcwd()))
	#parent_dir = current_dir.parent.absolute()
	parent_dir = "/home/meng/eva/eva/envs/mazelab"
	sample_maze_dir = os.path.join(parent_dir, "sample_maze")
	maze_file = str(env_id) + ".npy"
	file_path = os.path.join(sample_maze_dir, maze_file)
	if not os.path.exists(os.path.dirname(file_path)):
		raise ValueError("Cannot find the directory for %s." % file_path)
	else:
		np.save(file_path, maze, allow_pickle=False, fix_imports=True)