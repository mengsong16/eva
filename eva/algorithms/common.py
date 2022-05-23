import random
import gym
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
from tqdm.notebook import tqdm

from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from eva.utils.path import *
from eva.utils.data_utils import parse_config, get_device

# state is a numpy array: [B, state_dim]
# target is a numpy array: [B, target_dim]
# aug_state is a numpy array: [B, state_dim+target_dim]
def augment_state(state, target):
	""" Appends target to the original state vector.
	"""
	if state.ndim < 2:
		state = np.expand_dims(state, axis=0)
	if target.ndim < 2:
		target = np.expand_dims(target, axis=0)

	assert state.shape[0] == target.shape[0], "state and condtion should have the same shape in dimension 0"
	
	aug_state = np.concatenate((state, target), axis=1)
	
	return aug_state

# collect one episode for training (exploration)
def collect_one_episode(env, agent, replay_buffer, sparse_reward, teacher=None):
	episode = {
			'states': [],
			'actions': [],
			'rewards': [],
			'next_states': []
	}
	episode_return = 0

	# reset
	state = env.reset()
	done = False

	while not done:
		episode['states'].append(state)
		# augment state with target and pass it to the agent
		if teacher is not None:
			target = teacher.get_current_step_target()
			aug_state = augment_state(state, target)
			# aug_state is numpy array
			action = agent.get_actions(aug_state)
		else:
			# state is numpy array   
			action = agent.get_actions(state)

		if isinstance(action, np.ndarray) and agent.discrete_action:
			action = action[0]

		state, reward, done, _ = env.step(action)

		episode_return += reward
		episode['actions'].append(action)
		episode['next_states'].append(state)
		# get actual reward
		if not done: 
			if sparse_reward:
				actual_reward = 0
			else:
				actual_reward = reward  
		else:  
			if sparse_reward:        
				actual_reward = episode_return    # finally add total episode reward
			else:
				actual_reward = reward
		
		episode['rewards'].append(actual_reward)

		if teacher is not None:
			teacher.generate_next_step_target(actual_reward)		
	
	# add episode data to the replay buffer
	replay_buffer.add_episode(
		np.array(episode['states'], dtype=np.float32),
		np.array(episode['actions'], dtype=np.float32),
		np.array(episode['rewards'], dtype=np.float32),
		np.array(episode['next_states'], dtype=np.float32),
	)

	return episode_return

# episode = (S, A, R, S_)
# S,A,R,S_ are float numpy arrays
# return a float numpy array [T, state_dim]
def get_one_episode_states(episode):
	(S, A, R, S_) = episode
	a = np.append(S, np.expand_dims(S_[-1], axis=0), axis=0)

	return a

def seed_env(env: gym.Env, seed: int) -> None:
    """Set the random seed of the environment."""
    # if seed is None:
    #     seed = np.random.randint(2 ** 31 - 1)
    seed = int(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def seed_other(seed: int):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

# ========== for rlkit ====================
def yaml2variant(config_filename):
    config_file = os.path.join(config_path, config_filename)
    config = parse_config(config_file)
    #print("=========== Config ==============")
    #print(config)
    #print("=================================")
    return config

def run_experiment(config_filename, experiment):
    variant = yaml2variant(config_filename=config_filename)
    seed = variant["seed"]
    # experiment directory: based_log_dir/exp_prefix/exp_name 
    # exp_name = exp_prefix_timestamp--s-seed
    # log_dir: based_log_dir/exp_prefix
    # base_log_dir = conf.LOCAL_LOG_DIR by default, parameter for create_log_dir
    exp_prefix = ("%s-%s"%(variant["algorithm"], variant["env_id"])).lower()

    # set seed using our methods
    seed_other(seed)

    # set logger
    setup_logger(exp_prefix=exp_prefix, variant=variant, 
                base_log_dir=checkpoints_path, seed=seed)
    
    # set gpu mode
    ptu.set_gpu_mode(mode=True, gpu_id=variant["gpu_id"])  
    
    # start experiment
    experiment(variant)
