import random
import gym
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
from tqdm.notebook import tqdm

# state is a numpy array
# command is a numpy array
def augment_state(state, command):
	""" Appends command to the original state vector.
	"""
	
	aug_state = np.append(state, command)
	return aug_state

# collect one episode for training
def collect_one_episode(env, agent, replay_buffer, sparse_reward, teacher=None):
	episode = {
			'states': [],
			'actions': [],
			'rewards': [],
			'next_states': []
	}
	episode_reward = 0

	# reset
	state = env.reset()
	done = False

	if teacher is not None:
		teacher.reset_command()
	
	while not done:
		episode['states'].append(state)
		# augment state with command and pass it to the agent
		if teacher is not None:
			command = teacher.generate_command()
			aug_state = augment_state(state, command)
			action = agent.get_actions(aug_state)
		else:    
			action = agent.get_actions(state)
		
		state, reward, done, _ = env.step(action)
		episode_reward += reward
		episode['actions'].append(action)
		episode['next_states'].append(state)
		if not done: 
			if sparse_reward:
				episode['rewards'].append(0)
			else:
				episode['rewards'].append(reward)  
		else:  
			if sparse_reward:        
				episode['rewards'].append(episode_reward)    # finally add total episode reward
			else:
				episode['rewards'].append(reward)
	
	# add episode data to the replay buffer
	replay_buffer.add_episode(
		np.array(episode['states'], dtype=np.float),
		np.array(episode['actions'], dtype=np.int),
		np.array(episode['rewards'], dtype=np.float),
		np.array(episode['next_states'], dtype=np.float),
	)

def seed_env(env: gym.Env, seed: int) -> None:
    """Set the random seed of the environment."""
    if seed is None:
        seed = np.random.randint(2 ** 31 - 1)
        
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def seed_other(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False