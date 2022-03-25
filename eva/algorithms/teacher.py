import random
import gym
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
from tqdm.notebook import tqdm
import os
from eva.replay_buffer.behavior_dataset import BehaviorDataset
from eva.replay_buffer.trajectory_replay_buffer import PrioritizedTrajectoryReplayBuffer
from eva.algorithms.common import *

class Teacher:
    def __init__(self, replay_buffer, env, agent, sparse_reward=True):
        self.replay_buffer = replay_buffer
        self.env = env
        self.agent = agent
        self.sparse_rewards = sparse_reward
   

    def select_episodes(self):
        pass

    def generate_command(self):
        pass

class UDRL(Teacher):
    # command: (tgt_horizon, tgt_return)
    # command_scale: (horizon_scale, return_scale)
    def scale_command(command, command_scale):
        """
        scaled command values (horizon, reward)
        """
        tgt_horizon, tgt_return = command
        horizon_scale, return_scale = command_scale
        tgt_horizon *= horizon_scale
        tgt_return *= return_scale

        return [tgt_horizon, tgt_return]

    # generate goal or target return for current training iteration
    def generate_iteration_command(self, tgt_horizon, replay_buffer):
        top_episodes = replay_buffer.top_episodes(LAST_FEW) # [(S,A,R,S_), ... ]
        tgt_horizon = int(np.mean([x[0].shape[0] for x in top_episodes]))
        tgt_reward_mean = np.mean([np.sum(x[2]) for x in top_episodes])
        tgt_reward_std = np.std([np.sum(x[2]) for x in top_episodes])

        tgt_horizon = min(tgt_horizon, 200)
        tgt_reward = round(np.random.random_sample()*tgt_reward_std + tgt_reward_mean, 0)
        
        return tgt_horizon, tgt_reward

    # generate goal or target return for current episode when collecting new data
    def generate_episode_command(self, tgt_horizon, replay_buffer):    

    # generate goal or target return for current state when collecting new data
    # could be HER or other methods
    def generate_step_command(self):
        aug_state = augment_state(state, 
                    command=(command_horizon, command_reward), 
                    command_scale=(HORIZON_SCALE, RETURN_SCALE)) 

        command_horizon = max(1, command_horizon-1)
                        if not done: 
                            episode['rewards'].append(0) # sparse lunar lander
                            command_reward -= 0
                        else:
                            episode['rewards'].append(episode_reward)     # sparse lunar lander 
                            command_reward -= episode_reward  

    def select_episodes(self):
        self.episodes_to_train = self.replay_buffer.sample_episodes(5)
        return self.episodes_to_train

    def construct_train_dataset(self, replay_buffer):
        episodes_to_train = self.select_episodes()
        train_dataset = BehaviorDataset(episodes_to_train, 
                                    size=BATCH_SIZE*NUM_UPDATES_PER_ITER, 
                                    horizon_scale=HORIZON_SCALE, 
                                    return_scale=RETURN_SCALE)
        train_dataset_loader = TorchDataLoader(train_dataset, 
                                    batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        return train_dataset_loader
    
    def reset_command(self):
        command_horizon, command_reward = generate_command(tgt_horizon, 
                                                        tgt_reward_mean, 
                                                        tgt_reward_std)
                    
        experiment.log_metric("command_horizon", command_horizon, step=playing_step)
        experiment.log_metric("command_reward", command_reward, step=playing_step)