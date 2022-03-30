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
    def __init__(self, config, replay_buffer):
        self.config = config
        self.replay_buffer = replay_buffer


class UDRL(Teacher):
    def __init__(self, config, replay_buffer):
        super().__init__(config, replay_buffer)

        self.return_scale = float(self.config.get("return_scale"))
        self.horizon_scale = float(self.config.get("horizon_scale"))

    def scale_target(self, tgt_horizon, tgt_return):
        scaled_tgt_horizon = tgt_horizon * self.horizon_scale
        scaled_tgt_return = tgt_return * self.return_scale

        return [scaled_tgt_horizon, scaled_tgt_return]

    # generate target for current training iteration
    # for exploration
    def generate_iteration_target(self):
        top_episodes = self.replay_buffer.top_episodes(
            int(self.config.get("num_top_episodes"))) 
        
        # target horizon is the mean of top N episodes
        # x[0]: S, x[2]: R
        tgt_horizon = int(np.mean([x[0].shape[0] for x in top_episodes]))
        self.tgt_horizon = min(tgt_horizon, int(self.config.get("min_target_horizon")))
        
        # mean and std of target return of top N episodes
        self.tgt_return_std = np.std([np.sum(x[2]) for x in top_episodes])
        self.tgt_return_mean = np.mean([np.sum(x[2]) for x in top_episodes])

    
    # generate target for current episode when collecting new data
    # for exploration
    def generate_episode_target(self): 
        # sample target return from [mean, mean+std)
        # random_sample: sample from [0.0, 1.0)
        episode_tgt_return = np.random.random_sample() * self.tgt_return_std + self.tgt_return_mean
        # round to interger.0
        self.episode_tgt_return = round(episode_tgt_return, 0)

        self.episode_tgt_horizon = self.tgt_horizon 

        self.step_tgt_return = self.episode_tgt_return
        self.step_tgt_horizon = self.episode_tgt_horizon

    def get_episode_target_horizon(self):
        return self.episode_tgt_horizon

    def get_episode_target_return(self):
        return self.episode_tgt_return      
            
    # generate target for next state when collecting new data
    # for exploration
    def generate_next_step_target(self, reward):
        self.step_tgt_horizon = max(1, self.step_tgt_horizon-1)
        self.step_tgt_return -= reward  
    
    # for exploration
    # return a numpy array
    def get_current_step_target(self):
        scaled_target = self.scale_target(
            self.step_tgt_horizon, self.step_tgt_return)

        return np.asarray(scaled_target)    

    # for student learning
    def construct_train_dataset(self):
        # Randomly select episodes from replay buffer
        episodes_to_train = self.replay_buffer.sample_episodes(
            int(self.config.get("num_train_episodes_per_iter")))

        train_dataset = BehaviorDataset(config=self.config, 
            episodes=episodes_to_train, teacher=self)
        
        train_dataset_loader = TorchDataLoader(train_dataset, 
                                    batch_size=int(self.config.get("batch_size")), 
                                    shuffle=True, 
                                    num_workers=0)

        return train_dataset_loader  
    
    # get achieved target, could relable like HER or use other strategies
    # for student learning
    # R: reward array of this episode
    # return a numpy array
    def get_achieved_target(self, episode_len, start_index, R):
        # achieved target is the final state of the trajectory
        tgt_horizon = episode_len - start_index - 1
        tgt_return = np.sum(R[start_index:])
        scaled_target = self.scale_target(tgt_horizon, tgt_return)
        return np.asarray(scaled_target)