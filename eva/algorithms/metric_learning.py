import random
import gym
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
from tqdm.notebook import tqdm
import os
from eva.replay_buffer.triplet_dataset import Triplet, TripletBuffer, TripletDataset
from eva.replay_buffer.trajectory_replay_buffer import Episode
from eva.algorithms.common import *

def construct_triplets(episodes: Episode):
    buffer = TripletBuffer()
    for episode in episodes:
        states = episode.get_one_episode_states()
        assert states.shape[0] > 2, "Each episode should at least include 3 states."
        episode_length = episode.len
        for i in range(episode_length-2):
            buffer.add_triplet(states[i], states[i+1], states[i+2])

    return buffer        

