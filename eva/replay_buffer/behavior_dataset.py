import random
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from eva.algorithms.common import *

class BehaviorDataset(TorchDataset):
    """ Sample behavior segments for supervised learning 
    from given input episodes.
    """
    def __init__(self, episodes, size, horizon_scale, return_scale):
        super(BehaviorDataset, self).__init__()
        self.episodes = episodes
        self.horizon_scale = horizon_scale
        self.return_scale = return_scale
        self.size = size

    def __len__(self):
        # just returning a placeholder number for now
        return self.size

    def __getitem__(self, idx):
        # get episode
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        # randomly sample an episode
        episode = random.choice(self.episodes)
        S, A, R, S_ = episode

        # randomly extract a segment
        episode_len = S.shape[0]
        start_index = np.random.choice(episode_len - 1) # ensures cmd_steps >= 1
        command_horizon = (episode_len - start_index - 1)
        command_return = np.sum(R[start_index:])
        command = command_horizon, command_return
        command_scale = self.horizon_scale, self.return_scale

        # construct sample
        features = augment_state(
            S[start_index,:], command, command_scale
        )
        # ground truth action
        label = A[start_index]               
        sample = {
            'features': torch.tensor(features, dtype=torch.float), 
            'label': torch.tensor(label, dtype=torch.long) # categorical val
        }        
        return sample