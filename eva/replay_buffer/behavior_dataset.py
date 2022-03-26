import random
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from eva.algorithms.common import *

class BehaviorDataset(TorchDataset):
    """ Sample behavior segments for supervised learning 
    from given input episodes.
    """
    def __init__(self, config, episodes, teacher):
        super(BehaviorDataset, self).__init__()
        self.episodes = episodes
        self.config = config
        self.size = int(config.get("batch_size")) * int(config.get("num_updates_per_iter"))
        self.teacher = teacher

    def __len__(self):
        # just returning a placeholder number for now
        return self.size

    def get_target(self):
        pass    

    def __getitem__(self, idx):
        # get episode
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        # randomly sample an episode
        episode = random.choice(self.episodes)
        S, A, R, S_ = episode

        # randomly select a state
        episode_len = S.shape[0]
        start_index = np.random.choice(episode_len - 1) # ensures cmd_steps >= 1
        
        # get achieved target
        target = self.teacher.get_achieved_target(episode_len, start_index, R)

        # construct a sample: (aug_state, gt_action)
        aug_state = augment_state(S[start_index,:], target)
        
        # ground truth action
        gt_action = A[start_index]
               
        sample = {
            'augmented_state': torch.tensor(aug_state, dtype=torch.float), 
            'ground_truth_action': torch.tensor(gt_action, dtype=torch.long) # categorical val
        }        
        return sample