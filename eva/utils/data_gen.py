import sys
from pathlib import Path
import gym
import numpy as np
import os
import pickle
from eva.envs.common import *
from eva.algorithms.common import *
from eva.policy.policy import RandomPolicy
from eva.utils.data_utils import parse_config, get_device
from eva.replay_buffer.trajectory_replay_buffer import TrajectoryBuffer
from eva.utils.path import *

class TrajectoryDatasetGenerator:
    def __init__(self, env_id, max_episode_steps=None, seed=1):
        self.seed = seed
        self.env_id = env_id
        self.env = create_env(env_id)
        # reset max_episode_steps
        if hasattr(self.env.unwrapped, 'max_episode_steps') and max_episode_steps is not None:
            self.env.unwrapped.max_episode_steps = int(max_episode_steps)
            self.env.unwrapped.spec.max_episode_steps = int(max_episode_steps)
        
        self.agent = RandomPolicy(self.env)

        # seed everything
        seed_env(self.env, self.seed)
        seed_other(self.seed)

    def gen_random_trajectory_dataset(self, num_episodes, dataset_split=None):
        trajectory_buffer = TrajectoryBuffer()

        for i in range(num_episodes):
            collect_one_episode(self.env, self.agent, 
                trajectory_buffer, sparse_reward=False, teacher=None)

        trajectory_buffer.summary()

        if dataset_split is None:
            option = ""
        else:
            option = "-" + str(dataset_split).lower()    

        filename = str(self.env_id).lower() + "-random" + option + ".pkl"
        # save buffer to disk
        # needs to open in binary mode
        with open(os.path.join(data_path, filename), 'wb') as file_handler:
            pickle.dump(trajectory_buffer, file_handler)

        print("Generation done: %s"%(filename))    

    def gen_train_test(self, train_num_episodes, eval_num_episodes):
        self.gen_random_trajectory_dataset(train_num_episodes, dataset_split="train")
        self.gen_random_trajectory_dataset(eval_num_episodes, dataset_split="eval")

if __name__ == "__main__": 
    # 'empty-maze-v0', 'umaze-v0', 'four-room-v0'
    #generator = TrajectoryDatasetGenerator(env_id='empty-maze-v0', max_episode_steps=100) # 12*9
    #generator = TrajectoryDatasetGenerator(env_id='umaze-v0', max_episode_steps=100) # 12*9
    generator = TrajectoryDatasetGenerator(env_id='four-room-v0', max_episode_steps=200) # 19*19
    generator.gen_random_trajectory_dataset(num_episodes=100)

