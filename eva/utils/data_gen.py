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
from eva.replay_buffer.trajectory_replay_buffer import TrajectoryBuffer, Episode
from eva.replay_buffer.triplet_dataset import Triplet, TripletBuffer
from eva.utils.path import *

class TrajectoryDatasetGenerator:
    def __init__(self, env_id, max_episode_steps=None, seed=1):
        self.seed = seed
        self.env_id = env_id
        self.env = create_env(env_id)
        # reset max_episode_steps (max steps per episode)
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

        print("=================================================")
        print("Generation done: %s"%(filename))   
        print("Episode collected: %d"%(num_episodes))
        print("Episode no duplication: %d"%(len(trajectory_buffer))) 
        print("=================================================")

    def gen_train_test(self, train_num_episodes, eval_num_episodes):
        self.gen_random_trajectory_dataset(train_num_episodes, dataset_split="train")
        self.gen_random_trajectory_dataset(eval_num_episodes, dataset_split="eval")

class TripletDatasetGenerator:
    def __init__(self, trajectory_dataset_file="empty-maze-v0-random.pkl"):
        self.trajectory_buffer = TrajectoryBuffer()
        self.triplet_buffer = TripletBuffer()
        self.trajectory_dataset_file = trajectory_dataset_file

        trajectory_dataset_path = os.path.join(data_path, trajectory_dataset_file)
        with open(trajectory_dataset_path, 'rb') as file_handler:
            self.trajectory_buffer = pickle.load(file_handler)

    def gen_triplet_dataset(self):
        episodes = self.trajectory_buffer.get_all_episodes()
        for episode in episodes:
            states = episode.get_one_episode_states()
            num_states = states.shape[0]
            for i in range(num_states-2):
                self.triplet_buffer.add_triplet(states[i], states[i+1], states[i+2])
        
        filename = self.trajectory_dataset_file.replace("-random", "-triplet")
        # save buffer to disk
        # needs to open in binary mode
        with open(os.path.join(data_path, filename), 'wb') as file_handler:
            pickle.dump(self.triplet_buffer, file_handler)

        print("=================================================")
        print("Generation done: %s"%(filename))   
        print("Triplet generated (no duplication): %d"%(len(self.triplet_buffer)))
        print("=================================================")


def generate_trajectory_dataset(env_id='empty-maze-v0', num_episodes=100, max_episode_steps=100):
    generator = TrajectoryDatasetGenerator(env_id=env_id, max_episode_steps=max_episode_steps)
    generator.gen_random_trajectory_dataset(num_episodes=num_episodes)

def generate_triplet_dataset(trajectory_dataset_file="empty-maze-v0-random.pkl"):
    generator = TripletDatasetGenerator(trajectory_dataset_file=trajectory_dataset_file)
    generator.gen_triplet_dataset()

if __name__ == "__main__": 
    # 'empty-maze-v0' 12*9
    # 'umaze-v0' 12*9
    # 'four-room-v0' 19*19
    generate_trajectory_dataset(env_id='empty-maze-v0', 
            num_episodes=200, max_episode_steps=100)

    generate_triplet_dataset(trajectory_dataset_file="empty-maze-v0-random.pkl")
