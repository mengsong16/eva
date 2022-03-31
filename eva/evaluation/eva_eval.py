import random
import gym
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
from tqdm.notebook import tqdm
from eva.replay_buffer.trajectory_replay_buffer import PrioritizedTrajectoryReplayBuffer
from eva.policy.policy import RandomPolicy, Policy
from eva.algorithms.common import *
from eva.replay_buffer.behavior_dataset import BehaviorDataset
from eva.utils.data_utils import parse_config, get_device
from eva.utils.path import *
from eva.algorithms.teacher import UDRL
import wandb
import os
import datetime



class EVAEvaluator:
    def __init__(self, config_filename="udrl.yaml"):

        assert config_filename is not None, "needs config file to initialize trainer"
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        self.seed = int(self.config.get("seed"))

        self.env = gym.make(self.config.get("env_id")) 

        self.device = get_device(self.config)
        
        # seed everything
        seed_env(self.env, self.seed)
        seed_other(self.seed)

        self.sparse_reward = self.config.get("sparse_reward")

        self.agent = Policy(state_dim=int(np.prod(self.env.observation_space.shape)), 
                            action_space=self.env.action_space, 
                            target_dim=int(self.config.get("target_dim")),
                            deterministic=self.config.get("deterministic_policy"),
                            hidden_size=int(self.config.get("hidden_size")))


        # initialize an empty teacher
        if self.config.get("algorithm_name") == "udrl":
            self.teacher = UDRL(config=self.config, replay_buffer=None)
        else:
            print("Error: undefined teacher name: %s"%(self.config.get("algorithm_name")))
            exit()
        
        # load checkpoint
        self.load_checkpoint()
    

    def load_checkpoint(self):
        folder_name = self.config.get("eval_checkpoint_folder")
        folder_path = os.path.join(checkpoints_path, folder_name)
        checkpoint_number = int(self.config.get("eval_checkpoint_index"))
        checkpoint_path = os.path.join(folder_path, f"ckpt_{checkpoint_number}.pth")
        
        if not os.path.exists(checkpoint_path):
            print("Error: "+checkpoint_path+" does not exists.")
            exit()
        
        # load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("Loaded checkpoint at: "+str(checkpoint_path))

        # load agent network
        self.agent.load_state_dict(checkpoint)
        self.agent.to(self.device)
        print("Agent state loaded.")

    # evaluate one episode
    def evaluate_one_episode(self, target):
        # set agent to evaluation mode
        self.agent.eval()
        
        # reset env
        state = self.env.reset()
        done = False
        episode_return = 0

        # set initial target
        self.teacher.set_initial_step_target(target)

        while not done:
            target = self.teacher.get_current_step_target()
            aug_state = augment_state(state, target)
            # aug_state is numpy array
            action = self.agent.get_actions(aug_state)
           
            if isinstance(action, np.ndarray) and self.agent.discrete_action:
                action = action[0]

            state, reward, done, _ = self.env.step(action)

            episode_return += reward
            
            # get actual reward
            if not done: 
                if self.sparse_reward:
                    actual_reward = 0
                else:
                    actual_reward = reward  
            else:  
                if self.sparse_reward:        
                    actual_reward = episode_return    # finally add total episode reward
                else:
                    actual_reward = reward
            
            self.teacher.generate_next_step_target(actual_reward)		
        
        return episode_return
    
    def evaluate(self, target):
        target = np.array(target, dtype=np.float32)
        results = []
        num_eval_episodes = int(self.config.get("num_eval_episodes"))
        avg_episode_return = 0
        for i in range(num_eval_episodes):
            episode_return = self.evaluate_one_episode(target)
            results.append(episode_return)

            print('-----------------------------')
            print('Episode: %d'%(i))
            print("Target: %s"%(target))
            print("Return: %f"%(episode_return))
            print('-----------------------------')
            
            avg_episode_return += episode_return

        results = np.array(results, dtype=np.float32)
        print("-------------- Evaluation summary --------------")
        print("Number of episodes: %d"%(num_eval_episodes))
        print("Min return: %f"%(np.min(results, axis=0)))
        print("Mean return: %f"%(np.mean(results, axis=0)))
        print("Max return: %f"%(np.max(results, axis=0)))
        print("Std of return: %f"%(np.std(results, axis=0)))
        print("------------------------------------------------")

if __name__ == "__main__": 
    evaluator = EVAEvaluator()
    evaluator.evaluate(target=[100,100])