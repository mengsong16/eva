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
from eva.utils.data_utils import parse_config
from eva.utils.path import *
from eva.algorithms.teacher import UDRL
import wandb
import os


class EVATrainer:
    def __init__(self, config_filename="urdl.yaml"):

        assert config_filename is not None, "needs config file to initialize trainer"
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        self.seed = int(self.config.get("seed"))

        self.env = gym.make(self.config.get("env_id")) 
        
        # seed everything
        seed_env(self.env, self.seed)
        seed_other(self.seed)

        self.sparse_reward = self.config.get("sparse_reward")

        self.agent = Policy(state_dim=self.env.observation_space.shape[0], 
                            action_space=self.env.action_space, 
                            command_dim=int(self.config.get("command_dim")),
                            deterministic=self.config.get("deterministic_policy"),
                            hidden_size=int(self.config.get("hidden_size")))

        self.init_replay_buffer()

        self.teacher = UDRL()

        self.init_wandb()
    
    def init_replay_buffer(self):
        playing_step = 0

        self.replay_buffer = PrioritizedTrajectoryReplayBuffer(
            size=int(self.config.get("replay_buffer_size")),
            seed=self.seed)
               
        # warm up replay buffer with random episodes
        if self.config.get("warmup_with_random_trajectories"):
            num_episode = int(self.config.get("num_warmup_episodes"))
            for _ in tqdm(range(num_episode)):
                collect_one_episode(self.env, self.agent, 
                        self.replay_buffer, self.sparse_reward, 
                        teacher=None)
                
                playing_step += 1
                experiment.log_metric("episode_reward", episode_reward, step=playing_step)

        
    def init_wandb(self):
        project_name = 'eva'
        # initialize this run under project xxx
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project=project_name,
            config=variant,
            dir=os.path.join(root_path)
        )
    
    def compute_loss(self, aug_states, ground_truth_actions):
        """Compute loss of self._learner on the expert_actions.
        """

        if self.loss_type == "mse":
            predicted_actions = self.agent(aug_states)
            return torch.mean((ground_truth_actions - predicted_actions)**2)
        elif self.loss_type == "log_prob":
            log_probs = self.agent.get_log_probs(aug_states, ground_truth_actions)
            return -torch.mean(log_probs)
        else:
            print("Error: undefined loss type: %s"%(self.loss_type))
            exit()    

    # train for one iteration
    def train_one_iteration(self, train_dataset_loader):
        # switch model to training mode
        self.agent.train()

        # this runs for num_updates_per_iter rounds
        for behavior_batch in train_dataset_loader: 
            self.agent.zero_grad()
            aug_states = behavior_batch['features']
            ground_truth_actions = behavior_batch['label']
            loss = self.compute_loss(aug_states, ground_truth_actions)
            loss.backward()
            self.optimizer.step()

            training_step += 1
            experiment.log_metric("batch_loss", loss.cpu().detach(), step=training_step)

    def train(self):
        # set loss function and optimizer
        self.loss_type = self.config.get("loss_type")
        self.optimizer = torch.optim.Adam(self.agent.parameters(), 
            lr=float(self.config.get("learning_rate")))
        
        # start training
        num_iterations = int(self.config.get("num_iterations"))
        for _ in range(num_iterations):
            # train policy
            train_dataset_loader = self.teacher.construct_train_dataset()
            self.train_one_iteration(train_dataset_loader)
            

            # teacher generate target commands

            experiment.log_metric("tgt_reward_mean", tgt_reward_mean, step=playing_step)
            
            
            # generate new episode using latest policy network and generated commands
            num_episode_generate = int(self.config.get("num_episodes_per_iter"))
            for _ in range(num_episode_generate):
                # collect episode
                collect_one_episode(self.env, self.agent, 
                        self.replay_buffer, self.sparse_reward, 
                        self.teacher)

                playing_step += 1
                experiment.log_metric("episode_reward", episode_reward, step=playing_step)
    
        self.env.close()
        experiment.end()
        print("Training done.")