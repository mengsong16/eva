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
    def __init__(self, config_filename="udrl.yaml"):

        assert config_filename is not None, "needs config file to initialize trainer"
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        self.seed = int(self.config.get("seed"))

        self.env = gym.make(self.config.get("env_id")) 

        self.log_to_wandb = self.config.get("log_to_wandb")
        
        # seed everything
        seed_env(self.env, self.seed)
        seed_other(self.seed)

        self.sparse_reward = self.config.get("sparse_reward")

        self.agent = Policy(state_dim=self.env.observation_space.shape[0], 
                            action_space=self.env.action_space, 
                            target_dim=int(self.config.get("target_dim")),
                            deterministic=self.config.get("deterministic_policy"),
                            hidden_size=int(self.config.get("hidden_size")))

        # init wandb
        if self.log_to_wandb:
            self.init_wandb()

        self.init_replay_buffer()

        self.teacher = UDRL(config=self.config, replay_buffer=self.replay_buffer)


    
    def init_replay_buffer(self):
        # initialize data collection counter
        self.collected_episodes = 0

        self.replay_buffer = PrioritizedTrajectoryReplayBuffer(
            size=int(self.config.get("replay_buffer_size")))
               
        # warm up replay buffer with random episodes
        if self.config.get("warmup_with_random_trajectories"):
            num_episode = int(self.config.get("num_warmup_episodes"))

            for i in range(num_episode):
                
                episode_return = collect_one_episode(self.env, RandomPolicy(self.env), 
                        self.replay_buffer, self.sparse_reward, 
                        teacher=None)
                # step counter and log 
                self.collected_episodes += 1
                if self.log_to_wandb:
                    wandb.log({"achieved_episode_return": episode_return}, step=self.collected_episodes)

        
    def init_wandb(self):
        project_name = 'eva'

        env_name = str(self.config.get("env_id"))
        if self.sparse_reward:
            env_name = env_name + "-sparse"
        algorithm_name = self.config.get("algorithm_name")
        group_name = f'{algorithm_name}-{env_name}'

        # group_name - 6 digit random number
        experiment_name = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
        # initialize this run under project xxx
        wandb.init(
            name=experiment_name.lower(),
            group=group_name.lower(),
            project=project_name.lower(),
            config=self.config,
            dir=os.path.join(root_path)
        )
    
    # ground_truth_actions: discrete: [B], continuous: [B,action_dim]
    def compute_loss(self, aug_states, ground_truth_actions):
        """Compute loss of self._learner on the expert_actions.
        """

        if self.loss_type == "mse":
            # predicted_actions: discrete: [B], continuous: [B,action_dim]
            predicted_actions = self.agent.forward(aug_states)

            #print("------------------------")
            # if ground_truth_actions.dim() < 2:
            #     ground_truth_actions = torch.unsqueeze(ground_truth_actions,1)
            #print(predicted_actions.shape)
            
            #print(ground_truth_actions.shape)

           # exit()
            predicted_actions = predicted_actions.float()
            #print(predicted_actions.dtype)
            #print(ground_truth_actions.dtype)
            #predicted_actions.requires_grad = True
            #assert predicted_actions.requires_grad == True and ground_truth_actions.requires_grad == False
            
            #print(torch.mean((ground_truth_actions - predicted_actions)**2))
            #exit()
            return torch.mean((ground_truth_actions - predicted_actions)**2)
        elif self.loss_type == "log_prob":
            # log_probs: [B] for both discrete or continuous actions
            log_probs = self.agent.get_log_probs(aug_states, ground_truth_actions)
            #print(log_probs.size())
            #exit()
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
            aug_states = behavior_batch['augmented_state']
            ground_truth_actions = behavior_batch['ground_truth_action']
            #print(aug_states.shape)
            #print(ground_truth_actions.shape)
            # aug_states: [B,aug_state_dim]
            # ground_truth_actions: discrete: [B], continuous: [B,action_dim]
            loss = self.compute_loss(aug_states, ground_truth_actions)
            loss.backward()
            self.optimizer.step()

            # step counter and log
            self.training_step += 1
            if self.log_to_wandb:
                wandb.log({"batch_loss": loss.cpu().detach()}, step=self.training_step)

    def train(self):
        # set loss function and optimizer
        self.loss_type = self.config.get("loss_type")
        if self.loss_type == "mse" and self.agent.discrete_action == True:
            print("Error: the loss function should not be mse when the action space is discrete")
            exit()

        self.optimizer = torch.optim.Adam(self.agent.parameters(), 
            lr=float(self.config.get("learning_rate")))

        # get targe type
        target_type = list(self.config.get("target_type"))    
        
        # initialize training step counter
        self.training_step = 0

        # train for N iterations
        num_train_iterations = int(self.config.get("num_train_iterations"))
        for i in range(num_train_iterations):
            print("-------------------------")
            print("Iteration %d start ..."%i)
            # teacher generate training set
            train_dataset_loader = self.teacher.construct_train_dataset()

            # train agent
            self.train_one_iteration(train_dataset_loader)
            
            # teacher generate exploratory targets for current iteration
            self.teacher.generate_iteration_target()
            
            # agent generate new episodes using latest policy network and exploratory targets
            num_episode_generate = int(self.config.get("num_new_episodes_per_iter"))
            for _ in range(num_episode_generate):
                # teacher generate episode target
                self.teacher.generate_episode_target()
                # agent collect one episode
                episode_return = collect_one_episode(self.env, self.agent, 
                        self.replay_buffer, self.sparse_reward, 
                        self.teacher)

                # step counter and log
                self.collected_episodes += 1
                if self.log_to_wandb:
                    wandb.log({"achieved_episode_return": episode_return}, step=self.collected_episodes)
                    
                    if "horizon" in target_type:
                        wandb.log({"target_episode_horizon": self.teacher.get_episode_target_horizon()}, step=self.collected_episodes)
                    if "target" in target_type:
                        wandb.log({"target_episode_return": self.teacher.get_episode_target_return()}, step=self.collected_episodes)

            print("Iteration %d done."%i)

        self.env.close()
        
        print("Training done.")

if __name__ == "__main__": 
    trainer = EVATrainer()
    trainer.train()