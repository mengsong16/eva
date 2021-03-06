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
from eva.envs.common import *
import wandb
import os
import datetime



class EVATrainer:
    def __init__(self, config_filename="udrl.yaml"):

        assert config_filename is not None, "needs config file to initialize trainer"
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        self.seed = int(self.config.get("seed"))

        #self.env = gym.make(self.config.get("env_id"))
        self.env = create_env(self.config.get("env_id")) 

        self.log_to_wandb = self.config.get("log_to_wandb")

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

        # set experiment name
        self.set_experiment_name()

        # init wandb
        if self.log_to_wandb:
            self.init_wandb()

        self.init_replay_buffer()

        if self.config.get("algorithm_name") == "udrl":
            self.teacher = UDRL(config=self.config, replay_buffer=self.replay_buffer)
        else:
            print("Error: undefined teacher name: %s"%(self.config.get("algorithm_name")))
            exit()
    
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
                    wandb.log({"achieved_episode_return": episode_return, 
                    "collected_episodes": self.collected_episodes})

    def set_experiment_name(self):
        self.project_name = 'eva'.lower()

        env_name = str(self.config.get("env_id"))
        if self.sparse_reward:
            env_name = env_name + "-sparse"
        algorithm_name = self.config.get("algorithm_name")
        self.group_name = (f'{algorithm_name}-{env_name}').lower()

        # experiment_name - YearMonthDay-HourMiniteSecond
        now = datetime.datetime.now()
        self.experiment_name = "s%d-"%(self.seed)+now.strftime("%Y%m%d-%H%M%S").lower() 

    def init_wandb(self):
        
        #random.randint(int(1e5), int(1e6) - 1)
        
        # initialize this run under project xxx
        wandb.init(
            name=self.experiment_name,
            group=self.group_name,
            project=self.project_name,
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
            #predicted_actions = predicted_actions.float()
            #assert predicted_actions.requires_grad == True and ground_truth_actions.requires_grad == False

            return torch.mean((ground_truth_actions - predicted_actions)**2)
        elif self.loss_type == "log_prob":
            # log_probs: [B] for both discrete and continuous actions
            log_probs = self.agent.get_log_probs(aug_states, ground_truth_actions)
            
            return -torch.mean(log_probs)
        else:
            print("Error: undefined loss type: %s"%(self.loss_type))
            exit()    

    # train for one iteration
    def train_one_iteration(self, train_dataset_loader):
        # switch model to correct device
        self.agent.to(self.device)
        # switch model to training mode
        self.agent.train()

        # this runs for num_updates_per_iter rounds
        for behavior_batch in train_dataset_loader: 
            self.agent.zero_grad()
            aug_states = behavior_batch['augmented_state']
            ground_truth_actions = behavior_batch['ground_truth_action']

            # switch input to correct device
            aug_states = aug_states.to(self.device)
            ground_truth_actions = ground_truth_actions.to(self.device)
            
            # aug_states: [B,aug_state_dim]
            # ground_truth_actions: discrete: [B], continuous: [B,action_dim]
            loss = self.compute_loss(aug_states, ground_truth_actions)
            loss.backward()
            self.optimizer.step()

            # step counter and log
            self.training_step += 1
            if self.log_to_wandb:
                wandb.log({"batch_loss": loss.cpu().detach(), "training_step": self.training_step})

    def wandb_define_metric(self, target_type):
        # define our custom x axis metric
        wandb.define_metric("training_step", hidden=True)  # cannot hide it in automatic plotting
        wandb.define_metric("collected_episodes", hidden=True) # cannot hide it in automatic plotting
        # set metric to different x axis
        wandb.define_metric("batch_loss", step_metric="training_step")
        wandb.define_metric("achieved_episode_return", step_metric="collected_episodes")
        if "horizon" in target_type:
            wandb.define_metric("target_episode_horizon", step_metric="collected_episodes")
        if "return" in target_type:
            wandb.define_metric("target_episode_return", step_metric="collected_episodes")

    def train(self):
        # set loss function and optimizer
        self.loss_type = self.config.get("loss_type")
        if self.loss_type == "mse" and self.agent.discrete_action == True:
            print("Error: the loss function should not be mse when the action space is discrete")
            exit()

        self.optimizer = torch.optim.Adam(self.agent.parameters(), 
            lr=float(self.config.get("learning_rate")))

        
        # define wandb metric and corresponding x axis
        if self.log_to_wandb:
            # get target type
            target_type = list(self.config.get("target_type"))
            self.wandb_define_metric(target_type)
        
        # initialize training step counter
        self.training_step = 0

        # train for N iterations
        num_train_iterations = int(self.config.get("num_train_iterations"))
        save_every_iterations = int(self.config.get("save_every_iterations"))
        
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
                    wandb.log({"achieved_episode_return": episode_return, 
                    "collected_episodes": self.collected_episodes})
                    
                    if "horizon" in target_type:
                        wandb.log({"target_episode_horizon": self.teacher.get_episode_target_horizon(), 
                        "collected_episodes": self.collected_episodes})
                    if "return" in target_type:
                        wandb.log({"target_episode_return": self.teacher.get_episode_target_return(), 
                        "collected_episodes": self.collected_episodes})

            print("Iteration %d done."%i)

            # save checkpoint
            if (i+1) % save_every_iterations == 0:
                self.save_checkpoint(checkpoint_number = int((i+1) // save_every_iterations))

        # save last checkpoint if haven't
        if num_train_iterations % save_every_iterations != 0:
            self.save_checkpoint(checkpoint_number = int(num_train_iterations // save_every_iterations +1))

        self.env.close()

        print("-------------------------")
        print("Training done.")
    
    # Save checkpoint with specified name
    def save_checkpoint(self, checkpoint_number):
        # only save agent weights
        checkpoint = self.agent.state_dict()
        folder_name = self.group_name + "-" + self.experiment_name
        folder_path = os.path.join(checkpoints_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        checkpoint_path = os.path.join(folder_path, f"ckpt_{checkpoint_number}.pth")
        torch.save(checkpoint, checkpoint_path)

        print(f"Checkpoint {checkpoint_number} saved.")

if __name__ == "__main__": 
    trainer = EVATrainer()
    trainer.train()